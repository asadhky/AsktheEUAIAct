"""Ingestion step 3: split sections into bounded-size chunks that carry their metadata (R1.8, R1.9).

Every chunk sits inside exactly one article, recital or annex, so a citation names exactly the
section a chunk came from. Within a section, whole blocks (a numbered paragraph with its
sub-points, one definition, ...) are packed greedily into chunks; a block that is too big on its
own is split at line, sentence and finally word boundaries. There is no overlap: blocks are
self-contained, and overlapping text would be duplicated under two citations.

The size limit is on the *embedded string* (header + text, see `Chunk.embed_text`), because that is
what the embedding model reads. It is a character count, only a proxy for the model's token limit;
a slow test checks it against the real tokenizer.
"""

import re
from collections.abc import Iterable

from askact.ingest.parse import ParsedAct, Section
from askact.models import Chunk

# If less than this is left for text after the header, CHUNK_MAX_CHARS is too small for the section
# (Annex XII has a 215-character header) and chunks would degenerate to a few words.
MIN_TEXT_ROOM = 100

# Sentence-like boundaries. Legal text also puts ';' between list items, which makes a good cut.
_BOUNDARY = re.compile(r"(?<=[.;?!:])\s+")

# (separator to put before this unit when it follows another unit in the same chunk, text)
_Unit = tuple[str, str]


class ChunkError(ValueError):
    """A section cannot be chunked (no text, or CHUNK_MAX_CHARS is too small for its header)."""


def _line_units(line: str, room: int) -> list[_Unit]:
    """Pieces of one over-long line, each at most `room` characters.

    Sentences where they fit; otherwise words; and, as a last resort for a single word longer
    than the room (a long URL, say), fixed-size cuts. The first piece starts a new line.
    """
    units: list[_Unit] = []
    for sentence in _BOUNDARY.split(line):
        pieces = [sentence] if len(sentence) <= room else sentence.split()
        for piece in pieces:
            cuts = [piece[i : i + room] for i in range(0, len(piece), room)]
            for n, cut in enumerate(cuts):
                units.append(("" if n else (" " if units else "\n"), cut))
    return units


def _block_units(block: str, room: int) -> list[_Unit]:
    if len(block) <= room:
        return [("\n", block)]
    units: list[_Unit] = []
    for line in block.split("\n"):
        units += [("\n", line)] if len(line) <= room else _line_units(line, room)
    return units


def _pack(units: Iterable[_Unit], room: int) -> list[str]:
    """Greedily fill chunks with consecutive units. Every unit already fits in `room` by itself."""
    chunks: list[str] = []
    current = ""
    for separator, text in units:
        if current and len(current) + len(separator) + len(text) <= room:
            current += separator + text
        else:
            if current:
                chunks.append(current)
            current = text
    if current:
        chunks.append(current)
    return chunks


def chunk_section(section: Section, max_chars: int) -> list[Chunk]:
    """The chunks of one section, in order, numbered from 1. Each embed string is at most `max_chars`."""
    if not section.blocks:
        # Dropping it silently would delete a whole article from the index.
        raise ChunkError(f"{section.section_id} has no text to chunk")
    header = Chunk(section_id=section.section_id, ordinal=1, title=section.title, text="x").header
    room = max_chars - len(header) - 1  # 1 for the newline between header and text
    if room < MIN_TEXT_ROOM:
        raise ChunkError(
            f"CHUNK_MAX_CHARS={max_chars} is too small for {section.section_id}: its header "
            f"({len(header)} characters) leaves {room} for text, and at least {MIN_TEXT_ROOM} are needed"
        )
    units = (unit for block in section.blocks for unit in _block_units(block, room))
    return [
        Chunk(section_id=section.section_id, ordinal=n, title=section.title, text=text)
        for n, text in enumerate(_pack(units, room), start=1)
    ]


def chunk_act(act: ParsedAct, max_chars: int) -> list[Chunk]:
    """All chunks of the Act, in document order (recitals, articles, annexes)."""
    return [chunk for section in act.sections for chunk in chunk_section(section, max_chars)]
