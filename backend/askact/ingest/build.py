"""Ingestion, end to end: get the source, parse, chunk, embed, and write the index (R1, R2.6, R2.7).

Nothing is written to the index directory unless every step succeeds, and an index that is already
up to date is left alone and not re-embedded.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import httpx

from askact.config import Settings
from askact.embedders import Embedder, EmbedderError, embedder_name, get_embedder
from askact.index import (
    INDEX_DIR,
    SCHEMA_VERSION,
    IndexCounts,
    IndexUnavailableError,
    Manifest,
    SourceRecord,
    compute_build_hash,
    load_index,
    write_index,
)
from askact.ingest.chunk import chunk_act, chunking_params
from askact.ingest.fetch import RAW_DIR, SourceFile, get_source
from askact.ingest.parse import Counts, ParsedAct, parse_act, verify, verify_counts

log = logging.getLogger(__name__)

# What the pinned document is. The sha256 of the file that was actually used is recorded next to it.
REGULATION = "Regulation (EU) 2024/1689"
CELEX = "32024R1689"
OJ_REFERENCE = "OJ L, 2024/1689, 12.7.2024"  # as printed in the header of the document itself


@dataclass(frozen=True)
class BuildResult:
    status: Literal["built", "up_to_date"]
    manifest: Manifest
    index_dir: Path


def retrieval_date(source: SourceFile, today: date | None = None) -> date:
    """The day the source was obtained: today if just downloaded, else the file's own date."""
    if source.origin == "fetched":
        return today or datetime.now(UTC).date()
    return datetime.fromtimestamp(source.path.stat().st_mtime, UTC).date()


def _source_record(settings: Settings, source: SourceFile, today: date | None) -> SourceRecord:
    return SourceRecord(
        regulation=REGULATION,
        celex=CELEX,
        oj_reference=OJ_REFERENCE,
        url=settings.source_url,
        retrieved_at=retrieval_date(source, today),
        sha256=source.sha256,
        origin=source.origin,
    )


def _counts(act: ParsedAct, chunk_total: int) -> IndexCounts:
    return IndexCounts(
        recitals=len(act.recitals), articles=len(act.articles), annexes=len(act.annexes), chunks=chunk_total
    )


def build_index(
    settings: Settings,
    *,
    force: bool = False,
    source_file: Path | None = None,
    raw_dir: Path = RAW_DIR,
    index_dir: Path = INDEX_DIR,
    client: httpx.Client | None = None,
    expected_counts: Counts | None = None,
    embedder: Embedder | None = None,
    today: date | None = None,
) -> BuildResult:
    """Build the index unless an up-to-date one exists. Raises on any failure, writing nothing.

    Raises SourceError, ParseError, ChunkError, EmbedderError or OSError; the CLI turns them into a
    message and a non-zero exit code. `force` re-fetches the source (it does not force a re-embed:
    the same source with the same settings gives the same hash). The last four parameters exist
    for tests: `expected_counts` replaces the pinned counts, which is how a trimmed document is
    checked, and then only the counts are verified, not the gap-free numbering.
    """
    source = get_source(settings, force=force, source_file=source_file, raw_dir=raw_dir, client=client)
    act = parse_act(source.path.read_bytes())
    if expected_counts is None:
        verify(act)  # the pinned document: its counts and numbering without gaps; raises ParseError
    else:
        verify_counts(act, expected_counts)  # a trimmed test document: its sections keep their real numbers

    try:
        existing = load_index(settings, index_dir, source_sha256=source.sha256)
    except IndexUnavailableError as problem:
        log.info("Building the index (%s).", problem.kind)
    else:
        log.info("The index in %s is already up to date; nothing to do.", index_dir)
        return BuildResult("up_to_date", existing.manifest, index_dir)

    chunks = chunk_act(act, settings.chunk_max_chars)
    embedder = embedder or get_embedder(settings)
    if embedder.name != embedder_name(settings):
        # The hash uses the name from the settings; recording another would make every index look stale.
        raise EmbedderError(f"the embedder is called {embedder.name!r} but the settings say {embedder_name(settings)!r}")
    log.info("Embedding %d chunks with %s ...", len(chunks), embedder.name)
    embeddings = embedder.embed_documents([chunk.embed_text for chunk in chunks])

    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        source=_source_record(settings, source, today),
        counts=_counts(act, len(chunks)),
        chunking=chunking_params(settings),
        embedding_model=embedder.name,
        dimension=embedder.dimension,
        build_hash=compute_build_hash(source.sha256, chunking_params(settings), embedder.name),
    )
    write_index(index_dir, manifest=manifest, chunks=chunks, embeddings=embeddings)
    return BuildResult("built", manifest, index_dir)
