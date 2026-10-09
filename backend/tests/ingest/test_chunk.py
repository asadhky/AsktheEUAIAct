import random
import re
from pathlib import Path

import pytest

from askact.ingest.chunk import CHUNKER_VERSION, MIN_TEXT_ROOM, ChunkError, chunk_act, chunk_section, chunking_params
from askact.ingest.parse import ParsedAct, Section, parse_act
from askact.models import Chunk, SectionId

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_act.html"
REAL = Path(__file__).resolve().parents[3] / "data" / "raw" / "ai-act-oj-2024-1689.html"
MAX = 1800


def nows(text: str) -> str:
    return re.sub(r"\s+", "", text)


def make(sid: str, blocks: tuple[str, ...], title: str | None = "Title") -> Section:
    parsed = SectionId.parse(sid)
    return Section(parsed, None if parsed.kind == "recital" else title, blocks)


def words(n: int, word: str = "word") -> str:
    return " ".join(f"{word}{i}" for i in range(n))


@pytest.fixture(scope="module")
def mini() -> ParsedAct:
    return parse_act(FIXTURE.read_bytes())


def assert_sound(section: Section, chunks: list[Chunk], max_chars: int) -> None:
    """The properties every chunking must have, whatever the input."""
    assert chunks
    assert [c.ordinal for c in chunks] == list(range(1, len(chunks) + 1))
    for c in chunks:
        assert c.section_id == section.section_id and c.title == section.title
        assert c.text.strip(), "empty chunk"
        assert len(c.embed_text) <= max_chars, f"{c.chunk_id} is {len(c.embed_text)} > {max_chars}"
    assert nows("".join(c.text for c in chunks)) == nows(section.text), "text was lost, duplicated or reordered"


# --- the embedded string (header + text) is built in one place ------------------------------------

def test_the_embed_string_is_header_then_text():
    article = Chunk(section_id=SectionId.parse("article:5"), ordinal=1, title="Prohibited AI practices", text="1. The following...")
    assert article.header == "Article 5 — Prohibited AI practices"
    assert article.embed_text == "Article 5 — Prohibited AI practices\n1. The following..."


def test_a_recital_header_has_no_title():
    recital = Chunk(section_id=SectionId.parse("recital:12"), ordinal=1, title=None, text="Text.")
    assert recital.header == "Recital 12"
    assert recital.embed_text == "Recital 12\nText."


def test_the_header_is_not_part_of_the_stored_text():
    chunk = chunk_section(make("article:5", ("One paragraph.",)), MAX)[0]
    assert chunk.text == "One paragraph."
    assert "Article 5" not in chunk.text


# --- packing ------------------------------------------------------------------------------------------

def test_a_short_section_is_a_single_chunk():
    section = make("article:2", ("1. First.", "2. Second.\n(a) point;\n(b) point."))
    chunks = chunk_section(section, MAX)
    assert len(chunks) == 1
    assert chunks[0].text == "1. First.\n2. Second.\n(a) point;\n(b) point."
    assert chunks[0].chunk_id == "article:2#1"


def test_blocks_are_packed_together_up_to_the_limit_and_never_split_when_they_fit():
    blocks = tuple(f"{n}. {words(30)}" for n in range(1, 9))  # each about 270 characters
    chunks = chunk_section(make("article:9", blocks), 700)
    assert len(chunks) > 1
    assert_sound(make("article:9", blocks), chunks, 700)
    for c in chunks:  # every chunk is made of whole blocks
        assert all(line in blocks for line in c.text.split("\n"))


def test_a_block_that_fits_alone_moves_to_the_next_chunk_instead_of_being_split():
    header_room = 400 - len("Article 9 — Title") - 1  # characters available for text at a limit of 400
    first = "a" * 250                    # fits
    middle = "b " * 100                  # about 200 characters: fits alone, but not after `first`
    middle = middle.strip()
    assert len(first) + 1 + len(middle) > header_room >= len(middle)
    chunks = chunk_section(make("article:9", (first, middle)), 400)
    assert [c.text for c in chunks] == [first, middle]   # moved whole to a new chunk, not split across two


def test_long_articles_are_split_and_every_chunk_keeps_the_article_number_and_title():
    blocks = tuple(f"{n}. {words(60)}" for n in range(1, 13))
    section = make("article:9", blocks, title="Obligations of providers")
    chunks = chunk_section(section, 1000)
    assert len(chunks) > 3
    assert {c.section_id for c in chunks} == {SectionId("article", "9")}
    assert {c.title for c in chunks} == {"Obligations of providers"}
    assert all(c.header == "Article 9 — Obligations of providers" for c in chunks)
    assert_sound(section, chunks, 1000)


def test_a_recital_that_fits_stays_whole():
    chunks = chunk_section(make("recital:7", (words(100),)), MAX)
    assert len(chunks) == 1
    assert chunks[0].title is None


def test_a_long_recital_is_split_at_sentence_boundaries():
    """The real recitals reach 4,443 characters: they are not all short."""
    sentences = [f"Sentence {n} says {words(20)}." for n in range(30)]
    section = make("recital:53", (" ".join(sentences),))
    chunks = chunk_section(section, 900)
    assert len(chunks) > 1
    assert_sound(section, chunks, 900)
    for c in chunks:
        assert c.text.startswith("Sentence ") and c.text.endswith(".")  # cut between sentences, not inside


def test_an_oversized_block_is_split_by_lines_first_so_sub_points_stay_whole():
    lines = [f"({chr(97 + n)}) {words(25)};" for n in range(10)]
    chunks = chunk_section(make("article:9", ("\n".join(lines),)), 700)
    assert len(chunks) > 1
    for c in chunks:
        assert all(line in lines for line in c.text.split("\n"))


def test_text_is_cut_at_semicolons_and_colons_inside_a_long_line():
    long_line = "; ".join(f"clause {n} {words(12)}" for n in range(20)) + "."
    section = make("article:9", (long_line,))
    chunks = chunk_section(section, 600)
    assert_sound(section, chunks, 600)
    assert all(c.text.rstrip().endswith((";", ".")) for c in chunks)


def test_a_sentence_longer_than_the_room_is_cut_between_words_not_inside_them():
    section = make("article:9", (words(300),))  # one sentence, no punctuation at all
    chunks = chunk_section(section, 500)
    assert len(chunks) > 3
    assert_sound(section, chunks, 500)
    original_words = set(section.text.split())
    for c in chunks:
        assert set(c.text.split()) <= original_words, f"{c.chunk_id} contains a word cut in half"
        assert "\n" not in c.text  # one run of text stays on one line


def test_sentences_from_one_line_stay_on_one_line():
    sentences = [f"Sentence {n} says {words(15)}." for n in range(40)]
    section = make("recital:9", (" ".join(sentences),))
    chunks = chunk_section(section, 900)
    assert len(chunks) > 1
    assert all("\n" not in c.text for c in chunks)
    assert " Sentence " in chunks[0].text  # joined by a space, not a line break


def test_a_unit_that_exactly_fills_the_room_fits_in_one_chunk():
    room = 400 - len("Article 9 — Title") - 1
    first, second = "a" * 100, "b" * (room - 100 - 1)  # 100 + newline + the rest == room exactly
    chunks = chunk_section(make("article:9", (first, second)), 400)
    assert len(chunks) == 1 and len(chunks[0].embed_text) == 400
    chunks = chunk_section(make("article:9", (first, second + "b")), 400)  # one character more
    assert len(chunks) == 2


def test_a_single_word_longer_than_the_room_is_cut_but_not_lost():
    section = make("article:9", ("see " + "x" * 2500 + " here",))
    chunks = chunk_section(section, 600)
    assert_sound(section, chunks, 600)
    assert "".join(c.text for c in chunks).replace(" ", "").replace("\n", "").count("x") == 2500


def test_no_overlap_between_chunks():
    blocks = tuple(f"{n}. unique-{n} {words(40)}" for n in range(1, 10))
    chunks = chunk_section(make("article:9", blocks), 800)
    for n in range(1, 10):
        assert sum(c.text.count(f"unique-{n} ") for c in chunks) == 1


def test_the_limit_applies_to_header_plus_text_so_a_long_annex_title_leaves_less_room():
    title = "T" * 150
    section = make("annex:XII", tuple(f"{n}. {words(25)}" for n in range(12)), title=title)
    chunks = chunk_section(section, 500)
    assert_sound(section, chunks, 500)
    assert max(len(c.embed_text) for c in chunks) <= 500
    assert max(len(c.text) for c in chunks) <= 500 - len(chunks[0].header) - 1


# --- the chunker version is part of the index hash ------------------------------------------------------

def test_the_chunking_parameters_are_the_size_limit_and_the_chunker_version():
    from askact.config import Settings

    params = chunking_params(Settings(_env_file=None, chunk_max_chars=900))
    assert params == {"max_chars": 900, "chunker_version": CHUNKER_VERSION}


def test_the_chunker_version_matches_the_chunks_it_produces(mini):
    """If this fails, the chunker's output changed. Bump CHUNKER_VERSION in ingest/chunk.py (so an
    index built by the old code is seen as stale, not silently mixed with the new), then update both
    values here. Changing the digest without bumping the version defeats the purpose."""
    import hashlib

    chunks = chunk_act(mini, MAX)
    digest = hashlib.sha256("\n".join(f"{c.chunk_id}\n{c.embed_text}" for c in chunks).encode()).hexdigest()
    assert (CHUNKER_VERSION, digest) == (1, "bcf2cff5439a2d44966941d49f7f2bc7d08669fd41e9948488295f80b45e904d")


# --- stable ids and determinism ---------------------------------------------------------------------

def test_chunk_ids_are_stable_across_runs():
    section = make("article:9", tuple(f"{n}. {words(60)}" for n in range(1, 13)))
    first, second = chunk_section(section, 1000), chunk_section(section, 1000)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]
    assert [c.chunk_id for c in first][:3] == ["article:9#1", "article:9#2", "article:9#3"]


def test_the_limit_changes_the_chunks_and_is_the_only_thing_that_does():
    section = make("article:9", tuple(f"{n}. {words(60)}" for n in range(1, 13)))
    assert len(chunk_section(section, 700)) > len(chunk_section(section, 1800))


# --- errors, not silent loss ------------------------------------------------------------------------

def test_a_section_without_text_is_an_error_not_silently_dropped():
    with pytest.raises(ChunkError, match="article:9 has no text"):
        chunk_section(make("article:9", ()), MAX)


def test_a_limit_too_small_for_the_header_is_an_error_naming_the_section():
    section = make("annex:XII", (words(10),), title="T" * 150)
    with pytest.raises(ChunkError, match=r"CHUNK_MAX_CHARS=200 is too small for annex:XII"):
        chunk_section(section, 200)


def test_the_smallest_workable_limit_is_accepted():
    section = make("recital:1", (words(80),))
    header = len("Recital 1")
    chunks = chunk_section(section, header + 1 + MIN_TEXT_ROOM)
    assert_sound(section, chunks, header + 1 + MIN_TEXT_ROOM)


# --- the fixture (a trimmed copy of the real page) ------------------------------------------------

def test_fixture_chunks_cover_every_section_in_document_order(mini):
    chunks = chunk_act(mini, MAX)
    seen = list(dict.fromkeys(str(c.section_id) for c in chunks))
    assert seen == [str(s.section_id) for s in mini.sections]


def test_the_long_fixture_article_yields_several_chunks_with_its_number_and_title(mini):
    article_5 = [c for c in chunk_act(mini, MAX) if str(c.section_id) == "article:5"]
    assert len(article_5) >= 6  # Article 5 is about 11,000 characters
    assert {c.title for c in article_5} == {"Prohibited AI practices"}
    assert {c.number for c in article_5} == {"5"}
    assert {c.kind for c in article_5} == {"article"}
    assert [c.chunk_id for c in article_5] == [f"article:5#{n}" for n in range(1, len(article_5) + 1)]


def test_short_fixture_sections_stay_whole(mini):
    chunks = chunk_act(mini, MAX)
    for sid in ("recital:1", "recital:23", "article:1", "annex:VI"):
        assert len([c for c in chunks if str(c.section_id) == sid]) == 1


def test_every_fixture_section_is_sound_at_several_limits(mini):
    for limit in (400, 700, 1200, MAX):
        for section in mini.sections:
            assert_sound(section, chunk_section(section, limit), limit)


def test_fixture_chunks_are_ordered_and_ids_unique(mini):
    chunks = chunk_act(mini, MAX)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


# --- a seeded randomized check of the invariants -------------------------------------------------------

def random_section(rng: random.Random) -> Section:
    vocabulary = ["the", "provider", "shall", "ensure", "AI", "(a)", "10^25", "Regulation", "(EU)", "2024/1689", "‘model’"]

    def sentence() -> str:
        text = " ".join(rng.choice(vocabulary) for _ in range(rng.choice([1, 3, 8, 25, 60])))
        return text + rng.choice([".", ";", ":", "?", ""])

    def line() -> str:
        return " ".join(sentence() for _ in range(rng.choice([1, 1, 2, 5, 12])))

    def block() -> str:
        return "\n".join(line() for _ in range(rng.choice([1, 1, 2, 4, 9])))

    blocks = tuple(block() for _ in range(rng.choice([1, 2, 5, 12])))
    if rng.random() < 0.1:
        blocks += ("x" * rng.choice([150, 700, 2500]),)  # a "word" longer than any limit
    kind = rng.choice(["article", "recital", "annex"])
    number = {"article": str(rng.randint(1, 113)), "recital": str(rng.randint(1, 180)), "annex": rng.choice(["I", "III", "XII"])}[kind]
    title = None if kind == "recital" else rng.choice(["Scope", "A fairly long title that goes on " * rng.choice([1, 3])])
    return Section(SectionId.parse(f"{kind}:{number}"), title, blocks)


def test_random_sections_always_satisfy_the_invariants():
    rng = random.Random(20241012)
    for _ in range(400):
        section = random_section(rng)
        limit = rng.choice([300, 450, 800, 1800])
        try:
            chunks = chunk_section(section, limit)
        except ChunkError:
            continue  # a title too long for a tiny limit is a legitimate refusal
        assert_sound(section, chunks, limit)


# --- the real document (runs only when the file is present) ------------------------------------------

real_source = pytest.mark.skipif(not REAL.exists(), reason=f"the full source is not present at {REAL}")


@real_source
def test_real_source_every_chunk_is_within_the_limit_and_nothing_is_lost():
    act = parse_act(REAL.read_bytes())
    for section in act.sections:
        assert_sound(section, chunk_section(section, MAX), MAX)


@real_source
def test_real_source_long_recitals_are_split_rather_than_left_oversize():
    """The design assumed recitals were short; 35 of the 180 are longer than the default limit."""
    act = parse_act(REAL.read_bytes())
    chunks = chunk_act(act, MAX)
    long_recitals = [s for s in act.recitals if len(s.text) > MAX]
    assert long_recitals, "expected some recitals longer than the limit"
    for s in long_recitals:
        assert len([c for c in chunks if c.section_id == s.section_id]) > 1


@real_source
def test_real_source_chunk_ids_are_unique_and_cover_all_306_sections():
    act = parse_act(REAL.read_bytes())
    chunks = chunk_act(act, MAX)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert {c.section_id for c in chunks} == {s.section_id for s in act.sections}
