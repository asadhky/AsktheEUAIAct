import dataclasses
import json

import pytest
from pydantic import BaseModel, ValidationError

from askact.models import Chunk, RetrievalResult, ScoredChunk, SectionId


# --- SectionId: round trip -----------------------------------------------------------

@pytest.mark.parametrize("text", ["article:5", "recital:12", "annex:III", "article:113", "annex:XIII"])
def test_canonical_ids_round_trip(text):
    section = SectionId.parse(text)
    assert str(section) == text
    assert SectionId.parse(str(section)) == section


def test_fields_of_a_parsed_id():
    section = SectionId.parse("annex:III")
    assert (section.kind, section.number) == ("annex", "III")


# --- SectionId: normalisation of messy input -----------------------------------------

@pytest.mark.parametrize("messy, canonical", [
    ("Article 5", "article:5"),
    ("  article:5  ", "article:5"),
    ("ARTICLE:5", "article:5"),
    ("article:05", "article:5"),       # leading zeros dropped
    ("article :  5", "article:5"),
    ("annex:iii", "annex:III"),
    ("Annex III", "annex:III"),
    ("annex : xiii", "annex:XIII"),
    ("recital 12", "recital:12"),
    ("Recital:012", "recital:12"),
    ("Article\u00a05", "article:5"),   # non-breaking space, as found in EUR-Lex text
    ("\tarticle:5\n", "article:5"),
])
def test_messy_forms_are_normalised(messy, canonical):
    assert str(SectionId.parse(messy)) == canonical


# --- SectionId: rejection ------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "",
    "   ",
    "article",
    "article:",
    "5",
    "article5",                 # needs a separator
    "chapter:3",                # unknown kind
    "article:0",
    "article:-1",
    "article:+5",
    "article:5.1",
    "article:5(1)",             # sub-references are not sections
    "article:five",
    "article:\u0665",           # Arabic-Indic digit five: must not be read as 5
    "recital:0",
    "annex:5",                  # annexes are Roman
    "annex:",
    "annex:IIII",               # malformed numerals
    "annex:IC",
    "annex:VX",
    "article:5 extra",
    "article:5\narticle:6",
    "article:5, article:6",
    "ARTI\u0307CLE:5",          # lookalike letters
    "ARTİCLE:5",
])
def test_invalid_ids_are_rejected(bad):
    with pytest.raises(ValueError):
        SectionId.parse(bad)


def test_rejection_message_says_what_was_expected():
    with pytest.raises(ValueError, match=r"<article\|recital\|annex>:<number>"):
        SectionId.parse("chapter:3")


@pytest.mark.parametrize("kind, number", [
    ("article", "05"), ("article", "0"), ("article", "x"),
    ("annex", "iii"), ("annex", "5"), ("annex", ""),
    ("chapter", "1"), ("Article", "5"),
])
def test_direct_construction_requires_canonical_form(kind, number):
    with pytest.raises(ValueError):
        SectionId(kind, number)


# --- SectionId: value semantics ------------------------------------------------------

def test_equal_sections_compare_equal_and_hash_the_same():
    a, b = SectionId.parse("Article 5"), SectionId("article", "5")
    assert a == b and hash(a) == hash(b)
    assert len({a, b, SectionId.parse("article:6")}) == 2


def test_different_kinds_are_different_sections():
    assert SectionId.parse("article:5") != SectionId.parse("recital:5")


def test_section_ids_are_immutable():
    section = SectionId.parse("article:5")
    with pytest.raises(dataclasses.FrozenInstanceError):
        section.number = "6"  # type: ignore[misc]


@pytest.mark.parametrize("text, label", [
    ("article:5", "Article 5"),
    ("recital:12", "Recital 12"),
    ("annex:III", "Annex III"),
])
def test_label_is_human_readable(text, label):
    assert SectionId.parse(text).label == label


# --- SectionId inside pydantic models (used by the eval schema) ----------------------

class Holder(BaseModel):
    ids: list[SectionId]


def test_pydantic_parses_and_normalises_strings():
    holder = Holder.model_validate({"ids": ["Article 5", "annex:iii", "recital 7"]})
    assert holder.ids == [SectionId("article", "5"), SectionId("annex", "III"), SectionId("recital", "7")]


def test_pydantic_serialises_back_to_the_canonical_string():
    holder = Holder.model_validate({"ids": ["Article 5", "annex:iii"]})
    assert json.loads(holder.model_dump_json()) == {"ids": ["article:5", "annex:III"]}
    assert holder.model_dump(mode="json") == {"ids": ["article:5", "annex:III"]}


@pytest.mark.parametrize("bad_item", ["chapter:3", "article:0", "", 5, None, 5.0, ["article:5"]])
def test_pydantic_rejects_bad_entries_with_a_validation_error(bad_item):
    with pytest.raises(ValidationError):
        Holder.model_validate({"ids": ["article:1", bad_item]})


def test_pydantic_error_points_at_the_bad_entry():
    with pytest.raises(ValidationError) as excinfo:
        Holder.model_validate({"ids": ["article:1", "article:oops"]})
    assert excinfo.value.errors()[0]["loc"] == ("ids", 1)


# --- Chunk ---------------------------------------------------------------------------

def make_chunk(**overrides) -> Chunk:
    fields = dict(section_id="article:5", ordinal=2, title="Prohibited AI practices", text="1. The following ...")
    return Chunk(**{**fields, **overrides})


def test_chunk_id_is_section_id_plus_ordinal():
    assert make_chunk().chunk_id == "article:5#2"
    assert make_chunk(section_id="annex:III", ordinal=1).chunk_id == "annex:III#1"


def test_chunk_exposes_its_parent_section():
    chunk = make_chunk()
    assert chunk.section_id == SectionId.parse("article:5")
    assert (chunk.kind, chunk.number) == ("article", "5")


def test_chunk_accepts_a_messy_section_id_and_normalises_it():
    chunk = make_chunk(section_id="Article 05")
    assert chunk.section_id == SectionId("article", "5")
    assert chunk.chunk_id == "article:5#2"


def test_recital_chunk_has_no_title():
    chunk = make_chunk(section_id="recital:12", ordinal=1, title=None)
    assert chunk.title is None
    assert chunk.chunk_id == "recital:12#1"


def test_chunks_of_one_section_share_the_parent_but_not_the_id():
    first, second = make_chunk(ordinal=1), make_chunk(ordinal=2)
    assert first.section_id == second.section_id
    assert first.chunk_id != second.chunk_id


def test_chunk_json_line_is_self_describing():
    line = json.loads(make_chunk().model_dump_json())
    assert line == {
        "section_id": "article:5",
        "ordinal": 2,
        "title": "Prohibited AI practices",
        "text": "1. The following ...",
        "chunk_id": "article:5#2",
        "kind": "article",
        "number": "5",
    }


def test_chunk_survives_a_json_round_trip():
    chunk = make_chunk()
    assert Chunk.model_validate_json(chunk.model_dump_json()) == chunk


def test_derived_fields_in_json_cannot_override_the_real_ones():
    """chunk_id, kind and number are computed; a tampered file must not be able to disagree."""
    line = json.loads(make_chunk().model_dump_json())
    line.update(chunk_id="article:9#9", kind="recital", number="99")
    chunk = Chunk.model_validate(line)
    assert (chunk.chunk_id, chunk.kind, chunk.number) == ("article:5#2", "article", "5")


@pytest.mark.parametrize("overrides", [
    {"ordinal": 0},
    {"ordinal": -1},
    {"text": ""},
    {"text": "   \n"},
    {"title": ""},
    {"section_id": "chapter:1"},
    {"section_id": 5},
])
def test_invalid_chunks_are_rejected(overrides):
    with pytest.raises(ValidationError):
        make_chunk(**overrides)


def test_chunks_are_immutable():
    chunk = make_chunk()
    with pytest.raises(ValidationError):
        chunk.text = "changed"


# --- ScoredChunk and RetrievalResult --------------------------------------------------

def test_scored_chunk_allows_negative_scores():
    """Cross-encoder reranker scores can be negative."""
    assert ScoredChunk(chunk=make_chunk(), score=-3.2).score == -3.2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_scored_chunk_rejects_non_finite_scores(bad):
    with pytest.raises(ValidationError):
        ScoredChunk(chunk=make_chunk(), score=bad)


def test_retrieval_result_keeps_the_ranking_and_the_relevance_kind():
    ranked = [ScoredChunk(chunk=make_chunk(ordinal=i), score=1.0 / i) for i in (1, 2, 3)]
    result = RetrievalResult(chunks=ranked, relevance=0.71, relevance_kind="cosine")
    assert [c.chunk.ordinal for c in result.chunks] == [1, 2, 3]
    assert (result.relevance, result.relevance_kind) == (0.71, "cosine")


def test_retrieval_result_accepts_a_negative_rerank_relevance():
    result = RetrievalResult(
        chunks=[ScoredChunk(chunk=make_chunk(), score=-1.0)], relevance=-4.5, relevance_kind="rerank"
    )
    assert result.relevance == -4.5


def test_retrieval_result_requires_at_least_one_chunk():
    with pytest.raises(ValidationError):
        RetrievalResult(chunks=[], relevance=0.5, relevance_kind="cosine")


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_retrieval_result_rejects_non_finite_relevance(bad):
    """`nan < threshold` is False, so a NaN relevance would slip past the not-covered gate."""
    one = [ScoredChunk(chunk=make_chunk(), score=1.0)]
    with pytest.raises(ValidationError):
        RetrievalResult(chunks=one, relevance=bad, relevance_kind="cosine")


def test_retrieval_result_rejects_an_unknown_relevance_kind():
    one = [ScoredChunk(chunk=make_chunk(), score=1.0)]
    with pytest.raises(ValidationError):
        RetrievalResult(chunks=one, relevance=0.5, relevance_kind="bm25")  # type: ignore[arg-type]
