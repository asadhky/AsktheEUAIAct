"""Shared data model: section ids, chunks, and retrieval results.

A *section* is one article, recital or annex of the Act. A *chunk* is a retrievable piece of
one section. Users and the LLM cite sections, retrieval returns chunks, and evaluation compares
each chunk's section with the section ids a question expects.
"""

import re
from dataclasses import dataclass
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator
from pydantic_core import core_schema

Kind = Literal["article", "recital", "annex"]
# Maps a lower-case word to its typed kind, so parsing needs no cast and unknown kinds are rejected.
_KINDS: dict[str, Kind] = {kind: kind for kind in get_args(Kind)}

# Well-formed Roman numerals only ("IIII" and "IC" are rejected). The Act has annexes I to XIII.
_ROMAN = re.compile(r"(?=[MDCLXVI])M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})")
# Accepts "article:5", "Article 5", "annex : iii", ... The number is validated separately.
_SECTION_TEXT = re.compile(r"(article|recital|annex)\s*[:\s]\s*(\S+)", re.IGNORECASE)
_EXPECTED_FORM = "expected '<article|recital|annex>:<number>', for example 'article:5' or 'annex:III'"


def _canonical_number(kind: str, raw: str) -> str:
    """Articles and recitals: a positive integer without leading zeros. Annexes: upper-case Roman."""
    if kind == "annex":
        number = raw.upper()
        if not _ROMAN.fullmatch(number):
            raise ValueError(f"annex number must be a Roman numeral such as 'III', got {raw!r}")
        return number
    # Explicit [0-9]: \d would also accept non-ASCII digits.
    if not re.fullmatch(r"[0-9]+", raw) or int(raw) < 1:
        raise ValueError(f"{kind} number must be a positive integer, got {raw!r}")
    return str(int(raw))


@dataclass(frozen=True)
class SectionId:
    """Identifies one article, recital or annex, written `<kind>:<number>`.

    This is what the LLM cites, what the UI shows, and what `expected_ids` in the eval set
    contain. Instances are always in canonical form, so equal sections compare equal.
    `SectionId.parse` is the way in from text; it does not accept sub-references like
    "article:5(1)", because a section is the unit that is cited and matched.
    """

    kind: Kind
    number: str

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unknown section kind {self.kind!r}, {_EXPECTED_FORM}")
        if _canonical_number(self.kind, self.number) != self.number:
            raise ValueError(f"section number {self.number!r} is not in canonical form (e.g. '5', 'III')")

    @classmethod
    def parse(cls, text: str) -> "SectionId":
        """Parse and normalise, e.g. 'Article 5' -> article:5 and 'annex:iii' -> annex:III."""
        match = _SECTION_TEXT.fullmatch(text.strip())
        if match is None:
            raise ValueError(f"invalid section id {text!r}: {_EXPECTED_FORM}")
        kind = _KINDS.get(match.group(1).lower())
        if kind is None:  # only reachable through case-folding lookalike letters
            raise ValueError(f"invalid section id {text!r}: {_EXPECTED_FORM}")
        return cls(kind, _canonical_number(kind, match.group(2)))

    def __str__(self) -> str:
        return f"{self.kind}:{self.number}"

    @property
    def label(self) -> str:
        """Human-readable form used in prompts, embedded headers and the UI: 'Article 5'."""
        return f"{self.kind.capitalize()} {self.number}"

    # Lets pydantic models use SectionId as a field: strings are parsed (so a bad id in a YAML
    # file becomes a validation error) and it is written back out as its string form.
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate, serialization=core_schema.to_string_ser_schema()
        )

    @classmethod
    def _validate(cls, value: object) -> "SectionId":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls.parse(value)
        # Raising ValueError (not TypeError) is what turns this into a pydantic validation error.
        raise ValueError(f"section id must be a string like 'article:5', got {type(value).__name__}")


class Chunk(BaseModel):
    """One retrievable passage and the metadata needed to cite it (R1.8)."""

    model_config = ConfigDict(frozen=True)

    section_id: SectionId  # the parent section: used for hits and citations
    ordinal: int = Field(ge=1)  # 1-based position within the section
    title: str | None = Field(default=None, min_length=1)  # articles and annexes; None for recitals
    text: str  # body text only, as shown in the sources panel (no header)

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk text must not be blank")
        return value

    # Derived rather than stored, so they can never disagree with section_id and ordinal.
    # They are still written to chunks.jsonl, which keeps the file readable on its own.
    @computed_field
    @property
    def chunk_id(self) -> str:
        """Stable for a fixed source and chunking parameters, e.g. 'article:5#2'."""
        return f"{self.section_id}#{self.ordinal}"

    @computed_field
    @property
    def kind(self) -> Kind:
        return self.section_id.kind

    @computed_field
    @property
    def number(self) -> str:
        return self.section_id.number


class ScoredChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    # What the score means depends on the ranker (BM25, cosine, rank fusion, reranker), so it is
    # only comparable within one result. NaN is rejected: it would silently pass any threshold check.
    score: float = Field(allow_inf_nan=False)


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Best first. Never empty: the relevance gate reads the top-ranked chunk.
    chunks: list[ScoredChunk] = Field(min_length=1)
    # The input to the "not covered" gate (R3.7, R3.8): the reranker score of the top chunk for
    # hybrid+rerank, otherwise the dense cosine of the top chunk. The two are on different scales,
    # which is why there are two thresholds. NaN is rejected because `nan < threshold` is False,
    # so a NaN relevance would let a question through the gate.
    relevance: float = Field(allow_inf_nan=False)
    relevance_kind: Literal["cosine", "rerank"]
