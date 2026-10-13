"""The answer pipeline: from a retrieval result to a grounded, cited answer, or "not covered" (R3.1 to R3.8).

Everything that is free runs before anything that costs money, in this order:

  1. relevance gate   below the threshold for this kind of score: "not covered", no LLM call, no reservation
  2. prompt           passages labelled by section id (generation.prompt)
  3. reserve spend    refused if the daily cap would be exceeded (SpendGuard)
  4. LLM call         with the output-token cap
  5. reconcile spend  right after the call returns, even if the reply turns out to be unusable: it cost money
  6. parse            not the required structure: LLMBadOutputError, the raw reply is never shown
  7. validate         keep only citations that name a retrieved section; "not covered" if the model said so
                      or if no valid citation is left (an answer with nothing to check it against is withheld)

LLMUnavailableError, LLMBadOutputError, SpendRefusedError and GateNotConfiguredError are not caught here:
each has a `code` attribute that the API layer turns into an error response.
"""

import logging
import math
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from askact.config import Settings
from askact.generation.llm import LLM
from askact.generation.prompt import SYSTEM_PROMPT, build_user_prompt
from askact.generation.response import parse_reply
from askact.models import RetrievalResult, SectionId

log = logging.getLogger(__name__)

# Shown instead of any model text when the question is not covered, so that a refusal never carries
# something the model made up.
NOT_COVERED_MESSAGE = "The retrieved text of the AI Act does not cover this question."


class SpendRefusedError(Exception):
    """The daily spend cap would be exceeded, or the ledger or price table cannot be used (fail closed)."""

    code = "spend_cap_reached"


class GateNotConfiguredError(Exception):
    """The relevance threshold needed for this retrieval config is not set (or is not a finite number).

    Answering is unusable until the thresholds are tuned (Task 20), and silently passing every question
    to the LLM would defeat the gate and cost money, so this is an error and not a default.
    """

    code = "not_configured"


class SpendReservation(Protocol):
    def reconcile(self, input_tokens: int | None, output_tokens: int | None) -> None:
        """Replace the reserved worst case with the actual cost. `None` means the provider did not report
        usage, and then the reservation stands as the charge (R6.9). It should not raise; if it does, `answer`
        logs that and keeps the reply, because the call has already been paid for."""
        ...


class SpendGuard(Protocol):
    def reserve(self, system: str, user: str, max_output_tokens: int) -> SpendReservation:
        """Set aside the worst-case cost of one call (R6.7). Raises SpendRefusedError if that would exceed
        the daily cap or the ledger cannot be used (R6.8, R6.9)."""
        ...


class Source(BaseModel):
    """A retrieved passage that the answer cites, for the sources panel (R3.11)."""

    model_config = ConfigDict(frozen=True)

    section_id: SectionId
    label: str  # "Article 5"
    title: str | None  # None for recitals
    text: str  # the passage body, without the header
    score: float  # the retrieval config's own ranking score; only comparable within one result


class Usage(BaseModel):
    """What the provider reported for the one call that was made. `None` where it did not report."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int | None
    output_tokens: int | None


class AnswerResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["answered", "not_covered"]
    # The model's answer if answered; otherwise the fixed NOT_COVERED_MESSAGE, never model text.
    answer: str
    citations: list[SectionId]  # all in the retrieved set; empty unless answered
    sources: list[Source]  # the retrieved chunks of the cited sections, best first; empty unless answered
    usage: Usage | None  # None if no LLM call was made (the gate refused first)


def relevance_threshold(settings: Settings, kind: Literal["cosine", "rerank"]) -> float:
    """The relevance threshold for this kind of score, or GateNotConfiguredError. Public because the evaluation
    checks it before its first paid call instead of discovering a missing one half way through."""
    name, value = (
        ("RELEVANCE_THRESHOLD_COSINE", settings.relevance_threshold_cosine)
        if kind == "cosine"
        else ("RELEVANCE_THRESHOLD_RERANK", settings.relevance_threshold_rerank)
    )
    if value is None:
        raise GateNotConfiguredError(f"{name} is not set, so answers cannot be gated: tune it on the dev split first")
    # Settings rejects these when loaded, but a Settings can also be built or copied in code, and
    # `relevance < nan` is always False: a NaN threshold would let every question through.
    if not math.isfinite(value):
        raise GateNotConfiguredError(f"{name} must be a finite number, got {value}")
    return value


def _not_covered(usage: Usage | None) -> AnswerResult:
    return AnswerResult(status="not_covered", answer=NOT_COVERED_MESSAGE, citations=[], sources=[], usage=usage)


def _valid_citations(cited: list[str], retrieved: set[SectionId]) -> list[SectionId]:
    """The model's citations that name a retrieved section, normalised ("Article 5" -> article:5), in the
    order first cited and without repeats. Anything else is dropped (R3.5): text that is not a section id,
    and sections that were not among the retrieved passages."""
    kept: list[SectionId] = []
    for text in cited:
        try:
            section = SectionId.parse(text)
        except ValueError:
            continue
        if section in retrieved and section not in kept:
            kept.append(section)
    return kept


def answer(question: str, result: RetrievalResult, settings: Settings, llm: LLM, spend: SpendGuard) -> AnswerResult:
    threshold = relevance_threshold(settings, result.relevance_kind)
    if result.relevance < threshold:
        return _not_covered(usage=None)

    user_prompt = build_user_prompt(question, result.chunks)
    reservation = spend.reserve(SYSTEM_PROMPT, user_prompt, settings.max_output_tokens)
    # If this raises, nothing is reconciled: the reservation stands as the charge (R6.9).
    reply = llm.complete(SYSTEM_PROMPT, user_prompt, settings.max_output_tokens)
    try:
        reservation.reconcile(reply.input_tokens, reply.output_tokens)
    except Exception as exc:
        # R6.9: when reconciling does not happen the reservation stands as the charge. That is no reason to
        # throw away a reply that has already been paid for. Only the exception type is logged.
        log.warning("spend reconciliation failed (%s); the reservation stands as the charge", type(exc).__name__)
    usage = Usage(input_tokens=reply.input_tokens, output_tokens=reply.output_tokens)

    parsed = parse_reply(reply.text)
    if not parsed.covered:
        return _not_covered(usage)
    citations = _valid_citations(parsed.citations, {scored.chunk.section_id for scored in result.chunks})
    if not citations:
        return _not_covered(usage)

    cited = set(citations)
    sources = [
        Source(
            section_id=scored.chunk.section_id,
            label=scored.chunk.section_id.label,
            title=scored.chunk.title,
            text=scored.chunk.text,
            score=scored.score,
        )
        for scored in result.chunks
        if scored.chunk.section_id in cited
    ]
    return AnswerResult(status="answered", answer=parsed.answer.strip(), citations=citations, sources=sources, usage=usage)
