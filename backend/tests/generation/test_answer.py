import json
import logging
import math
from typing import Literal

import pytest

from askact.config import Settings
from askact.generation.answer import (
    NOT_COVERED_MESSAGE,
    AnswerResult,
    GateNotConfiguredError,
    SpendRefusedError,
    Usage,
    answer,
)
from askact.generation.llm import FakeLLM, LLMBadOutputError, LLMReply, LLMUnavailableError
from askact.generation.prompt import SYSTEM_PROMPT, build_user_prompt
from askact.models import Chunk, RetrievalResult, ScoredChunk, SectionId

QUESTION = "What does the Act say about real-time biometric identification?"


# --- doubles and builders -------------------------------------------------------------------

class Reservation:
    def __init__(self, spend: "Spend"):
        self.spend = spend

    def reconcile(self, input_tokens, output_tokens):
        self.spend.events.append("reconcile")
        self.spend.reconciled.append((input_tokens, output_tokens))
        if self.spend.reconcile_error is not None:
            raise self.spend.reconcile_error


class Spend:
    """A SpendGuard that records what it is asked, in order, and can be told to refuse or to fail."""

    def __init__(self, events: list[str], refuse: bool = False, reconcile_error: Exception | None = None):
        self.events = events
        self.refuse = refuse
        self.reconcile_error = reconcile_error
        self.reserved: list[tuple[str, str, int]] = []
        self.reconciled: list[tuple[int | None, int | None]] = []

    def reserve(self, system, user, max_output_tokens):
        self.events.append("reserve")
        self.reserved.append((system, user, max_output_tokens))
        if self.refuse:
            raise SpendRefusedError("daily cap reached")
        return Reservation(self)


class RecordingLLM(FakeLLM):
    """A FakeLLM that also notes in the shared event list when it is called."""

    def __init__(self, events: list[str], replies):
        super().__init__(replies)
        self.events = events

    def complete(self, system, user, max_output_tokens):
        self.events.append("complete")
        return super().complete(system, user, max_output_tokens)


class Rig:
    def __init__(self, replies=(), **spend_options):
        self.events: list[str] = []
        self.llm = RecordingLLM(self.events, replies)
        self.spend = Spend(self.events, **spend_options)


def scored(section: str, text: str = "passage text", title: str | None = None, ordinal: int = 1, score: float = 0.5) -> ScoredChunk:
    chunk = Chunk(section_id=SectionId.parse(section), ordinal=ordinal, title=title, text=text)
    return ScoredChunk(chunk=chunk, score=score)


ART5 = scored("article:5", "1. The following AI practices shall be prohibited ...", "Prohibited AI practices", score=0.9)
REC12 = scored("recital:12", "The notion of an AI system should be clearly defined.", score=0.7)
ANNEX3 = scored("annex:III", "1. Biometrics, in so far as their use is permitted ...", "High-risk AI systems", score=0.6)


def retrieved(chunks=(ART5, REC12, ANNEX3), relevance=0.8, kind: Literal["cosine", "rerank"] = "cosine") -> RetrievalResult:
    return RetrievalResult(chunks=list(chunks), relevance=relevance, relevance_kind=kind)


def settings(cosine: float | None = 0.5, rerank: float | None = 0.0, max_output_tokens: int = 500) -> Settings:
    return Settings(_env_file=None, relevance_threshold_cosine=cosine, relevance_threshold_rerank=rerank,
                    max_output_tokens=max_output_tokens)


def model_reply(answer_text="Real-time identification is prohibited, with narrow exceptions.", citations=("article:5",), covered=True) -> str:
    return json.dumps({"answer": answer_text, "citations": list(citations), "covered": covered})


def run(rig: Rig, result=None, config=None, question=QUESTION) -> AnswerResult:
    return answer(question, result or retrieved(), config or settings(), rig.llm, rig.spend)


# --- the relevance gate (R3.7, R3.8) ----------------------------------------------------------

def assert_refused_without_spending(rig: Rig, outcome: AnswerResult):
    assert outcome.status == "not_covered"
    assert outcome.answer == NOT_COVERED_MESSAGE
    assert (outcome.citations, outcome.sources, outcome.usage) == ([], [], None)
    assert rig.llm.call_count == 0 and rig.spend.reserved == [] and rig.events == []


def test_a_cosine_relevance_below_the_cosine_threshold_is_not_covered_with_no_call_and_no_reservation():
    rig = Rig([model_reply()])
    outcome = run(rig, retrieved(relevance=0.49, kind="cosine"), settings(cosine=0.5))
    assert_refused_without_spending(rig, outcome)


def test_a_rerank_relevance_below_the_rerank_threshold_is_not_covered_with_no_call_and_no_reservation():
    rig = Rig([model_reply()])
    outcome = run(rig, retrieved(relevance=-3.0, kind="rerank"), settings(rerank=-2.5))
    assert_refused_without_spending(rig, outcome)


def test_a_relevance_equal_to_the_threshold_is_let_through():
    cases: list[tuple[Literal["cosine", "rerank"], float, Settings]] = [
        ("cosine", 0.5, settings(cosine=0.5)),
        ("rerank", -2.5, settings(rerank=-2.5)),
    ]
    for kind, relevance, config in cases:
        rig = Rig([model_reply()])
        assert run(rig, retrieved(relevance=relevance, kind=kind), config).status == "answered"
        assert rig.llm.call_count == 1


def test_the_smallest_step_below_the_threshold_is_refused():
    rig = Rig([model_reply()])
    just_below = math.nextafter(0.5, 0.0)
    assert_refused_without_spending(rig, run(rig, retrieved(relevance=just_below), settings(cosine=0.5)))


def test_the_threshold_is_chosen_by_the_kind_of_score_not_by_its_size():
    # 0.4 would fail a cosine threshold of 0.5, but a rerank score of 0.4 is judged against the rerank threshold.
    rig = Rig([model_reply()])
    config = settings(cosine=0.9, rerank=0.0)
    assert run(rig, retrieved(relevance=0.4, kind="rerank"), config).status == "answered"
    # And the other way round: a cosine of 0.4 is judged against the cosine threshold, not the rerank one.
    rig = Rig([model_reply()])
    assert_refused_without_spending(rig, run(rig, retrieved(relevance=0.4, kind="cosine"), settings(cosine=0.5, rerank=-100.0)))


@pytest.mark.parametrize("kind, name, config", [
    ("cosine", "RELEVANCE_THRESHOLD_COSINE", settings(cosine=None)),
    ("rerank", "RELEVANCE_THRESHOLD_RERANK", settings(rerank=None)),
])
def test_an_unset_threshold_is_a_configuration_error_not_a_free_pass(kind, name, config):
    rig = Rig([model_reply()])
    with pytest.raises(GateNotConfiguredError) as info:
        run(rig, retrieved(relevance=1000.0, kind=kind), config)
    assert name in str(info.value) and info.value.code == "not_configured"
    assert rig.llm.call_count == 0 and rig.spend.reserved == []


def test_only_the_threshold_that_applies_has_to_be_set():
    assert run(Rig([model_reply()]), retrieved(kind="cosine"), settings(cosine=0.5, rerank=None)).status == "answered"
    assert run(Rig([model_reply()]), retrieved(relevance=5.0, kind="rerank"), settings(cosine=None, rerank=0.0)).status == "answered"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("kind, field", [("cosine", "relevance_threshold_cosine"), ("rerank", "relevance_threshold_rerank")])
def test_a_threshold_that_is_not_a_finite_number_is_refused_even_if_it_got_past_validation(kind, field, bad):
    # model_copy skips validation. `relevance < nan` and `relevance < -inf` are both False: the gate would be off.
    config = settings().model_copy(update={field: bad})
    rig = Rig([model_reply()])
    with pytest.raises(GateNotConfiguredError, match="finite"):
        run(rig, retrieved(relevance=0.9, kind=kind), config)
    assert rig.llm.call_count == 0 and rig.spend.reserved == []


# --- a normal answer (R3.1, R3.4, R3.11) ------------------------------------------------------

def test_an_answer_carries_the_text_the_citations_the_sources_and_the_usage():
    rig = Rig([LLMReply(model_reply(), input_tokens=900, output_tokens=40)])
    outcome = run(rig)
    assert outcome.status == "answered"
    assert outcome.answer == "Real-time identification is prohibited, with narrow exceptions."
    assert outcome.citations == [SectionId.parse("article:5")]
    [source] = outcome.sources
    assert (str(source.section_id), source.label, source.title, source.score) == ("article:5", "Article 5", "Prohibited AI practices", 0.9)
    assert source.text == "1. The following AI practices shall be prohibited ..."  # the body, without the header
    assert outcome.usage == Usage(input_tokens=900, output_tokens=40)


def test_the_model_sees_the_system_prompt_the_passages_and_the_question_with_the_output_cap():
    rig = Rig([model_reply()])
    run(rig, config=settings(max_output_tokens=123))
    [call] = rig.llm.calls
    assert call.system == SYSTEM_PROMPT
    assert call.user == build_user_prompt(QUESTION, retrieved().chunks)
    assert call.max_output_tokens == 123


def test_the_reservation_is_for_exactly_the_call_that_is_made():
    rig = Rig([model_reply()])
    run(rig, config=settings(max_output_tokens=123))
    [call] = rig.llm.calls
    assert rig.spend.reserved == [(call.system, call.user, 123)]


def test_the_order_is_reserve_then_call_then_reconcile():
    rig = Rig([LLMReply(model_reply(), 10, 5)])
    run(rig)
    assert rig.events == ["reserve", "complete", "reconcile"]


def test_reconcile_gets_the_reported_token_counts():
    rig = Rig([LLMReply(model_reply(), input_tokens=900, output_tokens=40)])
    run(rig)
    assert rig.spend.reconciled == [(900, 40)]


def test_reconcile_is_told_when_usage_was_not_reported():
    rig = Rig([model_reply()])  # a plain string reply carries no token counts
    outcome = run(rig)
    assert rig.spend.reconciled == [(None, None)]
    assert outcome.usage == Usage(input_tokens=None, output_tokens=None)


def test_the_answer_text_is_stripped():
    assert run(Rig([model_reply("  An answer.\n")])).answer == "An answer."


# --- citations are checked against the retrieved set (R3.5) -------------------------------------

def test_a_citation_of_a_section_that_was_not_retrieved_is_dropped():
    outcome = run(Rig([model_reply(citations=["article:99", "article:5"])]))
    assert outcome.citations == [SectionId.parse("article:5")]
    assert [str(s.section_id) for s in outcome.sources] == ["article:5"]


@pytest.mark.parametrize("junk", ["Art. 5(1)", "", "chapter:3", "article:five", "article:0", "see above", "article:5#2"])
def test_a_citation_that_is_not_a_section_id_is_dropped_without_failing_the_answer(junk):
    outcome = run(Rig([model_reply(citations=[junk, "article:5"])]))
    assert outcome.status == "answered" and outcome.citations == [SectionId.parse("article:5")]


def test_citations_are_normalised_deduplicated_and_kept_in_the_order_first_cited():
    outcome = run(Rig([model_reply(citations=["Annex III", "article:5", "ARTICLE 5", "annex:iii"])]))
    assert [str(c) for c in outcome.citations] == ["annex:III", "article:5"]


def test_a_cited_recital_is_a_valid_citation():
    outcome = run(Rig([model_reply(citations=["recital:12"])]))
    assert [str(c) for c in outcome.citations] == ["recital:12"]


def test_sources_are_the_retrieved_chunks_of_the_cited_sections_best_first_not_in_citation_order():
    outcome = run(Rig([model_reply(citations=["article:5", "annex:III"])]), retrieved(chunks=[ANNEX3, ART5, REC12]))
    assert [str(s.section_id) for s in outcome.sources] == ["annex:III", "article:5"]
    assert [str(c) for c in outcome.citations] == ["article:5", "annex:III"]


def test_every_retrieved_chunk_of_a_cited_section_is_a_source():
    first = scored("article:5", "first part", "Prohibited AI practices", ordinal=1, score=0.9)
    other = scored("article:9", "other article", "Risk management", score=0.8)
    second = scored("article:5", "second part", "Prohibited AI practices", ordinal=2, score=0.7)
    outcome = run(Rig([model_reply(citations=["article:5"])]), retrieved(chunks=[first, other, second]))
    assert [(str(s.section_id), s.text, s.score) for s in outcome.sources] == [
        ("article:5", "first part", 0.9), ("article:5", "second part", 0.7)]
    assert outcome.citations == [SectionId.parse("article:5")]  # but cited once


def test_a_recital_source_has_no_title():
    outcome = run(Rig([model_reply(citations=["recital:12"])]))
    [source] = outcome.sources
    assert (source.label, source.title) == ("Recital 12", None)


# --- not covered (R3.6) ---------------------------------------------------------------------------

def test_covered_with_no_valid_citation_is_withheld_as_not_covered():
    rig = Rig([LLMReply(model_reply(citations=["article:99", "nonsense"]), 800, 30)])
    outcome = run(rig)
    assert outcome.status == "not_covered" and outcome.answer == NOT_COVERED_MESSAGE
    assert outcome.citations == [] and outcome.sources == []
    assert outcome.usage == Usage(input_tokens=800, output_tokens=30)  # the call happened, so its cost is reported


def test_covered_with_no_citations_at_all_is_withheld_as_not_covered():
    outcome = run(Rig([model_reply(citations=[])]))
    assert outcome.status == "not_covered" and outcome.answer == NOT_COVERED_MESSAGE


def test_the_model_saying_not_covered_gives_the_fixed_message_never_the_models_own_words():
    rig = Rig([LLMReply(model_reply("I could not find it, but I believe it is Article 9.", ["article:5"], covered=False), 700, 20)])
    outcome = run(rig)
    assert outcome.status == "not_covered"
    assert outcome.answer == NOT_COVERED_MESSAGE and "Article 9" not in outcome.answer
    assert outcome.citations == [] and outcome.sources == []
    assert outcome.usage == Usage(input_tokens=700, output_tokens=20)
    assert rig.spend.reconciled == [(700, 20)]


# --- failures ---------------------------------------------------------------------------------------

def test_a_reply_in_the_wrong_structure_is_llm_bad_output_and_its_cost_is_still_reconciled():
    rig = Rig([LLMReply("Sure! Article 5 bans some things.", input_tokens=900, output_tokens=12)])
    with pytest.raises(LLMBadOutputError):
        run(rig)
    assert rig.events == ["reserve", "complete", "reconcile"]  # tokens were spent even though the reply is unusable
    assert rig.spend.reconciled == [(900, 12)]


def test_a_provider_failure_propagates_and_the_reservation_is_left_to_stand():
    rig = Rig([LLMUnavailableError("provider is down")])
    with pytest.raises(LLMUnavailableError):
        run(rig)
    assert rig.events == ["reserve", "complete"]  # no reconcile: the reservation stands as the charge (R6.9)


def test_a_refused_reservation_means_the_llm_is_never_called():
    rig = Rig([model_reply()], refuse=True)
    with pytest.raises(SpendRefusedError) as info:
        run(rig)
    assert info.value.code == "spend_cap_reached"
    assert rig.events == ["reserve"] and rig.llm.call_count == 0


def test_a_ledger_failure_while_reconciling_does_not_throw_away_a_paid_for_answer(caplog):
    rig = Rig([LLMReply(model_reply(), 900, 40)], reconcile_error=RuntimeError("database is locked: secret detail"))
    with caplog.at_level(logging.WARNING, logger="askact.generation.answer"):
        outcome = run(rig)
    assert outcome.status == "answered"
    [record] = caplog.records
    assert "RuntimeError" in record.getMessage() and "reservation stands" in record.getMessage()
    assert "secret detail" not in record.getMessage()  # only the exception type is logged


def test_the_error_codes_match_the_api_error_table():
    assert SpendRefusedError.code == "spend_cap_reached"
    assert GateNotConfiguredError.code == "not_configured"


def test_a_result_cannot_be_changed_afterwards():
    outcome = run(Rig([model_reply()]))
    with pytest.raises(Exception):
        outcome.status = "not_covered"


def test_each_call_to_answer_is_independent():
    rig = Rig([model_reply(), model_reply(citations=["annex:III"])])
    first, second = run(rig), run(rig)
    assert [str(c) for c in first.citations] == ["article:5"]
    assert [str(c) for c in second.citations] == ["annex:III"]
    assert rig.spend.reserved[0] == rig.spend.reserved[1]  # same question, same passages, same prompt
