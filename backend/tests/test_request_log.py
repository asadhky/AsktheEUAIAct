import json
import logging
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone

import pytest

from askact import request_log
from askact.config import ModelPrice, Settings
from askact.generation.answer import GateNotConfiguredError, SpendRefusedError, answer
from askact.generation.llm import FakeLLM, LLMBadOutputError, LLMReply, LLMUnavailableError
from askact.guards.ratelimit import RateLimiter, client_key
from askact.guards.spend import SpendLedger
from askact.index import IndexUnavailableError
from askact.models import Chunk, RetrievalResult, ScoredChunk, SectionId
from askact.request_log import (
    ERROR_OUTCOMES,
    MAX_LOGGED_QUESTION_CHARS,
    OUTCOMES,
    SUCCESS_OUTCOMES,
    TRUNCATION_MARK,
    describe_error,
    format_request,
    log_request,
)
from askact.retrieval import CONFIGS

NOW = datetime(2026, 10, 12, 12, 0, 0, 123456, tzinfo=UTC)
QUESTION = "QUESTION-MARKER-4f2a: what is banned?"
ADDRESS = "203.0.113.77"
ANSWER = "ANSWER-MARKER-9c1d"
PASSAGE = "PASSAGE-MARKER-7b3e"


def line(
    outcome: str = "answered",
    config: str = "hybrid",
    latency_ms: float = 812.4,
    input_tokens: int | None = 900,
    output_tokens: int | None = 40,
    error_type: str | None = None,
    question: str | None = None,
    now: datetime = NOW,
) -> dict:
    text = format_request(outcome=outcome, config=config, latency_ms=latency_ms, input_tokens=input_tokens,
                          output_tokens=output_tokens, error_type=error_type, question=question, now=now)
    return json.loads(text)


def records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "askact.request"]


# =====================================================================================================
# the line
# =====================================================================================================

def test_the_line_has_exactly_the_documented_fields():
    assert line() == {
        "ts": "2026-10-12T12:00:00.123Z", "event": "ask", "outcome": "answered", "config": "hybrid",
        "latency_ms": 812, "input_tokens": 900, "output_tokens": 40,
    }


def test_there_is_no_field_for_the_question_the_answer_a_passage_or_an_address_by_default():
    assert set(line()) == {"ts", "event", "outcome", "config", "latency_ms", "input_tokens", "output_tokens"}
    assert set(line(outcome="llm_unavailable", error_type="LLMUnavailableError")) == {
        "ts", "event", "outcome", "config", "latency_ms", "input_tokens", "output_tokens", "error_type"}


def test_the_functions_accept_exactly_these_parameters_and_none_that_can_carry_other_request_data():
    import inspect

    assert set(inspect.signature(format_request).parameters) == {
        "outcome", "config", "latency_ms", "input_tokens", "output_tokens", "error_type", "question", "now"}
    assert set(inspect.signature(log_request).parameters) == {
        "outcome", "config", "latency_ms", "input_tokens", "output_tokens", "error_type", "question", "include_question", "now"}


@pytest.mark.parametrize("name", ["answer", "passages", "sources", "client", "address", "ip", "text", "message", "user", "body", "headers"])
def test_there_is_nowhere_to_put_an_answer_a_passage_or_an_address(name):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        format_request(outcome="answered", config="hybrid", latency_ms=1, **{name: "x"})   # type: ignore[call-arg]
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        log_request(outcome="answered", config="hybrid", latency_ms=1, **{name: "x"})   # type: ignore[call-arg]


def test_missing_token_counts_are_json_null_not_zero_or_absent():
    fields = line(input_tokens=None, output_tokens=None)
    assert fields["input_tokens"] is None and fields["output_tokens"] is None
    assert "null" in format_request(outcome="not_covered", config="hybrid", latency_ms=3, now=NOW)


def test_zero_tokens_are_reported_as_zero_not_null():
    fields = line(input_tokens=0, output_tokens=0)
    assert (fields["input_tokens"], fields["output_tokens"]) == (0, 0)


def test_one_count_can_be_null_while_the_other_is_reported():
    fields = line(input_tokens=55, output_tokens=None)
    assert (fields["input_tokens"], fields["output_tokens"]) == (55, None)


@pytest.mark.parametrize("bad", [-1, 1.5, "10", True, [1]])
def test_a_token_count_that_is_not_a_non_negative_integer_is_refused(bad):
    with pytest.raises(ValueError, match="non-negative integer"):
        line(input_tokens=bad)
    with pytest.raises(ValueError, match="non-negative integer"):
        line(output_tokens=bad)


@pytest.mark.parametrize("latency, shown", [(0, 0), (0.4, 0), (0.5, 0), (0.6, 1), (812.4, 812), (812.6, 813), (1234567.0, 1234567)])
def test_latency_is_whole_milliseconds(latency, shown):
    assert line(latency_ms=latency)["latency_ms"] == shown


@pytest.mark.parametrize("bad", [-1, -0.001, float("nan"), float("inf"), float("-inf"), "5", None, True])
def test_a_latency_that_is_not_a_non_negative_finite_number_is_refused(bad):
    with pytest.raises(ValueError, match="latency_ms"):
        line(latency_ms=bad)


def test_the_timestamp_is_utc_with_milliseconds_whatever_timezone_the_time_is_given_in():
    plus_five = timezone(timedelta(hours=5))
    assert line(now=datetime(2026, 10, 12, 17, 0, 0, 123999, tzinfo=plus_five))["ts"] == "2026-10-12T12:00:00.123Z"
    assert line(now=datetime(2026, 10, 12, 23, 59, 59, 999999, tzinfo=UTC))["ts"] == "2026-10-12T23:59:59.999Z"


@pytest.mark.parametrize("microsecond, millis", [(0, "000"), (999, "000"), (1000, "001"), (5000, "005"), (50000, "050"), (99999, "099"), (123456, "123"), (999999, "999")])
def test_milliseconds_are_always_three_digits_and_truncated_not_rounded(microsecond, millis):
    stamp = line(now=datetime(2026, 10, 12, 12, 0, 0, microsecond, tzinfo=UTC))["ts"]
    assert stamp == f"2026-10-12T12:00:00.{millis}Z"


def test_a_naive_time_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="timezone-aware"):
        line(now=datetime(2026, 10, 12, 12, 0, 0))


def test_the_real_clock_is_used_when_no_time_is_given():
    before = datetime.now(UTC)
    fields = json.loads(format_request(outcome="answered", config="hybrid", latency_ms=1))
    after = datetime.now(UTC)
    stamp = datetime.strptime(fields["ts"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    assert before - timedelta(milliseconds=2) <= stamp <= after


def test_the_line_is_compact_json_with_no_spaces_after_separators():
    text = format_request(outcome="answered", config="hybrid", latency_ms=1, now=NOW)
    assert ", " not in text and ": " not in text and text.startswith("{") and text.endswith("}")


# --- outcomes and configs are closed sets --------------------------------------------------------

def test_the_outcomes_are_the_two_successes_and_the_api_error_codes():
    assert SUCCESS_OUTCOMES == {"answered", "not_covered"}
    assert ERROR_OUTCOMES == {"request_too_large", "invalid_request", "question_too_long", "rate_limited", "spend_cap_reached",
                              "llm_unavailable", "llm_bad_output", "index_unavailable", "not_configured", "internal_error"}
    assert OUTCOMES == SUCCESS_OUTCOMES | ERROR_OUTCOMES and not (SUCCESS_OUTCOMES & ERROR_OUTCOMES)


@pytest.mark.parametrize("outcome", sorted(OUTCOMES))
def test_every_known_outcome_is_accepted(outcome):
    assert line(outcome=outcome, error_type="SomeError" if outcome in ERROR_OUTCOMES else None)["outcome"] == outcome


@pytest.mark.parametrize("outcome", [
    "", "ANSWERED", "answered ", "success", "error", f"failed: {QUESTION}", f"{QUESTION}", "SpendRefusedError('x')", "answered\nINFO forged",
])
def test_free_text_cannot_be_passed_as_the_outcome(outcome):
    with pytest.raises(ValueError, match="unknown outcome"):
        line(outcome=outcome)


def test_the_error_message_for_an_unknown_outcome_does_not_need_to_repeat_the_caller_text_to_be_useful():
    with pytest.raises(ValueError) as info:
        line(outcome="bogus")
    assert "answered" in str(info.value)    # it lists what is allowed


@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_every_retrieval_config_is_accepted(config):
    assert line(config=config)["config"] == config


@pytest.mark.parametrize("config", ["", "Hybrid", "hybrid ", f"hybrid {QUESTION}", "semantic", None])
def test_free_text_cannot_be_passed_as_the_config(config):
    with pytest.raises(ValueError, match="unknown retrieval config"):
        line(config=config)


@pytest.mark.parametrize("error_type", ["LLMUnavailableError", "SpendRefusedError", "_Private", "Error2", "A"])
def test_an_exception_type_name_is_accepted(error_type):
    assert line(outcome="llm_unavailable", error_type=error_type)["error_type"] == error_type


@pytest.mark.parametrize("error_type", [
    "", "has space", "Error: the question was x", "a.b", "a-b", "1Starts", "x\ny", "É", "Err()", "A" * 101, f"Error({QUESTION!r})",
])
def test_anything_that_is_not_an_ascii_identifier_cannot_be_passed_as_the_error_type(error_type):
    with pytest.raises(ValueError, match="exception type name"):
        line(outcome="llm_unavailable", error_type=error_type)


@pytest.mark.parametrize("outcome", sorted(SUCCESS_OUTCOMES))
def test_an_error_type_on_a_successful_outcome_is_refused(outcome):
    with pytest.raises(ValueError, match="only for error outcomes"):
        line(outcome=outcome, error_type="SomeError")


# --- the optional question (LOG_QUESTIONS) -----------------------------------------------------------

def test_without_a_question_the_line_has_no_question_field():
    assert "question" not in line()


def test_a_question_is_added_as_its_own_field_and_changes_nothing_else():
    without = line()
    with_question = line(question=QUESTION)
    assert with_question.pop("question") == QUESTION
    assert with_question == without


def test_an_empty_question_is_still_a_question_field():
    assert line(question="")["question"] == ""


def test_a_question_at_the_limit_is_logged_whole_and_one_past_it_is_cut_with_a_mark():
    exact = "q" * MAX_LOGGED_QUESTION_CHARS
    assert line(question=exact)["question"] == exact
    over = line(question=exact + "z")["question"]
    assert over == exact + TRUNCATION_MARK and "z" not in over


def test_a_huge_question_does_not_make_a_huge_line():
    assert len(format_request(outcome="question_too_long", config="hybrid", latency_ms=1, question="x" * 5_000_000, now=NOW)) < 2000 + 400


# --- one physical line, whatever the question contains --------------------------------------------------

HOSTILE = [
    'q1 %s %(x)s %d {0} {x}',
    'line one\nINFO {"event":"ask","outcome":"answered","forged":true}',
    "carriage\rreturn\r\nand crlf",
    "unicode line sep \u2028 and para sep \u2029 and next-line \u0085",
    "vertical tab \x0b form feed \x0c file sep \x1c group sep \x1d record sep \x1e",
    'quotes " and \\ backslashes \\n and a } closing brace',
    "emoji 😀 and CJK 日本語 and accents é",
    "nul \x00 and bell \x07 and escape \x1b[31mred\x1b[0m",
    "\ud800 a lone surrogate",
]


@pytest.mark.parametrize("question", HOSTILE)
def test_a_hostile_question_stays_inside_one_json_string_and_round_trips(question):
    text = format_request(outcome="answered", config="hybrid", latency_ms=1, question=question, now=NOW)
    assert len(text.splitlines()) == 1
    assert text.isascii() and text.isprintable()
    assert json.loads(text)["question"] == question
    assert set(json.loads(text)) == {"ts", "event", "outcome", "config", "latency_ms", "input_tokens", "output_tokens", "question"}


@pytest.mark.parametrize("question", HOSTILE)
def test_a_hostile_question_cannot_forge_a_second_log_line_through_the_logger(question, caplog):
    with caplog.at_level(logging.INFO, logger="askact.request"):
        log_request(outcome="answered", config="hybrid", latency_ms=1, question=question, include_question=True, now=NOW)
    [record] = records(caplog)
    rendered = record.getMessage()
    assert len(rendered.splitlines()) == 1 and json.loads(rendered)["question"] == question


def test_percent_signs_in_a_question_are_not_read_as_a_format_directive(caplog):
    with caplog.at_level(logging.INFO, logger="askact.request"):
        log_request(outcome="answered", config="hybrid", latency_ms=1, question="100% %s %(name)s %d", include_question=True, now=NOW)
    [record] = records(caplog)
    assert json.loads(record.getMessage())["question"] == "100% %s %(name)s %d"


# =====================================================================================================
# log_request
# =====================================================================================================

def test_exactly_one_record_is_written_per_request_at_info_level(caplog):
    with caplog.at_level(logging.DEBUG):
        log_request(outcome="answered", config="hybrid", latency_ms=5, now=NOW)
    [record] = records(caplog)
    assert record.levelno == logging.INFO and record.name == "askact.request"
    assert len(caplog.records) == 1    # nothing else was logged


def test_the_record_is_the_line_and_nothing_more(caplog):
    with caplog.at_level(logging.INFO, logger="askact.request"):
        log_request(outcome="answered", config="hybrid", latency_ms=812.4, input_tokens=900, output_tokens=40, now=NOW)
    [record] = records(caplog)
    expected = format_request(outcome="answered", config="hybrid", latency_ms=812.4, input_tokens=900, output_tokens=40, now=NOW)
    assert record.getMessage() == expected
    assert record.msg == "%s" and record.args == (expected,)   # the line is an argument, never the template


def test_the_question_is_left_out_unless_logging_it_is_switched_on(caplog):
    with caplog.at_level(logging.DEBUG):
        log_request(outcome="answered", config="hybrid", latency_ms=1, question=QUESTION, include_question=False, now=NOW)
        log_request(outcome="answered", config="hybrid", latency_ms=1, question=QUESTION, now=NOW)
    assert QUESTION not in caplog.text and "QUESTION-MARKER" not in caplog.text


def test_switching_it_on_adds_only_the_question(caplog):
    with caplog.at_level(logging.INFO, logger="askact.request"):
        log_request(outcome="answered", config="hybrid", latency_ms=1, input_tokens=3, output_tokens=4, now=NOW)
        log_request(outcome="answered", config="hybrid", latency_ms=1, input_tokens=3, output_tokens=4, now=NOW,
                    question=QUESTION, include_question=True)
    plain, with_question = (json.loads(r.getMessage()) for r in records(caplog))
    assert with_question.pop("question") == QUESTION
    assert with_question == plain


def test_switching_it_on_without_a_question_adds_nothing(caplog):
    with caplog.at_level(logging.INFO, logger="askact.request"):
        log_request(outcome="answered", config="hybrid", latency_ms=1, include_question=True, now=NOW)
    assert "question" not in json.loads(records(caplog)[0].getMessage())


def test_an_invalid_request_is_never_half_logged(caplog):
    with caplog.at_level(logging.DEBUG), pytest.raises(ValueError):
        log_request(outcome="not-an-outcome", config="hybrid", latency_ms=1, question=QUESTION, include_question=True)
    assert records(caplog) == [] and QUESTION not in caplog.text


def test_the_default_for_logging_the_question_is_off():
    import inspect

    assert inspect.signature(log_request).parameters["include_question"].default is False
    assert Settings(_env_file=None).log_questions is False


def test_the_setting_that_switches_it_on_is_log_questions(monkeypatch):
    monkeypatch.setenv("LOG_QUESTIONS", "true")
    assert Settings(_env_file=None).log_questions is True


# =====================================================================================================
# describe_error: the code and the type, never the message
# =====================================================================================================

@pytest.mark.parametrize("exc, outcome, type_name", [
    (SpendRefusedError("m"), "spend_cap_reached", "SpendRefusedError"),
    (GateNotConfiguredError("m"), "not_configured", "GateNotConfiguredError"),
    (LLMUnavailableError("m"), "llm_unavailable", "LLMUnavailableError"),
    (LLMBadOutputError("m"), "llm_bad_output", "LLMBadOutputError"),
])
def test_our_own_errors_give_their_api_code_and_type_name(exc, outcome, type_name):
    assert describe_error(exc) == (outcome, type_name)


def test_every_code_our_errors_declare_is_a_known_outcome():
    for error in (SpendRefusedError, GateNotConfiguredError, LLMUnavailableError, LLMBadOutputError):
        assert error.code in ERROR_OUTCOMES


@pytest.mark.parametrize("exc", [ValueError("x"), KeyError("k"), RuntimeError(), OSError("disk"), ZeroDivisionError()])
def test_an_error_without_a_code_is_an_internal_error_with_its_type(exc):
    assert describe_error(exc) == ("internal_error", type(exc).__name__)


def test_an_exception_whose_code_is_not_a_known_outcome_is_not_trusted():
    class Odd(Exception):
        code = f"free text: {QUESTION}"

    class Numeric(Exception):
        code = 500

    class FromLibrary(Exception):
        code = "answered"    # a known outcome, but a success: not an error code

    assert describe_error(Odd()) == ("internal_error", "Odd")
    assert describe_error(Numeric()) == ("internal_error", "Numeric")
    assert describe_error(FromLibrary()) == ("internal_error", "FromLibrary")


@pytest.mark.parametrize("code", [["spend_cap_reached"], {"a": 1}, {"spend_cap_reached"}, None, 3.5, b"spend_cap_reached"])
def test_a_code_that_is_not_a_string_cannot_make_describe_error_itself_fail(code):
    # describe_error runs inside error handlers: it must never raise. An unhashable code would break a set lookup.
    class Odd(Exception):
        pass

    Odd.code = code  # type: ignore[attr-defined]
    assert describe_error(Odd("x")) == ("internal_error", "Odd")


def test_a_code_set_on_the_instance_instead_of_the_class_is_ignored():
    exc = ValueError("x")
    exc.code = "spend_cap_reached"  # type: ignore[attr-defined]
    assert describe_error(exc) == ("internal_error", "ValueError")


def test_the_index_error_has_no_code_yet_so_it_is_an_internal_error_until_the_api_maps_it():
    # Task 18 maps IndexUnavailableError to `index_unavailable` itself; describe_error must not guess from `kind`.
    assert describe_error(IndexUnavailableError("missing", "msg")) == ("internal_error", "IndexUnavailableError")


@pytest.mark.parametrize("exc", [
    SpendRefusedError(f"{QUESTION}"), LLMUnavailableError(f"{QUESTION} {ADDRESS}"), ValueError(QUESTION, ADDRESS),
    KeyError(QUESTION), OSError(2, QUESTION, ADDRESS), RuntimeError(QUESTION),
])
def test_describe_error_never_returns_any_of_the_exceptions_text(exc):
    outcome, type_name = describe_error(exc)
    for part in (outcome, type_name):
        assert "MARKER" not in part and ADDRESS not in part


def test_the_exception_cause_and_context_are_not_followed():
    try:
        try:
            raise ValueError(f"inner {QUESTION}")
        except ValueError as inner:
            raise LLMUnavailableError("outer") from inner
    except LLMUnavailableError as outer:
        text = " ".join(describe_error(outer))
    assert "MARKER" not in text and "ValueError" not in text


def test_the_outcome_from_describe_error_is_always_accepted_by_the_line():
    for exc in (SpendRefusedError("x"), ValueError("x"), LLMBadOutputError("x"), type("Weird", (Exception,), {"code": "nope"})("x")):
        outcome, type_name = describe_error(exc)
        assert line(outcome=outcome, error_type=type_name)["outcome"] == outcome


def test_a_dynamically_named_exception_class_cannot_inject_text_through_its_name():
    Evil = type("Evil Name\nINFO forged", (Exception,), {})     # type() accepts any string as a class name
    outcome, type_name = describe_error(Evil("x"))
    with pytest.raises(ValueError, match="exception type name"):
        line(outcome=outcome, error_type=type_name)


# =====================================================================================================
# the privacy guarantee over the real request path
# =====================================================================================================
# The tests above show that this module cannot write a question. These show that nothing else on the request
# path does either: they run the real rate limiter, relevance gate, spend ledger and answer pipeline with a
# distinctive question, address and answer, capture EVERY log record, and look for them.

def scored(section: str, text: str) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(section_id=SectionId.parse(section), ordinal=1, title="T", text=text), score=0.9)


RETRIEVED = RetrievalResult(chunks=[scored("article:5", PASSAGE + " " * 3 + "more text " * 10)], relevance=0.9, relevance_kind="cosine")
MODEL = "m"


def settings(tmp_path, **overrides) -> Settings:
    values = dict(spend_db_path=tmp_path / "s.sqlite3", daily_spend_cap_usd=1.0, llm_model=MODEL,
                  llm_price_table={MODEL: ModelPrice(input_per_mtok=1.0, output_per_mtok=4.0)},
                  relevance_threshold_cosine=0.5, max_output_tokens=100)
    values.update(overrides)
    return Settings(_env_file=None, **values)


def run(tmp_path, llm, **overrides):
    config = settings(tmp_path, **overrides)
    return answer(QUESTION, RETRIEVED, config, llm, SpendLedger.from_settings(config))


def assert_nothing_private_was_logged(caplog):
    everything = "\n".join(f"{r.name}|{r.levelname}|{r.getMessage()}|{r.exc_text}|{r.args}|{r.msg}" for r in caplog.records)
    for secret in (QUESTION, "QUESTION-MARKER", ADDRESS, ANSWER, PASSAGE):
        assert secret not in everything, f"{secret!r} appeared in the logs"
    assert all(r.exc_info is None for r in caplog.records), "a record carries a traceback"


GOOD = json.dumps({"answer": ANSWER, "citations": ["article:5"], "covered": True})


def test_a_successful_answer_logs_nothing_private_anywhere_in_the_stack(tmp_path, caplog):
    with caplog.at_level(logging.DEBUG):
        run(tmp_path, FakeLLM([LLMReply(GOOD, 400, 30)]))
    assert_nothing_private_was_logged(caplog)


def test_a_reconciliation_failure_logs_the_type_only(tmp_path, caplog):
    config = settings(tmp_path)
    ledger = SpendLedger.from_settings(config)
    import sqlite3

    class Broken(SpendLedger):
        def _reconcile(self, *a, **k):
            raise sqlite3.OperationalError(f"disk I/O error near {QUESTION} from {ADDRESS}")

    guard = Broken.from_settings(config)
    with caplog.at_level(logging.DEBUG):
        # the reservation is made normally; reconcile (which answer() guards) fails with a distinctive message
        answer(QUESTION, RETRIEVED, config, FakeLLM([LLMReply(GOOD, 400, 30)]), guard)
    assert_nothing_private_was_logged(caplog)
    assert ledger is not None


def test_a_ledger_that_cannot_be_used_logs_the_type_only(tmp_path, caplog):
    (tmp_path / "s.sqlite3").mkdir()       # the database path is a directory
    with caplog.at_level(logging.DEBUG), pytest.raises(SpendRefusedError):
        run(tmp_path, FakeLLM([LLMReply(GOOD)]))
    assert_nothing_private_was_logged(caplog)
    assert str(tmp_path) not in caplog.text


@pytest.mark.parametrize("failure", [
    LLMUnavailableError(f"{QUESTION} from {ADDRESS}"),
    LLMBadOutputError(f"{QUESTION} {ANSWER}"),
    ValueError(f"{QUESTION}"),
    RuntimeError(f"{ANSWER} {PASSAGE}"),
])
def test_a_provider_failure_propagates_without_the_pipeline_logging_its_text(tmp_path, caplog, failure):
    with caplog.at_level(logging.DEBUG), pytest.raises(type(failure)):
        run(tmp_path, FakeLLM([failure]))
    assert_nothing_private_was_logged(caplog)


def test_a_reply_that_is_not_the_required_structure_does_not_log_the_reply(tmp_path, caplog):
    with caplog.at_level(logging.DEBUG), pytest.raises(LLMBadOutputError):
        run(tmp_path, FakeLLM([LLMReply(f"I will not follow the format. {ANSWER} {QUESTION}", 100, 20)]))
    assert_nothing_private_was_logged(caplog)


def test_a_gate_refusal_logs_nothing_private(tmp_path, caplog):
    with caplog.at_level(logging.DEBUG):
        outcome = run(tmp_path, FakeLLM([LLMReply(GOOD)]), relevance_threshold_cosine=0.99)
    assert outcome.status == "not_covered"
    assert_nothing_private_was_logged(caplog)


def test_an_unset_threshold_logs_nothing_private(tmp_path, caplog):
    with caplog.at_level(logging.DEBUG), pytest.raises(GateNotConfiguredError):
        run(tmp_path, FakeLLM([LLMReply(GOOD)]), relevance_threshold_cosine=None)
    assert_nothing_private_was_logged(caplog)


def test_a_missing_price_logs_nothing_private(tmp_path, caplog):
    with caplog.at_level(logging.DEBUG), pytest.raises(SpendRefusedError):
        run(tmp_path, FakeLLM([LLMReply(GOOD)]), llm_model="unpriced-model")
    assert_nothing_private_was_logged(caplog)


def test_the_rate_limiter_never_logs_a_client_address(caplog):
    limiter = RateLimiter(2)
    with caplog.at_level(logging.DEBUG):
        for _ in range(5):
            limiter.check(client_key(ADDRESS))
            limiter.check(client_key("10.0.0.1", [f"6.6.6.6, {ADDRESS}"]))
    assert ADDRESS not in caplog.text and caplog.records == []


def request_path_modules():
    """Every module of the backend except ingestion (which never sees a user question) and evaluation."""
    import pathlib

    root = pathlib.Path(request_log.__file__).parent
    return root, [p for p in sorted(root.rglob("*.py")) if not {"ingest", "evaluation"} & set(p.relative_to(root).parts)]


def calls_in(path):
    """(line, dotted name of the callee, keyword argument names) for every call in a module, found with `ast`, so
    that prose in a docstring or comment can neither trigger nor hide a match."""
    import ast

    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call):
            parts, target = [], node.func
            while isinstance(target, ast.Attribute):
                parts.append(target.attr)
                target = target.value
            if isinstance(target, ast.Name):
                parts.append(target.id)
            yield node.lineno, ".".join(reversed(parts)), [k.arg for k in node.keywords]


def test_no_module_on_the_request_path_logs_a_traceback():
    # A traceback prints the text of the exception, which can contain request data.
    root, modules = request_path_modules()
    offenders = []
    for path in modules:
        for line_number, callee, keywords in calls_in(path):
            if callee.endswith(".exception") or callee.startswith("traceback.") or "exc_info" in keywords or "stack_info" in keywords:
                offenders.append(f"{path.relative_to(root)}:{line_number}: {callee}({', '.join(k or '**' for k in keywords)})")
    assert offenders == [], offenders


def test_the_checker_itself_would_catch_a_traceback_call_and_ignore_prose(tmp_path):
    source = """\
\"\"\"Mentions logger.exception() and exc_info in prose.\"\"\"
import logging

log = logging.getLogger("x")
# log.exception("in a comment")
try:
    pass
except Exception:
    log.exception("boom")
    log.warning("x", exc_info=True)
"""
    bad = tmp_path / "bad.py"
    bad.write_text(source)
    found = [(callee, kw) for _, callee, kw in calls_in(bad)]
    assert ("log.exception", []) in found and ("log.warning", ["exc_info"]) in found
    flagged = [f for f in found if f[0].endswith(".exception") or "exc_info" in f[1]]
    assert len(flagged) == 2   # the docstring and the comment, which mention the same words, are not counted


def test_the_only_loggers_on_the_request_path_are_the_ones_we_know_about():
    # If someone adds a log call to the request path, this fails and forces a look at what it logs.
    root, modules = request_path_modules()
    found = set()
    for path in modules:
        for _, callee, _ in calls_in(path):
            if callee.split(".")[0] in {"log", "logger"} and callee.split(".")[-1] in {"debug", "info", "warning", "error", "critical"}:
                found.add(path.relative_to(root).as_posix())
    # api.py logs once at startup that the index is unusable: the exception's kind and a fixed sentence, no path.
    assert found == {"api.py", "embedders.py", "generation/answer.py", "guards/spend.py", "request_log.py"}


def test_every_log_call_on_the_request_path_passes_its_values_as_arguments_never_inside_the_template():
    # An f-string or `+` in the first argument builds the message from data, which is how a question reaches a log.
    import ast

    root, modules = request_path_modules()
    offenders = []
    for path in modules:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"debug", "info", "warning", "error", "critical"}:
                owner = node.func.value
                if isinstance(owner, ast.Name) and owner.id in {"log", "logger"} and node.args:
                    template = node.args[0]
                    if not (isinstance(template, ast.Constant) and isinstance(template.value, str)):
                        offenders.append(f"{path.relative_to(root)}:{node.lineno}: the message is not a string literal")
    assert offenders == [], offenders


# =====================================================================================================
# importing this module must stay cheap
# =====================================================================================================

def test_importing_the_request_log_does_not_load_a_machine_learning_library():
    code = "import sys; import askact.request_log; bad = [m for m in ('torch', 'sentence_transformers', 'transformers') if m in sys.modules]; print(bad)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
