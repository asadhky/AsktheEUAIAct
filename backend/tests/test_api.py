import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from askact import api as api_module
from askact.api import ERRORS, IndexState, Services, build_services, create_app
from askact.config import ModelPrice, Settings
from askact.generation.answer import GateNotConfiguredError, SpendRefusedError
from askact.generation.llm import FakeLLM, LLMBadOutputError, LLMReply, LLMUnavailableError
from askact.guards.ratelimit import Decision, RateLimiter
from askact.guards.spend import SpendLedger
from askact.index import IndexUnavailableError
from askact.models import Chunk, RetrievalResult, ScoredChunk, SectionId
from askact.request_log import ERROR_OUTCOMES

QUESTION = "QUESTION-MARKER-31c7: what is banned?"
ANSWER = "ANSWER-MARKER-52d8"
PASSAGE = "PASSAGE-MARKER-63e9"
MODEL = "m"
PRICE = ModelPrice(input_per_mtok=1.0, output_per_mtok=4.0)


def chunk(section: str, text: str, title: str | None = "Prohibited AI practices", score: float = 0.9) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(section_id=SectionId.parse(section), ordinal=1, title=title, text=text), score=score)


RETRIEVED = RetrievalResult(chunks=[chunk("article:5", PASSAGE)], relevance=0.9, relevance_kind="cosine")
GOOD = json.dumps({"answer": ANSWER, "citations": ["article:5"], "covered": True})


class Search:
    """A stand-in for Retriever.search that records its calls and returns (or raises) what it is told."""

    def __init__(self, result: RetrievalResult | Exception = RETRIEVED):
        self.result = result
        self.calls: list[tuple[str, str, int]] = []

    def __call__(self, query, config, k):
        self.calls.append((query, config, k))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class Rig:
    def __init__(self, tmp_path, replies=None, search=None, limit=100, index=None, **setting_overrides):
        values = dict(spend_db_path=tmp_path / "spend.sqlite3", daily_spend_cap_usd=1.0, llm_model=MODEL,
                      llm_price_table={MODEL: PRICE}, relevance_threshold_cosine=0.5, max_output_tokens=100,
                      cors_origins=["http://localhost:3000"])
        values.update(setting_overrides)
        self.settings = Settings(_env_file=None, **values)
        self.search = search or Search()
        self.llm = FakeLLM([LLMReply(GOOD, 400, 20)] * 50 if replies is None else replies)
        self.limiter = RateLimiter(limit)
        services = Services(index=index or IndexState(self.search), llm=self.llm,
                            spend=SpendLedger.from_settings(self.settings), limiter=self.limiter)
        self.app = create_app(self.settings, services)
        self.client = TestClient(self.app)

    def ask(self, question=QUESTION, **kwargs):
        return self.client.post("/ask", json={"question": question}, **kwargs)


def error_of(response) -> dict:
    body = response.json()
    assert set(body) == {"error"} and set(body["error"]) == {"code", "message"}
    return body["error"]


@pytest.fixture
def rig(tmp_path) -> Rig:
    return Rig(tmp_path)


@pytest.fixture
def log_lines(caplog):
    """The request-log lines written during a test, parsed."""
    caplog.set_level(logging.DEBUG)

    def lines() -> list[dict]:
        return [json.loads(r.getMessage()) for r in caplog.records if r.name == "askact.request"]

    return lines


# =====================================================================================================
# the happy paths
# =====================================================================================================

def test_a_question_is_answered_with_the_documented_response_shape(rig):
    response = rig.ask()
    assert response.status_code == 200
    assert response.json() == {
        "status": "answered", "answer": ANSWER, "citations": ["article:5"],
        "sources": [{"section_id": "article:5", "label": "Article 5", "title": "Prohibited AI practices", "text": PASSAGE, "score": 0.9}],
    }


def test_the_response_is_json(rig):
    assert rig.ask().headers["content-type"] == "application/json"


def test_a_recital_source_has_a_null_title(tmp_path):
    reply = json.dumps({"answer": ANSWER, "citations": ["recital:12"], "covered": True})
    rig = Rig(tmp_path, replies=[LLMReply(reply, 10, 5)], search=Search(RetrievalResult(
        chunks=[chunk("recital:12", PASSAGE, title=None)], relevance=0.9, relevance_kind="cosine")))
    source = rig.ask().json()["sources"][0]
    assert (source["section_id"], source["label"], source["title"]) == ("recital:12", "Recital 12", None)


def test_a_question_the_model_says_is_not_covered_is_a_200_with_status_not_covered(tmp_path):
    reply = json.dumps({"answer": "", "citations": [], "covered": False})
    body = Rig(tmp_path, replies=[LLMReply(reply, 10, 5)]).ask().json()
    assert body["status"] == "not_covered" and body["citations"] == [] and body["sources"] == []
    assert "does not cover" in body["answer"]


def test_a_question_below_the_relevance_threshold_is_not_covered_and_never_reaches_the_llm(tmp_path):
    rig = Rig(tmp_path, relevance_threshold_cosine=0.99)
    response = rig.ask()
    assert response.status_code == 200 and response.json()["status"] == "not_covered"
    assert rig.llm.call_count == 0 and not (tmp_path / "spend.sqlite3").exists()    # and nothing was reserved


def test_the_search_is_called_with_the_question_the_active_config_and_top_k(tmp_path):
    rig = Rig(tmp_path, top_k=7)
    rig.ask("  spaced question?  ")
    assert rig.search.calls == [("  spaced question?  ", "hybrid", 7)]


@pytest.mark.parametrize("enabled, config", [(False, "hybrid"), (True, "hybrid+rerank")])
def test_the_active_config_follows_the_reranker_flag(tmp_path, enabled, config):
    rig = Rig(tmp_path, reranker_enabled=enabled)
    rig.ask()
    assert rig.search.calls[0][1] == config


def test_the_llm_is_called_once_per_answer_with_the_output_cap(tmp_path):
    rig = Rig(tmp_path, max_output_tokens=77)
    rig.ask()
    assert rig.llm.call_count == 1 and rig.llm.calls[0].max_output_tokens == 77


def test_the_cost_of_an_answer_is_charged_to_the_ledger(tmp_path):
    rig = Rig(tmp_path)
    rig.ask()
    import sqlite3

    [(spent,)] = sqlite3.connect(tmp_path / "spend.sqlite3").execute("SELECT micro_usd FROM spend").fetchall()
    assert spent == 400 * 1 + 20 * 4


# =====================================================================================================
# /health
# =====================================================================================================

def test_health_is_ok_when_the_index_is_ready(rig):
    response = rig.client.get("/health")
    assert response.status_code == 200 and response.json() == {"status": "ok", "index": "ready"}


def test_health_is_503_when_the_index_is_not_available_so_a_container_check_fails(tmp_path):
    rig = Rig(tmp_path, index=IndexState(None, IndexUnavailableError("stale", "details that are not for the public")))
    response = rig.client.get("/health")
    assert response.status_code == 503 and response.json() == {"status": "unavailable", "index": "unavailable"}
    assert "details" not in response.text


def test_health_says_nothing_about_the_deployment(rig):
    text = rig.client.get("/health").text.lower()
    for word in ("model", "key", "path", "version", "cap", "spend", "chunks", "hash"):
        assert word not in text


def test_health_does_not_use_the_rate_limiter_or_write_a_request_log_line(tmp_path, log_lines):
    rig = Rig(tmp_path, limit=1)
    for _ in range(10):
        assert rig.client.get("/health").status_code == 200
    assert log_lines() == []
    assert rig.ask().status_code == 200    # the single allowed /ask request was not used up by the health checks


# =====================================================================================================
# every error code in the design table
# =====================================================================================================

def test_the_error_table_matches_the_design():
    assert {c: s for c, (s, _) in ERRORS.items()} == {
        "request_too_large": 413, "rate_limited": 429, "invalid_request": 422, "question_too_long": 422,
        "index_unavailable": 503, "not_configured": 503, "spend_cap_reached": 503, "llm_unavailable": 503,
        "llm_bad_output": 502, "internal_error": 500,
    }


def test_every_error_the_api_can_send_is_something_the_log_knows_how_to_record():
    assert set(ERRORS) == ERROR_OUTCOMES


def test_error_messages_are_fixed_sentences_with_no_placeholders():
    for code, (_, message) in ERRORS.items():
        assert message and "{" not in message and "%" not in message and message.endswith((".", "!"))


def test_question_too_long_is_422(rig):
    response = rig.ask("x" * 501)
    assert response.status_code == 422 and error_of(response)["code"] == "question_too_long"


def test_a_question_at_exactly_the_limit_is_accepted(rig):
    assert rig.ask("x" * 500).status_code == 200


def test_the_limit_counts_characters_not_bytes(rig):
    assert rig.ask("é" * 500).status_code == 200            # 1000 bytes in UTF-8
    assert rig.ask("日" * 501).status_code == 422
    assert rig.ask("😀" * 500).status_code == 200            # 2000 bytes, and 1000 UTF-16 units


def test_the_limit_comes_from_settings(tmp_path):
    rig = Rig(tmp_path, max_question_chars=10)
    assert rig.ask("x" * 10).status_code == 200 and rig.ask("x" * 11).status_code == 422


def test_an_over_length_question_never_reaches_retrieval_the_llm_or_the_ledger(tmp_path):
    rig = Rig(tmp_path)
    assert rig.ask("x" * 501).status_code == 422
    assert rig.search.calls == [] and rig.llm.call_count == 0 and not (tmp_path / "spend.sqlite3").exists()


def test_rate_limited_is_429_with_retry_after(tmp_path):
    rig = Rig(tmp_path, limit=2)
    assert [rig.ask().status_code for _ in range(2)] == [200, 200]
    response = rig.ask()
    assert response.status_code == 429 and error_of(response)["code"] == "rate_limited"
    assert int(response.headers["retry-after"]) >= 1


def test_a_rate_limited_request_never_reaches_retrieval_the_llm_or_the_ledger(tmp_path):
    rig = Rig(tmp_path, limit=1)
    rig.ask()
    searches, calls = len(rig.search.calls), rig.llm.call_count
    for _ in range(5):
        assert rig.ask().status_code == 429
    assert len(rig.search.calls) == searches and rig.llm.call_count == calls


def test_the_rate_limit_comes_from_the_limiter(tmp_path):
    rig = Rig(tmp_path, limit=3)
    assert [rig.ask().status_code for _ in range(5)] == [200, 200, 200, 429, 429]


def test_spend_cap_reached_is_503(tmp_path):
    rig = Rig(tmp_path, daily_spend_cap_usd=0.0)
    response = rig.ask()
    assert response.status_code == 503 and error_of(response)["code"] == "spend_cap_reached"
    assert rig.llm.call_count == 0


def test_a_model_without_a_price_is_spend_cap_reached(tmp_path):
    rig = Rig(tmp_path, llm_price_table={})
    assert error_of(rig.ask())["code"] == "spend_cap_reached" and rig.llm.call_count == 0


def test_an_unusable_ledger_is_spend_cap_reached(tmp_path):
    (tmp_path / "spend.sqlite3").mkdir()
    rig = Rig(tmp_path)
    response = rig.ask()
    assert response.status_code == 503 and error_of(response)["code"] == "spend_cap_reached" and rig.llm.call_count == 0


def test_llm_unavailable_is_503(tmp_path):
    rig = Rig(tmp_path, replies=[LLMUnavailableError("down")])
    response = rig.ask()
    assert response.status_code == 503 and error_of(response)["code"] == "llm_unavailable"


def test_llm_bad_output_is_502(tmp_path):
    rig = Rig(tmp_path, replies=[LLMReply("this is not json", 100, 10)])
    response = rig.ask()
    assert response.status_code == 502 and error_of(response)["code"] == "llm_bad_output"


def test_a_bad_reply_is_still_charged(tmp_path):
    rig = Rig(tmp_path, replies=[LLMReply("not json", 100, 10)])
    rig.ask()
    import sqlite3

    [(spent,)] = sqlite3.connect(tmp_path / "spend.sqlite3").execute("SELECT micro_usd FROM spend").fetchall()
    assert spent == 100 * 1 + 10 * 4      # the tokens were spent even though the reply was unusable


def test_index_unavailable_is_503_when_the_index_failed_to_load(tmp_path):
    rig = Rig(tmp_path, index=IndexState(None, IndexUnavailableError("stale", "rebuild it")))
    response = rig.ask()
    assert response.status_code == 503 and error_of(response)["code"] == "index_unavailable"
    assert rig.llm.call_count == 0 and "rebuild" not in response.text


def test_index_unavailable_is_503_when_the_index_disappears_after_startup(tmp_path):
    rig = Rig(tmp_path, search=Search(IndexUnavailableError("corrupt", "the file is damaged")))
    response = rig.ask()
    assert response.status_code == 503 and error_of(response)["code"] == "index_unavailable"
    assert "damaged" not in response.text and rig.llm.call_count == 0


def test_an_index_that_never_loaded_and_has_no_recorded_reason_is_still_index_unavailable(tmp_path):
    rig = Rig(tmp_path, index=IndexState(None))
    assert error_of(rig.ask())["code"] == "index_unavailable"


def test_a_not_covered_answer_is_logged_as_not_covered_not_as_answered(tmp_path, log_lines):
    reply = json.dumps({"answer": "", "citations": [], "covered": False})
    Rig(tmp_path, replies=[LLMReply(reply, 30, 5)]).ask()
    [line] = log_lines()
    assert (line["outcome"], line["input_tokens"], line["output_tokens"]) == ("not_covered", 30, 5)


def test_a_gate_refusal_is_logged_as_not_covered_with_null_tokens(tmp_path, log_lines):
    Rig(tmp_path, relevance_threshold_cosine=0.99).ask()
    [line] = log_lines()
    assert (line["outcome"], line["input_tokens"], line["output_tokens"]) == ("not_covered", None, None)


def test_the_index_failure_is_logged_with_its_outcome_and_error_type(tmp_path, log_lines):
    Rig(tmp_path, search=Search(IndexUnavailableError("corrupt", "the file is damaged: /secret/path"))).ask()
    [line] = log_lines()
    assert (line["outcome"], line["error_type"]) == ("index_unavailable", "IndexUnavailableError")


def test_not_configured_is_503_when_the_threshold_is_unset(tmp_path):
    rig = Rig(tmp_path, relevance_threshold_cosine=None)
    response = rig.ask()
    assert response.status_code == 503 and error_of(response)["code"] == "not_configured"
    assert rig.llm.call_count == 0 and not (tmp_path / "spend.sqlite3").exists()


def test_only_the_threshold_for_the_active_config_has_to_be_set(tmp_path):
    assert Rig(tmp_path / "a", relevance_threshold_cosine=0.5, relevance_threshold_rerank=None).ask().status_code == 200
    # With the reranker on, a real retriever returns a rerank-kind score, which is judged by the rerank threshold.
    rerank_result = RetrievalResult(chunks=[chunk("article:5", PASSAGE)], relevance=3.0, relevance_kind="rerank")
    rig = Rig(tmp_path / "b", search=Search(rerank_result), reranker_enabled=True, relevance_threshold_cosine=0.5,
              relevance_threshold_rerank=None)
    assert error_of(rig.ask())["code"] == "not_configured"
    rig = Rig(tmp_path / "c", search=Search(rerank_result), reranker_enabled=True, relevance_threshold_cosine=None,
              relevance_threshold_rerank=0.0)
    assert rig.ask().status_code == 200


def test_an_unexpected_exception_is_a_500_with_a_fixed_message_and_no_details(tmp_path):
    rig = Rig(tmp_path, search=Search(RuntimeError(f"boom {QUESTION} {PASSAGE}")))
    response = rig.ask()
    assert response.status_code == 500 and error_of(response)["code"] == "internal_error"
    assert "boom" not in response.text and "MARKER" not in response.text


class BrokenLimiter:
    """A limiter that fails: that happens outside the endpoint's own error handling, so it reaches the safety net."""

    def check(self, key: str) -> Decision:
        raise RuntimeError(f"SECRET-LIMITER-DETAIL {QUESTION}")


def rig_with_a_failing_limiter(tmp_path) -> Rig:
    rig = Rig(tmp_path)
    services = Services(index=IndexState(rig.search), llm=rig.llm, spend=SpendLedger.from_settings(rig.settings), limiter=BrokenLimiter())
    rig.client = TestClient(create_app(rig.settings, services), raise_server_exceptions=False)
    return rig


def test_a_failure_outside_the_endpoint_is_a_500_with_a_fixed_message_and_no_details(tmp_path):
    response = rig_with_a_failing_limiter(tmp_path).ask()
    assert response.status_code == 500 and error_of(response) == {"code": "internal_error", "message": ERRORS["internal_error"][1]}
    assert "SECRET" not in response.text and "MARKER" not in response.text and "Traceback" not in response.text


def test_a_failure_outside_the_endpoint_is_logged_once_with_its_type_only(tmp_path, log_lines, caplog):
    rig_with_a_failing_limiter(tmp_path).ask()
    [line] = log_lines()
    assert (line["outcome"], line["error_type"]) == ("internal_error", "RuntimeError")
    assert "SECRET" not in caplog.text and "MARKER" not in caplog.text


def test_a_failure_outside_the_endpoint_never_reaches_retrieval_or_the_llm(tmp_path):
    rig = rig_with_a_failing_limiter(tmp_path)
    rig.ask()
    assert rig.search.calls == [] and rig.llm.call_count == 0


def test_an_unexpected_exception_does_not_crash_the_server_for_the_next_request(tmp_path):
    search = Search(RuntimeError("boom"))
    rig = Rig(tmp_path, search=search)
    assert rig.ask().status_code == 500
    search.result = RETRIEVED
    assert rig.ask().status_code == 200


@pytest.mark.parametrize("failure, code, status", [
    (SpendRefusedError("x"), "spend_cap_reached", 503),
    (GateNotConfiguredError("x"), "not_configured", 503),
    (LLMUnavailableError("x"), "llm_unavailable", 503),
    (LLMBadOutputError("x"), "llm_bad_output", 502),
    (IndexUnavailableError("missing", "x"), "index_unavailable", 503),
    (ValueError("x"), "internal_error", 500),
    (KeyError("x"), "internal_error", 500),
])
def test_each_error_class_raised_during_a_request_maps_to_exactly_its_own_code_and_status(tmp_path, log_lines, failure, code, status):
    # Raised from retrieval, the first step that can fail with any of them, so the mapping is the API's own.
    rig = Rig(tmp_path, search=Search(failure))
    response = rig.ask()
    assert (response.status_code, error_of(response)["code"]) == (status, code)
    [line] = log_lines()
    assert line["outcome"] == code and line["error_type"] == type(failure).__name__


