import json
import logging
import sqlite3

import pytest
from fastapi.testclient import TestClient

from askact.api import BODY_MARGIN_BYTES, BYTES_PER_QUESTION_CHAR, ERRORS, IndexState, Services, create_app
from askact.config import ModelPrice, Settings
from askact.generation.llm import FakeLLM, LLMReply
from askact.guards.ratelimit import RateLimiter
from askact.guards.spend import SpendLedger
from askact.models import Chunk, RetrievalResult, ScoredChunk, SectionId

QUESTION = "QUESTION-MARKER-77a1: what is banned?"
ANSWER = "ANSWER-MARKER-88b2"
PASSAGE = "PASSAGE-MARKER-99c3"
ADDRESS = "203.0.113.45"
GOOD = json.dumps({"answer": ANSWER, "citations": ["article:5"], "covered": True})
RETRIEVED = RetrievalResult(
    chunks=[ScoredChunk(chunk=Chunk(section_id=SectionId.parse("article:5"), ordinal=1, title="T", text=PASSAGE), score=0.9)],
    relevance=0.9, relevance_kind="cosine")


class Search:
    def __init__(self):
        self.calls = 0

    def __call__(self, query, config, k):
        self.calls += 1
        return RETRIEVED


class Rig:
    def __init__(self, tmp_path, limit=1000, **overrides):
        values = dict(spend_db_path=tmp_path / "s.sqlite3", daily_spend_cap_usd=1.0, llm_model="m",
                      llm_price_table={"m": ModelPrice(input_per_mtok=1.0, output_per_mtok=4.0)},
                      relevance_threshold_cosine=0.5, max_output_tokens=100, cors_origins=["http://localhost:3000"])
        values.update(overrides)
        self.settings = Settings(_env_file=None, **values)
        self.search = Search()
        self.llm = FakeLLM([LLMReply(GOOD, 400, 20)] * 200)
        self.limiter = RateLimiter(limit)
        services = Services(index=IndexState(self.search), llm=self.llm, spend=SpendLedger.from_settings(self.settings), limiter=self.limiter)
        self.client = TestClient(create_app(self.settings, services))

    def untouched(self) -> bool:
        return self.search.calls == 0 and self.llm.call_count == 0


def error_code(response) -> str:
    body = response.json()
    assert set(body) == {"error"} and set(body["error"]) == {"code", "message"}
    return body["error"]["code"]


@pytest.fixture
def log_lines(caplog):
    caplog.set_level(logging.DEBUG)
    return lambda: [json.loads(r.getMessage()) for r in caplog.records if r.name == "askact.request"]


JSON = {"content-type": "application/json"}


# =====================================================================================================
# bodies that are not a valid request: FastAPI's own 422 is replaced, and nothing is echoed back
# =====================================================================================================

MALFORMED = [
    ("not json", dict(content=b"this is not json", headers=JSON)),
    ("empty body", dict(content=b"", headers=JSON)),
    ("missing field", dict(json={})),
    ("wrong field name", dict(json={"q": "x"})),
    ("question is a number", dict(json={"question": 5})),
    ("question is null", dict(json={"question": None})),
    ("question is a list", dict(json={"question": ["a"]})),
    ("question is an object", dict(json={"question": {"a": 1}})),
    ("body is a list", dict(json=["question"])),
    ("body is a string", dict(json="question")),
    ("body is null", dict(content=b"null", headers=JSON)),
    ("an extra field", dict(json={"question": "hi", "extra": 1})),
    ("form encoded", dict(data={"question": "hi"})),
    ("truncated json", dict(content=b'{"question": "hi', headers=JSON)),
    ("invalid utf-8", dict(content=b'{"question": "\xff\xfe"}', headers=JSON)),
    ("no content type", dict(content=b'{"question": "hi"}')),
]


@pytest.mark.parametrize("label, kwargs", MALFORMED, ids=[m[0] for m in MALFORMED])
def test_a_malformed_request_is_a_422_invalid_request_in_the_documented_shape(tmp_path, label, kwargs):
    rig = Rig(tmp_path)
    response = rig.client.post("/ask", **kwargs)
    assert (response.status_code, error_code(response)) == (422, "invalid_request"), label
    assert rig.untouched()


# Each is invalid in a different way and carries the marker somewhere FastAPI's default 422 would echo it back.
ECHO_BAIT = [
    ("number question, marker in a key", dict(json={"question": 5, QUESTION: 1})),
    ("object question", dict(json={"question": {QUESTION: 1}})),
    ("list question", dict(json={"question": [QUESTION]})),
    ("null question, extra marker field", dict(json={"question": None, "note": QUESTION})),
    ("missing question, marker field", dict(json={QUESTION: "x"})),
    ("body is a list of markers", dict(json=[QUESTION])),
    ("body is a marker string", dict(json=QUESTION)),
    ("not json, marker text", dict(content=QUESTION.encode(), headers=JSON)),
    ("truncated json with marker", dict(content=('{"question": "' + QUESTION).encode(), headers=JSON)),
    ("extra field with marker", dict(json={"question": "hi", "extra": QUESTION})),
]


@pytest.mark.parametrize("label, kwargs", ECHO_BAIT, ids=[m[0] for m in ECHO_BAIT])
def test_the_error_for_a_malformed_request_never_echoes_what_was_sent(tmp_path, label, kwargs):
    rig = Rig(tmp_path)
    response = rig.client.post("/ask", **kwargs)
    assert response.status_code == 422, label                       # it really was rejected (a 200 would prove nothing)
    assert "MARKER" not in response.text and "input" not in response.text and "loc" not in response.text
    assert "detail" not in response.text


def test_a_valid_request_sent_with_extra_whitespace_and_a_charset_is_accepted(tmp_path):
    rig = Rig(tmp_path)
    response = rig.client.post("/ask", content=b'  {"question": "hi"}  ', headers={"content-type": "application/json; charset=utf-8"})
    assert response.status_code == 200


@pytest.mark.parametrize("question", ["", " ", "   ", "\n", "\t", " \n\t ", "\u00a0", "\u2003\u2003", "\r\n"])
def test_a_blank_question_is_an_invalid_request_not_a_server_error_and_does_no_work(tmp_path, log_lines, question):
    # The real retriever raises on an empty query; that would surface as a 500 for what is the client's mistake.
    rig = Rig(tmp_path)
    response = rig.client.post("/ask", json={"question": question})
    assert (response.status_code, error_code(response)) == (422, "invalid_request")
    assert rig.untouched() and not (tmp_path / "s.sqlite3").exists()
    [line] = log_lines()
    assert line["outcome"] == "invalid_request" and "error_type" not in line


@pytest.mark.parametrize("question", ["?", "!!!", "a", "the", "日本語", "  padded  ", "0"])
def test_a_short_or_odd_but_not_blank_question_is_passed_on(tmp_path, question):
    rig = Rig(tmp_path)
    assert rig.client.post("/ask", json={"question": question}).status_code == 200 and rig.search.calls == 1


def test_unicode_in_the_question_survives(tmp_path):
    rig = Rig(tmp_path)
    assert rig.client.post("/ask", json={"question": "Hochrisiko-KI — „Artikel 6“ 日本語 😀"}).status_code == 200


# =====================================================================================================
# the body-size cap
# =====================================================================================================

def limit_for(max_chars: int) -> int:
    return max_chars * BYTES_PER_QUESTION_CHAR + BODY_MARGIN_BYTES


def test_the_limit_is_generous_enough_for_the_worst_case_question(tmp_path):
    # A maximum-length question made entirely of characters that JSON escapes to 12 bytes each (an emoji as a
    # surrogate pair, \\ud83d\\ude00) must still fit, or a legitimate user would be refused.
    rig = Rig(tmp_path)
    body = json.dumps({"question": "😀" * 500}, ensure_ascii=True)
    assert len(body) <= limit_for(500)
    assert rig.client.post("/ask", content=body, headers=JSON).status_code == 200


def test_a_body_far_over_the_limit_is_a_413_and_is_never_parsed(tmp_path):
    rig = Rig(tmp_path)
    huge = json.dumps({"question": "x" * 5_000_000})
    response = rig.client.post("/ask", content=huge, headers=JSON)
    assert (response.status_code, error_code(response)) == (413, "request_too_large")
    assert rig.untouched()


def test_a_body_one_byte_over_the_limit_is_refused_and_one_at_the_limit_is_not(tmp_path):
    rig = Rig(tmp_path)
    limit = limit_for(500)
    at_limit = b" " * (limit - len(b'{"question": "hi"}')) + b'{"question": "hi"}'
    assert len(at_limit) == limit
    assert rig.client.post("/ask", content=at_limit, headers=JSON).status_code == 200
    assert rig.client.post("/ask", content=b" " + at_limit, headers=JSON).status_code == 413


def test_a_chunked_body_with_no_content_length_is_capped_too(tmp_path):
    rig = Rig(tmp_path)

    def chunks():
        for _ in range(300):
            yield b"x" * 1000

    response = rig.client.post("/ask", content=chunks(), headers=JSON)
    assert (response.status_code, error_code(response)) == (413, "request_too_large")
    assert rig.untouched()


def test_a_small_chunked_body_is_still_served(tmp_path):
    rig = Rig(tmp_path)

    def chunks():
        yield b'{"question": "'
        yield b"y" * 50
        yield b'"}'

    assert rig.client.post("/ask", content=chunks(), headers=JSON).status_code == 200


def test_a_lying_content_length_cannot_get_a_big_body_past_the_cap(tmp_path):
    rig = Rig(tmp_path)

    def chunks():
        yield b"x" * 200_000

    response = rig.client.post("/ask", content=chunks(), headers={**JSON, "content-length": "10"})
    assert response.status_code in {413, 400, 422}
    assert rig.untouched()


def test_the_cap_follows_max_question_chars(tmp_path):
    rig = Rig(tmp_path, max_question_chars=10)
    assert rig.client.post("/ask", content=b" " * (limit_for(10) + 1) + b'{"question": "hi"}', headers=JSON).status_code == 413


def test_the_cap_does_not_apply_to_get_requests(tmp_path):
    rig = Rig(tmp_path)
    assert rig.client.get("/health", headers={"content-length": "0"}).status_code == 200


def test_a_refused_body_is_logged_once_and_never_logs_the_body(tmp_path, log_lines):
    rig = Rig(tmp_path)
    rig.client.post("/ask", content=json.dumps({"question": QUESTION * 100000}), headers=JSON)
    [line] = log_lines()
    assert line["outcome"] == "request_too_large" and "question" not in line


# =====================================================================================================
# exactly one log line per /ask, on every path
# =====================================================================================================

def test_every_outcome_writes_exactly_one_log_line(tmp_path, log_lines):
    rig = Rig(tmp_path, limit=1000)
    cases = [
        (dict(json={"question": "hi"}), "answered"),
        (dict(json={"question": "x" * 501}), "question_too_long"),
        (dict(json={}), "invalid_request"),
        (dict(content=b"nope", headers=JSON), "invalid_request"),
        (dict(content=b"x" * 100_000, headers=JSON), "request_too_large"),
    ]
    for kwargs, outcome in cases:
        before = len(log_lines())
        rig.client.post("/ask", **kwargs)
        lines = log_lines()
        assert len(lines) == before + 1 and lines[-1]["outcome"] == outcome, outcome


def test_a_rate_limited_request_writes_exactly_one_line(tmp_path, log_lines):
    rig = Rig(tmp_path, limit=1)
    rig.client.post("/ask", json={"question": "hi"})
    rig.client.post("/ask", json={"question": "hi"})
    assert [l["outcome"] for l in log_lines()] == ["answered", "rate_limited"]


def test_a_rate_limited_malformed_request_writes_one_line_and_is_rate_limited_not_invalid(tmp_path, log_lines):
    rig = Rig(tmp_path, limit=1)
    rig.client.post("/ask", json={"question": "hi"})
    response = rig.client.post("/ask", content=b"garbage", headers=JSON)
    assert (response.status_code, error_code(response)) == (429, "rate_limited")
    assert [l["outcome"] for l in log_lines()] == ["answered", "rate_limited"]


def test_a_malformed_request_counts_against_the_rate_limit(tmp_path):
    rig = Rig(tmp_path, limit=3)
    assert [rig.client.post("/ask", content=b"garbage", headers=JSON).status_code for _ in range(5)] == [422, 422, 422, 429, 429]


def test_the_oversize_body_check_runs_before_the_rate_limiter_so_it_does_not_use_up_the_limit(tmp_path):
    # The cap is outermost and costs nothing; a refused giant body is not a request that "used" the client's quota.
    rig = Rig(tmp_path, limit=1)
    assert rig.client.post("/ask", content=b"x" * 100_000, headers=JSON).status_code == 413
    assert rig.client.post("/ask", json={"question": "hi"}).status_code == 200


def test_the_log_line_for_an_answer_carries_the_token_counts(tmp_path, log_lines):
    Rig(tmp_path).client.post("/ask", json={"question": "hi"})
    [line] = log_lines()
    assert (line["config"], line["input_tokens"], line["output_tokens"]) == ("hybrid", 400, 20)


def test_the_log_line_for_an_error_has_null_tokens_and_the_error_type(tmp_path, log_lines):
    Rig(tmp_path, daily_spend_cap_usd=0.0).client.post("/ask", json={"question": "hi"})
    [line] = log_lines()
    assert (line["outcome"], line["error_type"], line["input_tokens"], line["output_tokens"]) == ("spend_cap_reached", "SpendRefusedError", None, None)


def test_the_logged_latency_is_measured_by_the_clock_the_app_was_given(tmp_path, log_lines):
    from askact.api import IndexState, Services

    ticks = iter([100.0, 100.25, 100.25, 100.25])
    rig = Rig(tmp_path)
    services = Services(index=IndexState(rig.search), llm=rig.llm, spend=SpendLedger.from_settings(rig.settings),
                        limiter=RateLimiter(100), clock=lambda: next(ticks, 100.25))
    TestClient(create_app(rig.settings, services)).post("/ask", json={"question": "hi"})
    [line] = log_lines()
    assert line["latency_ms"] == 250


# =====================================================================================================
# privacy on the real HTTP surface
# =====================================================================================================

def everything_logged(caplog) -> str:
    return "\n".join(f"{r.name}|{r.levelname}|{r.getMessage()}|{r.exc_text}" for r in caplog.records)


def test_nothing_private_is_logged_on_any_path_by_default(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    rig = Rig(tmp_path, limit=4)
    for kwargs in [dict(json={"question": QUESTION}), dict(json={"question": QUESTION + "x" * 600}), dict(json={"question": QUESTION, "extra": 1}),
                   dict(content=QUESTION.encode()), dict(json={"question": QUESTION}), dict(json={"question": QUESTION})]:
        rig.client.post("/ask", **kwargs, headers={"x-forwarded-for": ADDRESS, "content-type": "application/json"})
    text = everything_logged(caplog)
    for secret in ("QUESTION-MARKER", ANSWER, PASSAGE, ADDRESS):
        assert secret not in text, secret


def test_a_question_is_logged_only_when_log_questions_is_on_and_then_only_that(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    Rig(tmp_path / "off").client.post("/ask", json={"question": QUESTION})
    assert "QUESTION-MARKER" not in everything_logged(caplog)
    caplog.clear()
    Rig(tmp_path / "on", log_questions=True).client.post("/ask", json={"question": QUESTION})
    [record] = [r for r in caplog.records if r.name == "askact.request"]
    fields = json.loads(record.getMessage())
    assert fields["question"] == QUESTION and ANSWER not in record.getMessage() and PASSAGE not in record.getMessage()


def test_an_error_response_never_contains_the_question_the_answer_or_a_passage(tmp_path):
    rigs = [Rig(tmp_path / "a", daily_spend_cap_usd=0.0), Rig(tmp_path / "b", relevance_threshold_cosine=None),
            Rig(tmp_path / "c", llm_price_table={})]
    for rig in rigs:
        response = rig.client.post("/ask", json={"question": QUESTION})
        assert response.status_code == 503 and "MARKER" not in response.text


def test_error_responses_do_not_carry_the_exception_text_or_a_traceback(tmp_path):
    from askact.api import IndexState

    class Exploding(Search):
        def __call__(self, query, config, k):
            raise RuntimeError(f"secret detail {QUESTION}")

    rig = Rig(tmp_path)
    services = Services(index=IndexState(Exploding()), llm=rig.llm, spend=SpendLedger.from_settings(rig.settings), limiter=RateLimiter(100))
    response = TestClient(create_app(rig.settings, services)).post("/ask", json={"question": QUESTION})
    assert response.status_code == 500 and "secret" not in response.text and "Traceback" not in response.text and "MARKER" not in response.text


def test_the_client_address_is_never_in_a_response_or_a_log(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    rig = Rig(tmp_path, trusted_proxy_header="X-Forwarded-For", limit=1)
    responses = [rig.client.post("/ask", json={"question": "hi"}, headers={"X-Forwarded-For": f"6.6.6.6, {ADDRESS}"}) for _ in range(3)]
    assert ADDRESS not in "".join(r.text + str(dict(r.headers)) for r in responses) and ADDRESS not in everything_logged(caplog)


# =====================================================================================================
# the rate limiter's client key, from real request headers
# =====================================================================================================

def test_without_a_trusted_header_every_client_shares_the_peer_address_bucket(tmp_path):
    rig = Rig(tmp_path, limit=2)
    codes = [rig.client.post("/ask", json={"question": "hi"}, headers={"X-Forwarded-For": f"9.9.9.{n}"}).status_code for n in range(4)]
    assert codes == [200, 200, 429, 429]    # a header nobody trusts cannot buy a fresh bucket


def test_with_a_trusted_header_the_last_entry_is_the_client(tmp_path):
    rig = Rig(tmp_path, trusted_proxy_header="X-Forwarded-For", limit=1)
    def go(entry):
        return rig.client.post("/ask", json={"question": "hi"}, headers={"X-Forwarded-For": entry}).status_code
    assert go("1.1.1.1, 198.51.100.1") == 200
    assert go("2.2.2.2, 198.51.100.1") == 429      # a different spoofed prefix, the same real client
    assert go("1.1.1.1, 198.51.100.2") == 200      # the same prefix, a different real client


def test_a_second_header_line_added_by_the_proxy_is_the_one_that_counts(tmp_path):
    rig = Rig(tmp_path, trusted_proxy_header="X-Forwarded-For", limit=1)
    def go(*lines):
        return rig.client.post("/ask", json={"question": "hi"}, headers=[("X-Forwarded-For", l) for l in lines]).status_code
    assert go("6.6.6.6", "198.51.100.1") == 200
    assert go("7.7.7.7", "198.51.100.1") == 429
    assert go("6.6.6.6", "198.51.100.2") == 200


def test_an_unusable_trusted_header_falls_back_to_the_peer_bucket(tmp_path):
    rig = Rig(tmp_path, trusted_proxy_header="X-Forwarded-For", limit=2)
    codes = [rig.client.post("/ask", json={"question": "hi"}, headers={"X-Forwarded-For": f"junk-{n}"}).status_code for n in range(4)]
    assert codes == [200, 200, 429, 429]


def test_the_configured_header_name_is_case_insensitive(tmp_path):
    rig = Rig(tmp_path, trusted_proxy_header="x-real-ip", limit=1)
    assert rig.client.post("/ask", json={"question": "hi"}, headers={"X-Real-IP": "198.51.100.1"}).status_code == 200
    assert rig.client.post("/ask", json={"question": "hi"}, headers={"X-REAL-IP": "198.51.100.1"}).status_code == 429
    assert rig.client.post("/ask", json={"question": "hi"}, headers={"X-Real-Ip": "198.51.100.2"}).status_code == 200


# =====================================================================================================
# CORS, paths and methods
# =====================================================================================================

def test_the_configured_origin_is_allowed_on_a_successful_response(tmp_path):
    response = Rig(tmp_path).client.post("/ask", json={"question": "hi"}, headers={"Origin": "http://localhost:3000"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_an_unlisted_origin_gets_no_cors_headers(tmp_path):
    response = Rig(tmp_path).client.post("/ask", json={"question": "hi"}, headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("kwargs", [dict(json={"question": "x" * 501}), dict(json={}), dict(content=b"x" * 100_000, headers=JSON)])
def test_error_responses_carry_cors_headers_so_the_browser_can_read_them(tmp_path, kwargs):
    kwargs = {**kwargs, "headers": {**kwargs.get("headers", {}), "Origin": "http://localhost:3000"}}
    response = Rig(tmp_path).client.post("/ask", **kwargs)
    assert response.status_code >= 400 and response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_a_rate_limited_response_carries_cors_and_exposes_retry_after(tmp_path):
    rig = Rig(tmp_path, limit=1)
    rig.client.post("/ask", json={"question": "hi"})
    response = rig.client.post("/ask", json={"question": "hi"}, headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 429 and response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "retry-after" in response.headers["access-control-expose-headers"].lower()


def test_the_preflight_request_is_answered_without_using_the_rate_limit(tmp_path):
    rig = Rig(tmp_path, limit=1)
    preflight = {"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
    for _ in range(5):
        response = rig.client.options("/ask", headers=preflight)
        assert response.status_code == 200 and response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert rig.client.post("/ask", json={"question": "hi"}).status_code == 200


def test_only_get_and_post_are_allowed_cross_origin(tmp_path):
    preflight = {"Origin": "http://localhost:3000", "Access-Control-Request-Method": "DELETE"}
    response = Rig(tmp_path).client.options("/ask", headers=preflight)
    assert response.status_code == 400


def test_several_origins_can_be_configured(tmp_path):
    rig = Rig(tmp_path, cors_origins=["https://app.example.org", "http://localhost:3000"])
    for origin in ("https://app.example.org", "http://localhost:3000"):
        assert rig.client.get("/health", headers={"Origin": origin}).headers["access-control-allow-origin"] == origin
    assert "access-control-allow-origin" not in rig.client.get("/health", headers={"Origin": "https://other.example"}).headers


def test_credentials_are_not_allowed_cross_origin(tmp_path):
    response = Rig(tmp_path).client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert "access-control-allow-credentials" not in response.headers


@pytest.mark.parametrize("path", ["/", "/ask/", "/health/", "/docs", "/redoc", "/openapi.json", "/health/x", "/admin", "/.env"])
def test_other_paths_are_a_plain_404_in_the_error_shape(tmp_path, path):
    response = Rig(tmp_path).client.get(path)
    assert response.status_code == 404 and response.json()["error"]["code"] == "not_found"


def test_a_trailing_slash_is_a_404_and_not_a_redirect_to_a_second_path_to_the_endpoint(tmp_path):
    rig = Rig(tmp_path, limit=1)
    client = TestClient(rig.client.app, follow_redirects=False)
    for method in ("POST", "GET"):
        response = client.request(method, "/ask/", json={"question": "hi"})
        assert response.status_code == 404 and "location" not in response.headers
    assert rig.untouched()


def test_the_schema_and_docs_are_not_exposed(tmp_path):
    rig = Rig(tmp_path)
    assert [rig.client.get(p).status_code for p in ("/docs", "/redoc", "/openapi.json")] == [404, 404, 404]


@pytest.mark.parametrize("method, path", [("GET", "/ask"), ("PUT", "/ask"), ("DELETE", "/ask"), ("POST", "/health"), ("PUT", "/health")])
def test_a_wrong_method_is_a_405_in_the_error_shape(tmp_path, method, path):
    response = Rig(tmp_path).client.request(method, path)
    assert response.status_code == 405 and response.json()["error"]["code"] == "method_not_allowed"


@pytest.mark.parametrize("method, path, status", [("POST", "/health", 405), ("POST", "/nope", 404), ("POST", "/ask/", 404),
                                                 ("POST", "/asking", 404), ("POST", "/ASK", 404), ("PUT", "/ask", 405)])
def test_only_post_to_exactly_ask_uses_the_rate_limit(tmp_path, method, path, status):
    rig = Rig(tmp_path, limit=1)
    assert rig.client.post("/ask", json={"question": "hi"}).status_code == 200      # the one allowed request is used up
    assert rig.client.request(method, path, json={}).status_code == status          # and these are not counted as /ask


def test_the_api_keys_clients_by_the_address_in_the_connection_scope(tmp_path):
    # The same check, through the real app: two different peers must not share a bucket.
    import asyncio

    rig = Rig(tmp_path, limit=1)
    app = create_app(rig.settings, Services(index=IndexState(rig.search), llm=rig.llm,
                                            spend=SpendLedger.from_settings(rig.settings), limiter=rig.limiter))

    async def call(peer):
        sent = []
        body = json.dumps({"question": "hi"}).encode()
        messages = [{"type": "http.request", "body": body, "more_body": False}]

        async def receive():
            return messages.pop(0) if messages else {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
                 "path": "/ask", "raw_path": b"/ask", "query_string": b"", "root_path": "", "server": ("testserver", 80),
                 "client": (peer, 5555), "state": {},
                 "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), (b"host", b"testserver")]}
        await app(scope, receive, send)
        return next(m["status"] for m in sent if m["type"] == "http.response.start")

    async def scenario():
        return [await call("198.51.100.1"), await call("198.51.100.2"), await call("198.51.100.1")]

    assert asyncio.run(scenario()) == [200, 200, 429]


def test_a_framework_400_on_some_other_path_is_not_reported_as_an_invalid_ask(tmp_path):
    from fastapi import HTTPException

    rig = Rig(tmp_path)
    app = create_app(rig.settings, Services(index=IndexState(rig.search), llm=rig.llm,
                                            spend=SpendLedger.from_settings(rig.settings), limiter=rig.limiter))

    @app.get("/probe")
    def probe():
        raise HTTPException(status_code=400, detail="bad")

    rig.client = TestClient(app)

    response = rig.client.get("/probe")
    assert response.status_code == 400 and response.json()["error"]["code"] == "http_error"


def test_a_wrong_method_on_ask_does_not_use_the_rate_limit_or_reach_anything(tmp_path):
    rig = Rig(tmp_path, limit=1)
    for _ in range(5):
        assert rig.client.get("/ask").status_code == 405
    assert rig.untouched() and rig.client.post("/ask", json={"question": "hi"}).status_code == 200


# =====================================================================================================
# the application is wired from settings, nothing is loaded at import
# =====================================================================================================

def test_importing_the_api_does_not_load_a_machine_learning_library():
    import subprocess
    import sys

    code = "import sys; import askact.api; print([m for m in ('torch', 'sentence_transformers', 'transformers') if m in sys.modules])"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_creating_the_app_loads_nothing_and_touches_no_disk(tmp_path):
    rig = Rig(tmp_path)
    assert not (tmp_path / "s.sqlite3").exists() and rig.search.calls == 0 and rig.llm.call_count == 0
