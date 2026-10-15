"""The API on the real retriever and a real index (built from the small fixture document with the stub embedder).

The other API tests use a stand-in for `Retriever.search`, which is what let a blank question through as a 500. These
run the real thing, so the seams between the API, the retriever, the answer pipeline and the ledger are exercised.
"""

import io
import json
import logging
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from askact.api import IndexState, Services, build_app, build_services, configure_logging, create_app
from askact.config import ModelPrice, Settings
from askact.generation.llm import FakeLLM, LLMReply
from askact.guards.ratelimit import RateLimiter
from askact.guards.spend import SpendLedger
from askact.index import INDEX_DIR
from askact.ingest.build import build_index
from askact.ingest.fetch import CACHE_FILENAME
from askact.ingest.parse import Counts
from askact.request_log import log_request
from askact.retrieval import Retriever

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_act.html"
GOOD = json.dumps({"answer": "Some practices are prohibited.", "citations": ["article:5"], "covered": True})


def settings_for(tmp_path, **overrides) -> Settings:
    values = dict(embedder="stub", reranker="stub", spend_db_path=tmp_path / "spend.sqlite3", daily_spend_cap_usd=1.0,
                  llm_model="m", llm_price_table={"m": ModelPrice(input_per_mtok=1.0, output_per_mtok=4.0)},
                  relevance_threshold_cosine=0.0, relevance_threshold_rerank=-100.0, max_output_tokens=100)
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def project(tmp_path):
    """A scratch project with a real index built from the fixture document."""
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / CACHE_FILENAME).write_bytes(FIXTURE.read_bytes())
    settings = settings_for(tmp_path)
    build_index(settings, raw_dir=raw, index_dir=tmp_path / "data" / "index", expected_counts=Counts(3, 2, 1))
    return tmp_path, settings


def client_for(tmp_path, settings, replies=None, index_dir=None) -> tuple[TestClient, FakeLLM]:
    retriever = Retriever.from_settings(settings, index_dir or tmp_path / "data" / "index")
    llm = FakeLLM([LLMReply(GOOD, 100, 10)] * 50 if replies is None else replies)
    services = Services(index=IndexState(retriever.search), llm=llm, spend=SpendLedger.from_settings(settings), limiter=RateLimiter(1000))
    return TestClient(create_app(settings, services)), llm


def test_a_question_goes_through_the_real_retriever_and_pipeline(project):
    tmp_path, settings = project
    client, llm = client_for(tmp_path, settings)
    response = client.post("/ask", json={"question": "Which AI practices shall be prohibited?"})
    body = response.json()
    assert response.status_code == 200 and body["status"] == "answered" and body["citations"] == ["article:5"]
    # The fixture splits Article 5 into several chunks, and every retrieved chunk of a cited section is a source.
    assert body["sources"] and {src["section_id"] for src in body["sources"]} == {"article:5"}
    first = body["sources"][0]
    assert (first["label"], first["title"]) == ("Article 5", "Prohibited AI practices")
    assert all(src["text"] and isinstance(src["score"], float) for src in body["sources"])
    assert [src["score"] for src in body["sources"]] == sorted((src["score"] for src in body["sources"]), reverse=True)   # best first
    assert llm.call_count == 1


def test_the_prompt_sent_to_the_model_contains_the_passages_the_real_retriever_found(project):
    tmp_path, settings = project
    client, llm = client_for(tmp_path, settings)
    client.post("/ask", json={"question": "Which AI practices shall be prohibited?"})
    [call] = llm.calls
    assert "[article:5] Article 5 — Prohibited AI practices" in call.user and "Which AI practices shall be prohibited?" in call.user


@pytest.mark.parametrize("question", ["", "   ", "\n"])
def test_a_blank_question_is_a_422_with_the_real_retriever_not_a_500(project, question):
    tmp_path, settings = project
    client, llm = client_for(tmp_path, settings)
    response = client.post("/ask", json={"question": question})
    assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_request" and llm.call_count == 0


@pytest.mark.parametrize("question", ["?", "!!!", "the", "日本語", "x" * 500, "Article 5", "a b c d e f g"])
def test_odd_but_non_blank_questions_do_not_crash_the_real_retriever(project, question):
    tmp_path, settings = project
    client, _ = client_for(tmp_path, settings)
    assert client.post("/ask", json={"question": question}).status_code == 200


def test_the_reranking_config_works_through_the_api_with_the_stub_reranker(project):
    tmp_path, _ = project
    settings = settings_for(tmp_path, reranker_enabled=True)
    client, llm = client_for(tmp_path, settings)
    response = client.post("/ask", json={"question": "Which AI practices shall be prohibited?"})
    assert response.status_code == 200 and llm.call_count == 1


def test_a_relevance_below_the_threshold_is_not_covered_through_the_real_retriever(project):
    tmp_path, _ = project
    settings = settings_for(tmp_path, relevance_threshold_cosine=0.999)
    client, llm = client_for(tmp_path, settings)
    body = client.post("/ask", json={"question": "What is the capital of France?"}).json()
    assert body["status"] == "not_covered" and llm.call_count == 0 and not (tmp_path / "spend.sqlite3").exists()


def test_the_cost_is_recorded_after_an_answer_through_the_real_stack(project):
    tmp_path, settings = project
    client, _ = client_for(tmp_path, settings, replies=[LLMReply(GOOD, 250, 30)])
    client.post("/ask", json={"question": "Which AI practices shall be prohibited?"})
    [(spent,)] = sqlite3.connect(tmp_path / "spend.sqlite3").execute("SELECT micro_usd FROM spend").fetchall()
    assert spent == 250 * 1 + 30 * 4


# --- startup: a missing or stale index does not crash the process ---------------------------------------------------

def test_a_missing_index_leaves_the_app_up_with_health_503_and_ask_index_unavailable(tmp_path):
    settings = settings_for(tmp_path)
    services = build_services_with_index_dir(settings, tmp_path / "no-index-here")
    client = TestClient(create_app(settings, services))
    assert client.get("/health").status_code == 503
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 503 and response.json()["error"]["code"] == "index_unavailable"
    assert "ingest" not in response.text and str(tmp_path) not in response.text    # the fix instructions are for the operator


def test_a_stale_index_is_refused_the_same_way(project):
    tmp_path, settings = project
    stale = settings_for(tmp_path, chunk_max_chars=1500)       # the index was built with the default chunking
    services = build_services_with_index_dir(stale, tmp_path / "data" / "index")
    client = TestClient(create_app(stale, services))
    assert client.get("/health").status_code == 503
    assert client.post("/ask", json={"question": "hi"}).json()["error"]["code"] == "index_unavailable"


def test_a_good_index_makes_health_ok(project):
    tmp_path, settings = project
    services = build_services_with_index_dir(settings, tmp_path / "data" / "index")
    assert TestClient(create_app(settings, services)).get("/health").json() == {"status": "ok", "index": "ready"}


def test_build_services_limits_requests_as_the_setting_says(project, monkeypatch):
    tmp_path, _ = project
    monkeypatch.chdir(tmp_path)
    services = build_services(settings_for(tmp_path, rate_limit_per_minute=7))
    assert sum(services.limiter.check("one-client").allowed for _ in range(50)) == 7


def test_build_services_gives_the_ledger_the_cap_and_prices_from_settings(project, monkeypatch):
    from askact.generation.answer import SpendRefusedError

    tmp_path, _ = project
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SpendRefusedError):
        build_services(settings_for(tmp_path, daily_spend_cap_usd=0.0)).spend.reserve("system", "user", 10)
    build_services(settings_for(tmp_path, daily_spend_cap_usd=1.0)).spend.reserve("system", "user", 10)
    with pytest.raises(SpendRefusedError):
        build_services(settings_for(tmp_path, daily_spend_cap_usd=1.0, llm_price_table={})).spend.reserve("system", "user", 10)


def test_a_missing_index_is_logged_once_at_startup_with_its_kind_and_no_path(tmp_path, caplog, monkeypatch):
    import logging

    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.INFO):
        build_services(settings_for(tmp_path))
    [record] = [r for r in caplog.records if r.name == "askact.api"]
    assert record.levelno == logging.ERROR and "missing" in record.getMessage() and "python -m askact.ingest" in record.getMessage()
    assert str(tmp_path) not in caplog.text


def test_a_stale_index_is_logged_as_stale(project, caplog, monkeypatch):
    import logging

    tmp_path, _ = project
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.INFO):
        build_services(settings_for(tmp_path, chunk_max_chars=1500))
    assert "index is stale" in caplog.text and str(tmp_path) not in caplog.text


def test_a_good_index_logs_nothing_at_startup(project, caplog, monkeypatch):
    import logging

    tmp_path, settings = project
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.INFO):
        build_services(settings)
    assert [r for r in caplog.records if r.name == "askact.api"] == []


def test_the_index_error_is_remembered_for_the_operator_but_not_sent_to_the_client(tmp_path):
    settings = settings_for(tmp_path)
    services = build_services_with_index_dir(settings, tmp_path / "nowhere")
    assert services.index.error is not None and "python -m askact.ingest" in str(services.index.error)


def build_services_with_index_dir(settings, index_dir) -> Services:
    """`build_services` loads from INDEX_DIR; point it elsewhere for the test by building the same pieces."""
    from askact.index import IndexUnavailableError

    try:
        index = IndexState(Retriever.from_settings(settings, index_dir).search)
    except IndexUnavailableError as exc:
        index = IndexState(None, exc)
    return Services(index=index, llm=FakeLLM([]), spend=SpendLedger.from_settings(settings), limiter=RateLimiter(1000))


def test_build_services_degrades_instead_of_raising_when_the_index_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)           # INDEX_DIR is relative: there is no data/index here
    services = build_services(settings_for(tmp_path))
    assert not services.index.ready and services.index.error is not None


def test_build_services_makes_the_real_parts_from_settings(project, monkeypatch):
    tmp_path, settings = project
    monkeypatch.chdir(tmp_path)
    services = build_services(settings)
    assert services.index.ready and isinstance(services.spend, SpendLedger) and isinstance(services.limiter, RateLimiter)


def test_an_llm_that_is_not_configured_is_an_llm_unavailable_answer_not_a_startup_crash(project, monkeypatch):
    tmp_path, settings = project
    monkeypatch.chdir(tmp_path)
    unconfigured = settings.model_copy(update={"llm_provider": None})
    client = TestClient(create_app(unconfigured, build_services(unconfigured)))
    assert client.get("/health").status_code == 200                         # the app is up
    response = client.post("/ask", json={"question": "Which AI practices shall be prohibited?"})
    assert response.status_code == 503 and response.json()["error"]["code"] == "llm_unavailable"
    assert "LLM_PROVIDER" not in response.text                                # the instruction is for the operator


def test_the_deferred_llm_is_built_once_not_per_request(project, monkeypatch):
    tmp_path, settings = project
    monkeypatch.chdir(tmp_path)
    built = []
    real = FakeLLM([LLMReply(GOOD, 10, 5)] * 5)

    def fake_get_llm(s):
        built.append(1)
        return real

    monkeypatch.setattr("askact.api.get_llm", fake_get_llm)
    client = TestClient(create_app(settings, build_services(settings)))
    for _ in range(3):
        assert client.post("/ask", json={"question": "Which AI practices shall be prohibited?"}).status_code == 200
    assert len(built) == 1 and real.call_count == 3


def test_build_app_reads_settings_from_the_environment(project, monkeypatch):
    tmp_path, settings = project
    monkeypatch.chdir(tmp_path)
    for name, value in {"EMBEDDER": "stub", "RERANKER": "stub", "LLM_PROVIDER": "fake", "LLM_MODEL": "m",
                        "SPEND_DB_PATH": str(tmp_path / "s.sqlite3"), "RELEVANCE_THRESHOLD_COSINE": "0.0",
                        "CORS_ORIGINS": "https://app.example.org"}.items():
        monkeypatch.setenv(name, value)
    client = TestClient(build_app())
    assert client.get("/health", headers={"Origin": "https://app.example.org"}).headers["access-control-allow-origin"] == "https://app.example.org"
    assert INDEX_DIR == Path("data/index")


# --- logging in the running process ---------------------------------------------------------------------------

def test_configure_logging_makes_the_request_line_appear_on_the_stream():
    # Found while writing the Dockerfile: the request line is logged at INFO, Python's default handler prints
    # WARNING and above, and nothing else installs a handler under uvicorn, so a container showed no request lines.
    stream = io.StringIO()
    configure_logging(stream)
    log_request(outcome="answered", config="hybrid", latency_ms=12.3, input_tokens=5, output_tokens=7)
    [line] = stream.getvalue().splitlines()
    record = json.loads(line)  # the bare message: no logger name, level or timestamp prefix in front of the JSON
    assert record["event"] == "ask" and record["outcome"] == "answered" and record["input_tokens"] == 5


def test_configure_logging_also_shows_the_startup_error_and_nothing_from_other_libraries():
    stream = io.StringIO()
    configure_logging(stream)
    logging.getLogger("askact.api").error("the search index is missing")
    logging.getLogger("httpx").info("HTTP Request: POST https://example.invalid 200 OK")
    logging.getLogger("huggingface_hub").info("Downloading model")
    assert stream.getvalue() == "the search index is missing\n"


def test_calling_configure_logging_again_replaces_its_handler_so_lines_are_not_doubled():
    first, second = io.StringIO(), io.StringIO()
    configure_logging(first)
    configure_logging(second)
    log_request(outcome="answered", config="hybrid", latency_ms=1.0)
    assert first.getvalue() == "" and len(second.getvalue().splitlines()) == 1


def test_configure_logging_leaves_handlers_that_are_not_its_own(caplog):
    # pytest's caplog handler is on the root logger and must keep working: propagation stays on.
    configure_logging(io.StringIO())
    with caplog.at_level(logging.INFO, logger="askact.request"):
        log_request(outcome="answered", config="hybrid", latency_ms=1.0)
    assert len(caplog.records) == 1


def test_build_app_installs_the_logging(project, monkeypatch):
    tmp_path, _ = project
    monkeypatch.chdir(tmp_path)
    for name, value in {"EMBEDDER": "stub", "RERANKER": "stub", "LLM_PROVIDER": "fake", "SPEND_DB_PATH": str(tmp_path / "s.sqlite3")}.items():
        monkeypatch.setenv(name, value)
    before = [h for h in logging.getLogger("askact").handlers if getattr(h, "_askact_stderr", False)]
    build_app()
    after = [h for h in logging.getLogger("askact").handlers if getattr(h, "_askact_stderr", False)]
    assert before == [] and len(after) == 1


def test_create_app_does_not_touch_logging(project):
    # Tests build the app through create_app; if it changed global logger state they would leak into each other.
    tmp_path, settings = project
    before = list(logging.getLogger("askact").handlers)
    create_app(settings, Services(index=IndexState(None), llm=FakeLLM(), spend=SpendLedger.from_settings(settings), limiter=RateLimiter(10)))
    assert list(logging.getLogger("askact").handlers) == before


def test_in_a_real_process_the_request_line_reaches_stderr_only_after_configure_logging():
    code = (
        "import sys; from askact.request_log import log_request\n"
        "from askact.api import configure_logging\n"
        "if sys.argv[1] == 'configured': configure_logging()\n"
        "log_request(outcome='answered', config='hybrid', latency_ms=1.0)\n"
    )
    env = {**os.environ, "EMBEDDER": "stub", "RERANKER": "stub"}
    plain = subprocess.run([sys.executable, "-c", code, "plain"], env=env, capture_output=True, text=True)
    configured = subprocess.run([sys.executable, "-c", code, "configured"], env=env, capture_output=True, text=True)
    assert plain.returncode == 0 and plain.stderr == "" and plain.stdout == ""  # the bug this fixes, kept visible
    assert json.loads(configured.stderr)["outcome"] == "answered" and configured.stdout == ""
