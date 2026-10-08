import re
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsError

from askact.config import DEFAULT_SOURCE_URL, ModelPrice, Settings

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"
# Documented in .env.example but read by the Next.js build, not by Python.
FRONTEND_ONLY = {"NEXT_PUBLIC_API_URL"}
ENV_NAMES = [name.upper() for name in Settings.model_fields]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """CI sets EMBEDDER=stub etc. Tests here must see the real defaults, not the ambient env."""
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def load(monkeypatch, **env: str) -> Settings:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return Settings(_env_file=None)  # never read a developer's real .env in tests


def test_defaults_match_the_design(monkeypatch):
    s = load(monkeypatch)
    assert s.source_url == DEFAULT_SOURCE_URL
    assert s.source_fallback_path == Path("data/raw/ai-act-oj-2024-1689.manual.html")
    assert s.chunk_max_chars == 1800
    assert (s.embedder, s.reranker) == ("real", "real")
    assert s.reranker_enabled is False
    assert s.top_k == 5
    assert s.max_output_tokens == 500
    assert s.max_question_chars == 500
    assert s.rate_limit_per_minute == 10
    assert s.trusted_proxy_header is None
    assert s.daily_spend_cap_usd == 1.00
    assert s.spend_db_path == Path("/var/lib/askact/spend.sqlite3")
    assert s.cors_origins == ["http://localhost:3000"]
    assert s.eval_max_llm_calls == 250
    assert s.log_questions is False


def test_llm_settings_have_no_default(monkeypatch):
    s = load(monkeypatch)
    assert s.llm_provider is None
    assert s.llm_model is None
    assert s.llm_api_key is None
    assert s.llm_base_url is None
    assert s.llm_price_table == {}


def test_relevance_thresholds_are_none_until_tuned(monkeypatch):
    s = load(monkeypatch)
    assert s.relevance_threshold_cosine is None
    assert s.relevance_threshold_rerank is None


def test_blank_value_counts_as_unset(monkeypatch):
    s = load(monkeypatch, RELEVANCE_THRESHOLD_COSINE="", LLM_PROVIDER="", TOP_K="")
    assert s.relevance_threshold_cosine is None
    assert s.llm_provider is None
    assert s.top_k == 5


def test_env_overrides_are_converted_to_the_right_types(monkeypatch):
    s = load(
        monkeypatch,
        CHUNK_MAX_CHARS="900",
        RERANKER_ENABLED="true",
        EMBEDDER="stub",
        RERANKER="stub",
        RELEVANCE_THRESHOLD_COSINE="0.42",
        RELEVANCE_THRESHOLD_RERANK="-1.5",
        DAILY_SPEND_CAP_USD="2.5",
        SPEND_DB_PATH="/tmp/spend.sqlite3",
        LOG_QUESTIONS="true",
        TRUSTED_PROXY_HEADER="X-Forwarded-For",
    )
    assert s.chunk_max_chars == 900
    assert s.reranker_enabled is True
    assert (s.embedder, s.reranker) == ("stub", "stub")
    assert s.relevance_threshold_cosine == 0.42
    assert s.relevance_threshold_rerank == -1.5  # reranker scores can be negative
    assert s.daily_spend_cap_usd == 2.5
    assert s.spend_db_path == Path("/tmp/spend.sqlite3")
    assert s.log_questions is True
    assert s.trusted_proxy_header == "X-Forwarded-For"


def test_price_table_is_parsed_from_json(monkeypatch):
    table = '{"some-model": {"input_per_mtok": 1.5, "output_per_mtok": 6}}'
    s = load(monkeypatch, LLM_PRICE_TABLE=table)
    assert s.llm_price_table == {"some-model": ModelPrice(input_per_mtok=1.5, output_per_mtok=6)}


@pytest.mark.parametrize("raw, expected", [
    ("http://a.example", ["http://a.example"]),
    ("http://a.example, http://b.example ,", ["http://a.example", "http://b.example"]),
])
def test_cors_origins_are_comma_separated(monkeypatch, raw, expected):
    assert load(monkeypatch, CORS_ORIGINS=raw).cors_origins == expected


def test_api_key_is_kept_out_of_repr(monkeypatch):
    s = load(monkeypatch, LLM_API_KEY="sk-test-not-a-real-key")
    assert s.llm_api_key.get_secret_value() == "sk-test-not-a-real-key"
    assert "sk-test-not-a-real-key" not in repr(s)
    assert "sk-test-not-a-real-key" not in str(s)


@pytest.mark.parametrize("name, value", [
    ("CHUNK_MAX_CHARS", "0"),
    ("TOP_K", "-1"),
    ("MAX_OUTPUT_TOKENS", "0"),
    ("MAX_QUESTION_CHARS", "-5"),
    ("RATE_LIMIT_PER_MINUTE", "0"),
    ("DAILY_SPEND_CAP_USD", "-0.5"),
    ("EVAL_MAX_LLM_CALLS", "-1"),
    ("EMBEDDER", "cloud"),
    ("RERANKER", "cloud"),
    ("LOG_QUESTIONS", "maybe"),
    ("RELEVANCE_THRESHOLD_COSINE", "high"),
    ("LLM_PRICE_TABLE", '{"m": {"input_per_mtok": -1, "output_per_mtok": 1}}'),
    ("LLM_PRICE_TABLE", '{"m": {"input_per_mtok": 1}}'),
    ("LLM_PRICE_TABLE", "{not json"),
])
def test_invalid_values_raise_an_error_naming_the_variable(monkeypatch, name, value):
    with pytest.raises((ValidationError, SettingsError)) as excinfo:
        load(monkeypatch, **{name: value})
    assert name.lower() in str(excinfo.value).lower()


# --- .env.example must document exactly what config.py reads -------------------------

def env_example_entries() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text().splitlines():
        match = re.fullmatch(r"([A-Z][A-Z0-9_]*)=(.*)", line)
        if match:
            assert match.group(1) not in entries, f"{match.group(1)} listed twice"
            entries[match.group(1)] = match.group(2)
    return entries


def test_env_example_lists_exactly_the_variables_config_defines():
    assert set(env_example_entries()) == set(ENV_NAMES) | FRONTEND_ONLY


def test_loading_env_example_gives_the_built_in_defaults():
    """The example's values must not drift from the code's defaults."""
    assert Settings(_env_file=ENV_EXAMPLE) == Settings(_env_file=None)


def test_env_example_contains_no_secret():
    assert env_example_entries()["LLM_API_KEY"] == ""
