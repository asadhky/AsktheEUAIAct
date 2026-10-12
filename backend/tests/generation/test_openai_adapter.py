import json
import logging

import httpx
import pytest

from askact.config import Settings
from askact.generation.llm import FakeLLM, LLMReply, LLMUnavailableError, get_llm
from askact.generation.openai_adapter import DEFAULT_BASE_URL, OpenAIChat

KEY = "sk-test-SECRET-KEY-0123456789"


def completion(content: object = "Hello!", prompt_tokens=19, completion_tokens=7, **extra) -> dict:
    """A response in the shape OpenAI documents for createChatCompletion."""
    body = {
        "id": "chatcmpl-123", "object": "chat.completion", "model": "some-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content, "refusal": None}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens, "completion_tokens_details": {"reasoning_tokens": 0}},
    }
    body.update(extra)
    return body


class Server:
    """A mock provider: records the requests it gets and answers with a fixed response (or raises)."""

    def __init__(self, response: httpx.Response | Exception | None = None):
        self.requests: list[httpx.Request] = []
        self.response = response if response is not None else httpx.Response(200, json=completion())

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def make(server: Server, **overrides) -> OpenAIChat:
    options = {"api_key": KEY, "model": "some-model", **overrides}
    return OpenAIChat(client=server.client(), **options)


def sent(server: Server) -> dict:
    [request] = server.requests
    return json.loads(request.content)


def error_body(status: int, code: str | None = None, type_: str | None = None, message="free text from the provider") -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message, "type": type_, "param": None, "code": code}})


def failure(server: Server) -> str:
    with pytest.raises(LLMUnavailableError) as info:
        make(server).complete("sys", "usr", 100)
    assert info.value.code == "llm_unavailable"
    return str(info.value)


# --- the request: what is sent, checked against OpenAI's documentation -----------------------

def test_the_request_goes_to_chat_completions_with_a_bearer_key_and_json():
    server = Server()
    make(server).complete("sys", "usr", 100)
    [request] = server.requests
    assert request.method == "POST"
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert request.headers["Content-Type"] == "application/json"


def test_the_body_has_the_model_the_two_messages_and_the_output_cap_under_the_current_parameter_name():
    server = Server()
    make(server, model="my-model").complete("the system prompt", "the user prompt", 321)
    assert sent(server) == {
        "model": "my-model",
        "messages": [{"role": "system", "content": "the system prompt"}, {"role": "user", "content": "the user prompt"}],
        "max_completion_tokens": 321,
    }


def test_the_deprecated_max_tokens_parameter_is_never_sent():
    # It is deprecated and rejected by o-series models (OpenAI's OpenAPI spec).
    server = Server()
    make(server).complete("s", "u", 50)
    assert "max_tokens" not in sent(server)


def test_no_optional_parameters_are_sent_that_some_models_reject():
    server = Server()
    make(server).complete("s", "u", 50)
    assert set(sent(server)) == {"model", "messages", "max_completion_tokens"}  # no temperature, response_format, stream, n, ...


def test_the_output_cap_is_passed_through_exactly():
    for cap in (1, 500, 4096):
        server = Server()
        make(server).complete("s", "u", cap)
        assert sent(server)["max_completion_tokens"] == cap


def test_prompts_with_non_ascii_text_and_newlines_survive_the_round_trip():
    server = Server()
    make(server).complete("Système — règles\n1. ok", "Question: „Hochrisiko“ ✓", 10)
    messages = sent(server)["messages"]
    assert messages[0]["content"] == "Système — règles\n1. ok" and messages[1]["content"] == "Question: „Hochrisiko“ ✓"


@pytest.mark.parametrize("base_url, expected", [
    ("http://localhost:8000/v1", "http://localhost:8000/v1/chat/completions"),
    ("http://localhost:8000/v1/", "http://localhost:8000/v1/chat/completions"),
    ("https://example.test/openai/v1///", "https://example.test/openai/v1/chat/completions"),
])
def test_a_custom_base_url_replaces_the_default_and_a_trailing_slash_is_harmless(base_url, expected):
    server = Server()
    make(server, base_url=base_url).complete("s", "u", 10)
    assert str(server.requests[0].url) == expected


def test_the_default_base_url_is_openais():
    assert DEFAULT_BASE_URL == "https://api.openai.com/v1"


def test_a_request_is_made_exactly_once_with_no_retries():
    server = Server(httpx.Response(503, json={"error": {"code": "server_is_overloaded"}}))
    with pytest.raises(LLMUnavailableError):
        make(server).complete("s", "u", 10)
    assert len(server.requests) == 1  # a retry could bill twice, behind the spend guard's back


def test_redirects_are_not_followed_so_the_key_cannot_be_forwarded():
    server = Server(httpx.Response(307, headers={"Location": "https://evil.example/steal"}))
    with pytest.raises(LLMUnavailableError):
        make(server).complete("s", "u", 10)
    assert len(server.requests) == 1


def test_the_real_client_does_not_follow_redirects_and_has_a_timeout():
    client = OpenAIChat(api_key=KEY, model="m")._client
    assert client.follow_redirects is False
    assert client.timeout.read is not None and client.timeout.connect is not None


# --- the response: text and usage -----------------------------------------------------------

def test_the_reply_text_and_token_usage_are_read():
    reply = make(Server(httpx.Response(200, json=completion("The answer.", 900, 42)))).complete("s", "u", 100)
    assert reply == LLMReply("The answer.", input_tokens=900, output_tokens=42)


def test_the_text_is_returned_exactly_as_sent_including_whitespace():
    reply = make(Server(httpx.Response(200, json=completion("  {\"a\": 1}\n")))).complete("s", "u", 100)
    assert reply.text == '  {"a": 1}\n'


def test_a_reply_without_usage_has_none_token_counts_not_zero():
    body = completion()
    del body["usage"]
    reply = make(Server(httpx.Response(200, json=body))).complete("s", "u", 100)
    assert (reply.text, reply.input_tokens, reply.output_tokens) == ("Hello!", None, None)


@pytest.mark.parametrize("usage", [
    None, [], "lots", {}, {"prompt_tokens": None, "completion_tokens": None},
    {"prompt_tokens": "19", "completion_tokens": "7"}, {"prompt_tokens": 1.5, "completion_tokens": 2.5},
    {"prompt_tokens": True, "completion_tokens": False}, {"prompt_tokens": -3, "completion_tokens": -1},
])
def test_usage_that_is_not_a_plain_count_is_treated_as_not_reported(usage):
    body = completion()
    body["usage"] = usage
    reply = make(Server(httpx.Response(200, json=body))).complete("s", "u", 100)
    assert (reply.input_tokens, reply.output_tokens) == (None, None)


def test_one_count_can_be_missing_without_losing_the_other():
    body = completion()
    body["usage"] = {"prompt_tokens": 55}
    reply = make(Server(httpx.Response(200, json=body))).complete("s", "u", 100)
    assert (reply.input_tokens, reply.output_tokens) == (55, None)


def test_zero_is_a_valid_count():
    reply = make(Server(httpx.Response(200, json=completion("x", 0, 0)))).complete("s", "u", 100)
    assert (reply.input_tokens, reply.output_tokens) == (0, 0)


def test_the_output_tokens_reported_are_the_completion_tokens_which_include_reasoning():
    body = completion("short", 100, 480)
    body["usage"]["completion_tokens_details"] = {"reasoning_tokens": 450}
    assert make(Server(httpx.Response(200, json=body))).complete("s", "u", 500).output_tokens == 480


def test_a_refusal_has_null_content_and_comes_back_as_empty_text_with_its_usage_still_reported():
    body = completion(None, 300, 12)
    body["choices"][0]["message"]["refusal"] = "I can't help with that."
    reply = make(Server(httpx.Response(200, json=body))).complete("s", "u", 100)
    assert (reply.text, reply.input_tokens, reply.output_tokens) == ("", 300, 12)  # billed tokens must reach the ledger


def test_empty_content_when_the_budget_was_spent_thinking_keeps_its_usage():
    body = completion("", 300, 500)
    body["choices"][0]["finish_reason"] = "length"
    reply = make(Server(httpx.Response(200, json=body))).complete("s", "u", 500)
    assert (reply.text, reply.output_tokens) == ("", 500)


def test_only_the_first_choice_is_used():
    body = completion("first")
    body["choices"].append({"index": 1, "message": {"role": "assistant", "content": "second"}, "finish_reason": "stop"})
    assert make(Server(httpx.Response(200, json=body))).complete("s", "u", 100).text == "first"


@pytest.mark.parametrize("content", [["a", "list"], {"a": "dict"}, 5, True])
def test_content_that_is_not_text_is_treated_as_empty(content):
    assert make(Server(httpx.Response(200, json=completion(content)))).complete("s", "u", 100).text == ""


# --- an answer that is not a chat completion at all ----------------------------------------

@pytest.mark.parametrize("response", [
    httpx.Response(200, text="<html>Welcome to nginx</html>"),
    httpx.Response(200, text=""),
    httpx.Response(200, json=[]),
    httpx.Response(200, json="ok"),
    httpx.Response(200, json={}),
    httpx.Response(200, json={"choices": []}),
    httpx.Response(200, json={"choices": "none"}),
    httpx.Response(200, json={"choices": [None]}),
    httpx.Response(200, json={"choices": [{"message": None}]}),
    httpx.Response(200, json={"choices": [{"message": "text"}]}),
    httpx.Response(200, json={"choices": [{}]}),
])
def test_a_200_that_is_not_a_chat_completion_points_at_the_base_url(response):
    assert "not a chat completion: check LLM_BASE_URL" in failure(Server(response))


# --- HTTP errors become llm_unavailable with a message that helps and leaks nothing ------------

def test_401_says_the_key_was_rejected():
    message = failure(Server(error_body(401, "invalid_api_key", "invalid_request_error")))
    assert "rejected the API key" in message and "HTTP 401" in message and "invalid_api_key" in message and "LLM_API_KEY" in message


def test_403_says_access_was_refused():
    message = failure(Server(error_body(403, "unsupported_country_region_territory")))
    assert "refused access" in message and "HTTP 403" in message and "unsupported_country_region_territory" in message


def test_404_points_at_the_model_and_the_base_url():
    message = failure(Server(error_body(404, "model_not_found")))
    assert "does not know this model" in message and "LLM_MODEL" in message and "LLM_BASE_URL" in message


def test_400_points_at_the_model_and_the_base_url():
    message = failure(Server(error_body(400, "unsupported_parameter")))
    assert "rejected the request" in message and "HTTP 400" in message and "unsupported_parameter" in message


@pytest.mark.parametrize("code", ["rate_limit_exceeded", "slow_down", None])
def test_a_429_that_is_about_speed_says_to_try_again_shortly(code):
    message = failure(Server(error_body(429, code, "rate_limit_error")))
    assert "rate limiting" in message and "try again shortly" in message and "out of credit" not in message


@pytest.mark.parametrize("code", [
    "credit_balance_exhausted", "insufficient_quota", "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded", "organization_usage_limit_exceeded",
])
def test_a_429_that_is_about_money_says_retrying_will_not_help(code):
    message = failure(Server(error_body(429, code, "some_other_type")))  # the code alone must be enough
    assert "out of credit or over a spending limit" in message and "retrying will not help" in message
    assert "try again shortly" not in message


def test_a_429_with_a_neutral_type_and_no_money_code_is_only_about_speed():
    message = failure(Server(error_body(429, "rate_limit_exceeded", "some_other_type")))
    assert "rate limiting" in message and "out of credit" not in message


def test_a_429_is_about_money_if_only_the_type_says_so():
    # The guide says billing errors can carry the broad type insufficient_quota with a more specific code.
    message = failure(Server(error_body(429, "something_new", "insufficient_quota")))
    assert "out of credit" in message


@pytest.mark.parametrize("status, code", [(500, None), (502, None), (503, "server_is_overloaded"), (504, None)])
def test_5xx_says_the_provider_had_a_problem(status, code):
    message = failure(Server(error_body(status, code, "service_unavailable_error")))
    assert "the provider had a problem" in message and f"HTTP {status}" in message


@pytest.mark.parametrize("status", [301, 302, 307, 308, 402, 409, 418, 422])
def test_any_other_status_is_still_llm_unavailable(status):
    assert f"HTTP {status}" in failure(Server(httpx.Response(status, text="whatever")))


def test_a_200_is_the_only_success():
    for status in (201, 202, 204):
        with pytest.raises(LLMUnavailableError):
            make(Server(httpx.Response(status, json=completion()))).complete("s", "u", 10)


@pytest.mark.parametrize("response", [
    httpx.Response(401, text="not json"),
    httpx.Response(401, json=["a", "list"]),
    httpx.Response(401, json={"error": "a string, not an object"}),
    httpx.Response(401, json={"error": {"code": 12345}}),
    httpx.Response(401, json={"error": None}),
    httpx.Response(401, content=b""),
])
def test_an_error_body_in_an_unexpected_shape_still_gives_a_clean_error(response):
    assert "rejected the API key" in failure(Server(response))


# --- transport failures -------------------------------------------------------------------------

@pytest.mark.parametrize("exc", [httpx.ReadTimeout("slow"), httpx.ConnectTimeout("slow"), httpx.PoolTimeout("slow")])
def test_a_timeout_says_the_provider_did_not_answer_in_time(exc):
    assert "did not answer in time" in failure(Server(exc))


@pytest.mark.parametrize("exc", [
    httpx.ConnectError("detail-one"), httpx.ReadError("detail-two"), httpx.RemoteProtocolError("detail-three"),
    httpx.UnsupportedProtocol("detail-four"),
])
def test_a_network_failure_says_the_provider_could_not_be_reached_with_only_the_error_type(exc):
    message = failure(Server(exc))
    assert "could not reach the provider" in message and type(exc).__name__ in message
    assert str(exc) not in message  # the text of a transport error can contain a URL


# --- the key never appears anywhere it should not ------------------------------------------------

LEAKY = [
    httpx.ConnectError(f"failed with header Authorization: Bearer {KEY}"),
    httpx.ReadTimeout(f"timed out talking to https://x/?key={KEY}"),
    error_body(401, "invalid_api_key", message=f"Incorrect API key provided: {KEY}. You can find your key..."),
    error_body(400, "bad", message=f"echoing your prompt {KEY}"),
    error_body(429, "rate_limit_exceeded", message=f"limit for {KEY}"),
    error_body(500, f"{KEY}"),  # a code that is not a short identifier at all
    httpx.Response(200, text=f"<html>{KEY}</html>"),
]


@pytest.mark.parametrize("response", LEAKY, ids=lambda r: type(r).__name__)
def test_no_error_message_contains_the_key_or_the_providers_free_text(response):
    message = failure(Server(response))
    assert KEY not in message and "SECRET" not in message
    assert "free text from the provider" not in message and "echoing your prompt" not in message


@pytest.mark.parametrize("echo", [KEY, KEY[:12], KEY[8:20], f"{KEY}-extra", f"prefix-{KEY}"])
def test_an_error_code_that_contains_or_is_a_fragment_of_the_key_is_left_out(echo):
    message = failure(Server(error_body(401, echo)))
    assert echo not in message and "HTTP 401" in message


def test_a_short_fragment_of_the_key_is_not_enough_to_hide_an_ordinary_code():
    # Only fragments of 8+ characters count, otherwise common words that happen to be in a key would hide real codes.
    llm_key = "sk-model_not_found-and-more-0123456789"
    server = Server(error_body(404, "model_not_found"))
    with pytest.raises(LLMUnavailableError) as info:
        OpenAIChat(api_key=llm_key, model="m", client=server.client()).complete("s", "u", 10)
    assert "model_not_found" not in str(info.value)  # 15 characters, inside the key: hidden

    short_key = "sk-abc-0123"
    server = Server(error_body(401, "abc"))
    with pytest.raises(LLMUnavailableError) as info:
        OpenAIChat(api_key=short_key, model="m", client=server.client()).complete("s", "u", 10)
    assert "(abc)" in str(info.value)  # 3 characters, also inside the key, but too short to count as a fragment


def test_a_provider_error_code_that_is_not_a_short_identifier_is_left_out():
    message = failure(Server(error_body(400, "x" * 200)))
    assert "x" * 200 not in message and "HTTP 400" in message
    message = failure(Server(error_body(400, "has spaces and ' quotes")))
    assert "has spaces" not in message


@pytest.mark.parametrize("response", [
    httpx.ConnectError(f"Authorization: Bearer {KEY}"),
    httpx.ReadTimeout(f"key={KEY}"),
    error_body(401, "invalid_api_key"),
    httpx.Response(200, text="not json"),
], ids=["connect", "timeout", "http-error", "bad-body"])
def test_the_raised_error_does_not_chain_the_original_exception(response):
    # `from None` matters: a chained exception is printed in full by tracebacks and logs, and a transport
    # error's text can contain the key.
    with pytest.raises(LLMUnavailableError) as info:
        make(Server(response)).complete("s", "u", 10)
    error = info.value
    # A traceback prints a chain if there is a __cause__, or a __context__ that is not suppressed.
    assert error.__cause__ is None
    assert error.__context__ is None or error.__suppress_context__


def test_the_key_is_not_in_the_repr_or_str_of_the_adapter():
    llm = make(Server(), base_url="https://user:pass@example.test/v1")
    for text in (repr(llm), str(llm), f"{llm!r}"):
        assert KEY not in text and "pass" not in text and "example.test" not in text
    assert "some-model" in repr(llm)


def test_nothing_is_logged_during_a_failing_call(caplog):
    with caplog.at_level(logging.DEBUG):
        failure(Server(error_body(401, "invalid_api_key", message=KEY)))
    assert KEY not in caplog.text


# --- building the provider from settings (R3.10) ---------------------------------------------------

def openai_settings(**overrides) -> Settings:
    values = {"llm_provider": "openai", "llm_api_key": KEY, "llm_model": "some-model"}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def unavailable(settings: Settings) -> str:
    with pytest.raises(LLMUnavailableError) as info:
        get_llm(settings)
    assert info.value.code == "llm_unavailable"
    assert KEY not in str(info.value)
    return str(info.value)


def test_openai_is_built_from_the_key_and_model_in_settings():
    llm = get_llm(openai_settings())
    assert isinstance(llm, OpenAIChat) and llm.model == "some-model"
    assert llm._url == "https://api.openai.com/v1/chat/completions"


def test_the_base_url_setting_is_used():
    assert get_llm(openai_settings(llm_base_url="http://localhost:11434/v1"))._url == "http://localhost:11434/v1/chat/completions"  # type: ignore[attr-defined]


def test_an_empty_base_url_means_the_default():
    assert get_llm(openai_settings(llm_base_url="   "))._url == DEFAULT_BASE_URL + "/chat/completions"  # type: ignore[attr-defined]


def test_a_missing_key_names_the_variable():
    assert "LLM_API_KEY is not set" in unavailable(openai_settings(llm_api_key=None))


@pytest.mark.parametrize("model", [None, "", "   "])
def test_a_missing_model_names_the_variable(model):
    assert "LLM_MODEL is not set" in unavailable(openai_settings(llm_model=model))


def test_both_missing_are_named_together():
    message = unavailable(openai_settings(llm_api_key=None, llm_model=None))
    assert "LLM_API_KEY is not set" in message and "LLM_MODEL is not set" in message


@pytest.mark.parametrize("key", ["", "   ", "\n"])
def test_a_blank_key_counts_as_missing(key):
    assert "LLM_API_KEY is not set" in unavailable(openai_settings(llm_api_key=key))


@pytest.mark.parametrize("key", ["sk-abc def", "sk-abc\tdef", "sk-abc\ndef", "sk-\x00abc"])
def test_a_key_with_spaces_or_control_characters_is_refused_without_echoing_it(key):
    message = unavailable(openai_settings(llm_api_key=key))
    assert "spaces or control characters" in message and "sk-abc" not in message


def test_a_key_pasted_with_surrounding_whitespace_is_trimmed():
    # A trailing newline from a copy-paste would otherwise be an invalid header.
    server = Server()
    llm = get_llm(openai_settings(llm_api_key=f"  {KEY}\n"))
    assert isinstance(llm, OpenAIChat)
    llm._client = server.client()
    llm.complete("s", "u", 10)
    assert server.requests[0].headers["Authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize("url", ["localhost:8000/v1", "ftp://example.test", "api.openai.com/v1"])
def test_a_base_url_without_http_or_https_is_refused(url):
    assert "LLM_BASE_URL must start with http:// or https://" in unavailable(openai_settings(llm_base_url=url))


def test_a_model_name_is_trimmed():
    assert get_llm(openai_settings(llm_model="  some-model  ")).model == "some-model"  # type: ignore[attr-defined]


# --- the provider is chosen by an environment variable, with no other change (R3.9) ---------------------

def test_the_same_settings_give_the_fake_or_the_openai_provider_depending_only_on_llm_provider():
    base = {"llm_api_key": KEY, "llm_model": "some-model"}
    assert isinstance(get_llm(Settings(_env_file=None, llm_provider="fake", **base)), FakeLLM)
    assert isinstance(get_llm(Settings(_env_file=None, llm_provider="openai", **base)), OpenAIChat)


def test_the_provider_is_chosen_by_the_environment_variable(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", KEY)
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert isinstance(get_llm(Settings(_env_file=None)), OpenAIChat)
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    assert isinstance(get_llm(Settings(_env_file=None)), FakeLLM)


@pytest.mark.parametrize("provider", ["anthropic", "gemini", "azure", "ollama"])
def test_any_other_provider_is_rejected_with_the_configuration_error(provider):
    message = unavailable(openai_settings(llm_provider=provider))
    assert f"{provider!r} is not supported" in message and "fake, openai" in message


@pytest.mark.parametrize("written", ["OpenAI", "OPENAI", "  openai  "])
def test_the_provider_name_ignores_case_and_spaces(written):
    assert isinstance(get_llm(openai_settings(llm_provider=written)), OpenAIChat)


def test_building_the_fake_provider_does_not_need_a_key_or_a_model():
    assert isinstance(get_llm(Settings(_env_file=None, llm_provider="fake")), FakeLLM)


def test_building_the_openai_provider_makes_no_network_call(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("no request may be made just by building the provider")

    monkeypatch.setattr(httpx.Client, "send", refuse)
    get_llm(openai_settings())
