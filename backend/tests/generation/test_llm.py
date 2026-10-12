import pytest

from askact.config import Settings
from askact.generation.llm import (
    LLM,
    FakeLLM,
    LLMBadOutputError,
    LLMReply,
    LLMUnavailableError,
    get_llm,
)


def settings(provider: str | None) -> Settings:
    return Settings(_env_file=None, llm_provider=provider)


# --- LLMReply --------------------------------------------------------------------------

def test_a_reply_without_token_counts_has_none_not_zero():
    reply = LLMReply("hello")
    assert (reply.text, reply.input_tokens, reply.output_tokens) == ("hello", None, None)


def test_a_reply_can_carry_token_counts():
    reply = LLMReply("hello", input_tokens=120, output_tokens=7)
    assert (reply.input_tokens, reply.output_tokens) == (120, 7)


# --- FakeLLM ---------------------------------------------------------------------------

def test_replies_come_back_in_script_order():
    fake = FakeLLM(["first", "second", "third"])
    assert [fake.complete("s", "u", 10).text for _ in range(3)] == ["first", "second", "third"]


def test_a_string_in_the_script_becomes_a_reply_without_token_counts():
    reply = FakeLLM(["hello"]).complete("s", "u", 10)
    assert reply == LLMReply("hello", None, None)


def test_a_full_reply_in_the_script_is_returned_as_given():
    scripted = LLMReply("hello", input_tokens=50, output_tokens=5)
    assert FakeLLM([scripted]).complete("s", "u", 10) is scripted


def test_an_exception_in_the_script_is_raised_at_that_call():
    fake = FakeLLM(["fine", LLMUnavailableError("provider is down"), "recovered"])
    assert fake.complete("s", "u", 10).text == "fine"
    with pytest.raises(LLMUnavailableError, match="provider is down"):
        fake.complete("s", "u", 10)
    assert fake.complete("s", "u", 10).text == "recovered"


def test_calls_are_counted_including_the_ones_that_raise():
    fake = FakeLLM(["a", ValueError("boom")])
    assert fake.call_count == 0
    fake.complete("s", "u", 10)
    assert fake.call_count == 1
    with pytest.raises(ValueError):
        fake.complete("s", "u", 10)
    assert fake.call_count == 2


def test_each_request_is_recorded_so_tests_can_check_what_was_sent():
    fake = FakeLLM(["a", "b"])
    fake.complete("system one", "user one", 500)
    fake.complete("system two", "user two", 64)
    assert [(c.system, c.user, c.max_output_tokens) for c in fake.calls] == [
        ("system one", "user one", 500), ("system two", "user two", 64)]


def test_a_call_beyond_the_script_is_an_error_not_an_invented_reply():
    fake = FakeLLM(["only one"])
    fake.complete("s", "u", 10)
    with pytest.raises(RuntimeError, match="called 2 times but only 1 replies were scripted"):
        fake.complete("s", "u", 10)
    assert fake.call_count == 2  # the unexpected call is still visible to the test


def test_an_empty_script_fails_on_the_first_call():
    with pytest.raises(RuntimeError, match="called 1 times but only 0 replies"):
        FakeLLM().complete("s", "u", 10)


def test_a_single_string_is_refused_instead_of_being_split_into_characters():
    with pytest.raises(TypeError, match="list of replies"):
        FakeLLM("abc")


def test_changing_the_list_passed_in_does_not_change_the_script():
    replies = ["a", "b"]
    fake = FakeLLM(replies)
    replies.clear()
    assert fake.complete("s", "u", 10).text == "a"


def test_a_fake_can_be_used_wherever_an_llm_is_expected():
    def ask(llm: LLM) -> str:
        return llm.complete("s", "u", 10).text

    assert ask(FakeLLM(["hi"])) == "hi"


# --- error types -----------------------------------------------------------------------

def test_the_two_errors_carry_the_api_error_codes():
    assert LLMUnavailableError.code == "llm_unavailable"
    assert LLMBadOutputError.code == "llm_bad_output"


# --- get_llm: the provider is chosen by LLM_PROVIDER (R3.9, R3.10) ----------------------

def test_the_fake_provider_is_chosen_by_name():
    llm = get_llm(settings("fake"))
    assert isinstance(llm, FakeLLM) and llm.call_count == 0


@pytest.mark.parametrize("written", ["FAKE", "Fake", "  fake  "])
def test_the_provider_name_ignores_case_and_surrounding_spaces(written):
    assert isinstance(get_llm(settings(written)), FakeLLM)


def test_each_call_builds_a_fresh_provider_so_call_counts_are_not_shared():
    first, second = get_llm(settings("fake")), get_llm(settings("fake"))
    assert isinstance(first, FakeLLM) and isinstance(second, FakeLLM)
    assert first is not second
    with pytest.raises(RuntimeError):
        first.complete("s", "u", 10)
    assert (first.call_count, second.call_count) == (1, 0)


def test_the_env_provided_fake_has_no_script_and_says_so_when_called():
    with pytest.raises(RuntimeError, match="only 0 replies were scripted"):
        get_llm(settings("fake")).complete("s", "u", 10)


@pytest.mark.parametrize("provider", [None, "", "   "])
def test_no_provider_is_an_llm_unavailable_error_that_names_the_variable(provider):
    with pytest.raises(LLMUnavailableError) as info:
        get_llm(settings(provider))
    assert info.value.code == "llm_unavailable"
    assert "LLM_PROVIDER is not set" in str(info.value)
    assert "fake, openai" in str(info.value)  # and what it could be set to


@pytest.mark.parametrize("provider", ["anthropic", "gemini", "not-a-provider", "fake2", "fake;", "openai2", "open ai"])
def test_an_unknown_provider_is_an_llm_unavailable_error_listing_the_supported_ones(provider):
    with pytest.raises(LLMUnavailableError) as info:
        get_llm(settings(provider))
    message = str(info.value)
    assert info.value.code == "llm_unavailable"
    assert f"{provider!r} is not supported" in message
    assert "use one of: fake, openai" in message


def test_the_provider_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    assert isinstance(get_llm(Settings(_env_file=None)), FakeLLM)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(LLMUnavailableError):
        get_llm(Settings(_env_file=None))
