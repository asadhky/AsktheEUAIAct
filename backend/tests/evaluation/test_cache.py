import json
import re

import pytest

from askact.evaluation.cache import BudgetExceededError, CacheError, CachingLLM, cache_key
from askact.generation.llm import FakeLLM, LLMReply, LLMUnavailableError


def make(tmp_path, replies=(), *, max_calls=10, provider="openai", model="gpt-x", name="llm.jsonl"):
    fake = FakeLLM(list(replies))
    return CachingLLM(fake, tmp_path / name, provider=provider, model=model, max_calls=max_calls), fake


def entries(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- the key ----------------------------------------------------------------------------------------------

def test_the_key_is_stable_across_versions():
    # Pinned: a cache written by an earlier version must keep working, and paid replies must not be orphaned by
    # an accidental change to how the key is built. (Checked against an independent sha256 of the JSON list.)
    assert cache_key("openai", "gpt-x", "SYS", "USER", 500) == "0365c34f8ae65d7e5ac58754fa83895361e4e4e7377f6e64518003f283431972"


def test_every_part_of_the_request_changes_the_key():
    base = ("openai", "gpt-x", "SYS", "USER", 500)
    variants = [("other", *base[1:]), (base[0], "gpt-y", *base[2:]), (*base[:2], "SYS2", *base[3:]),
                (*base[:3], "USER2", base[4]), (*base[:4], 501)]
    keys = {cache_key(*base)} | {cache_key(*v) for v in variants}
    assert len(keys) == 6


def test_fields_cannot_run_together_into_the_same_key():
    assert cache_key("a", "bc", "s", "u", 1) != cache_key("ab", "c", "s", "u", 1)
    assert cache_key("p", "m", "ab", "c", 1) != cache_key("p", "m", "a", "bc", 1)


def test_the_key_is_a_sha256_hex_digest():
    assert re.fullmatch(r"[0-9a-f]{64}", cache_key("p", "m", "s", "u", 1))


# --- hits and misses --------------------------------------------------------------------------------------

def test_a_repeat_is_answered_from_the_cache_without_a_call(tmp_path):
    llm, fake = make(tmp_path, [LLMReply("first", 11, 3)])
    assert llm.complete("sys", "user", 100) == LLMReply("first", 11, 3)
    assert llm.complete("sys", "user", 100) == LLMReply("first", 11, 3)
    assert (fake.call_count, llm.uncached_calls, llm.cache_hits) == (1, 1, 1)


def test_unreported_token_counts_survive_the_cache(tmp_path):
    llm, _ = make(tmp_path, [LLMReply("text", None, None)])
    llm.complete("s", "u", 1)
    assert llm.complete("s", "u", 1) == LLMReply("text", None, None)


def test_an_empty_reply_is_cached_too(tmp_path):
    # A refusal comes back as empty text, and it was paid for: asking again would pay again for the same thing.
    llm, fake = make(tmp_path, [LLMReply("", 5, 0)])
    llm.complete("s", "u", 1)
    assert llm.complete("s", "u", 1) == LLMReply("", 5, 0)
    assert fake.call_count == 1


def test_a_new_run_reads_what_an_earlier_one_stored(tmp_path):
    first, _ = make(tmp_path, ["answer one", "answer two"])
    first.complete("s", "u1", 1)
    first.complete("s", "u2", 1)
    second, fake = make(tmp_path, [])  # no scripted replies: any real call would fail
    assert second.complete("s", "u2", 1).text == "answer two"
    assert second.complete("s", "u1", 1).text == "answer one"
    assert (fake.call_count, second.uncached_calls, second.cache_hits) == (0, 0, 2)


def test_a_reply_with_zero_tokens_is_stored_and_read_back(tmp_path):
    # An empty reply, or a reasoning model that spent its whole budget thinking, can report 0 for a count.
    first, _ = make(tmp_path, [LLMReply("", 0, 0)])
    first.complete("s", "u", 1)
    again, fake = make(tmp_path, [])
    assert again.complete("s", "u", 1) == LLMReply("", 0, 0) and fake.call_count == 0


@pytest.mark.parametrize("second", [("other system", "u", 1), ("s", "other user", 1), ("s", "u", 2)])
def test_a_changed_request_misses_the_cache(tmp_path, second):
    llm, fake = make(tmp_path, ["a", "b"])
    llm.complete("s", "u", 1)
    assert llm.complete(*second).text == "b"
    assert fake.call_count == 2


@pytest.mark.parametrize("other", [dict(model="gpt-y"), dict(provider="other")])
def test_a_different_model_or_provider_misses_the_cache(tmp_path, other):
    first, _ = make(tmp_path, ["from gpt-x"])
    first.complete("s", "u", 1)
    second, fake = make(tmp_path, ["from the other"], **other)
    assert second.complete("s", "u", 1).text == "from the other"
    assert fake.call_count == 1


def test_the_same_request_twice_in_one_run_is_one_call(tmp_path):
    # This is what happens when two retrieval configs return the same passages for a question.
    llm, fake = make(tmp_path, ["only"])
    for _ in range(3):
        llm.complete("s", "u", 1)
    assert fake.call_count == 1


# --- the call cap -----------------------------------------------------------------------------------------

def test_the_call_that_would_exceed_the_cap_is_not_made(tmp_path):
    llm, fake = make(tmp_path, ["a", "b", "c"], max_calls=2)
    llm.complete("s", "u1", 1)
    llm.complete("s", "u2", 1)
    with pytest.raises(BudgetExceededError) as raised:
        llm.complete("s", "u3", 1)
    assert fake.call_count == 2  # the third was never sent
    assert raised.value.limit == 2 and "2" in str(raised.value) and "EVAL_MAX_LLM_CALLS" in str(raised.value)


def test_cached_replies_are_still_served_once_the_cap_is_reached(tmp_path):
    llm, fake = make(tmp_path, ["a"], max_calls=1)
    llm.complete("s", "u1", 1)
    assert llm.complete("s", "u1", 1).text == "a"
    assert fake.call_count == 1


def test_a_cap_of_zero_allows_only_cached_replies(tmp_path):
    first, _ = make(tmp_path, ["a"])
    first.complete("s", "u1", 1)
    llm, fake = make(tmp_path, [], max_calls=0)
    assert llm.complete("s", "u1", 1).text == "a"
    with pytest.raises(BudgetExceededError):
        llm.complete("s", "new question", 1)
    assert fake.call_count == 0


def test_a_cap_of_zero_with_no_cache_makes_no_call_and_creates_no_file(tmp_path):
    llm, fake = make(tmp_path, [], max_calls=0)
    with pytest.raises(BudgetExceededError):
        llm.complete("s", "u", 1)
    assert fake.call_count == 0 and not (tmp_path / "llm.jsonl").exists()


def test_the_cap_counts_only_uncached_calls_across_runs(tmp_path):
    first, _ = make(tmp_path, ["a", "b"], max_calls=2)
    first.complete("s", "u1", 1)
    first.complete("s", "u2", 1)
    # Same cap again, but the earlier replies are free, so two new calls fit.
    second, fake = make(tmp_path, ["c", "d"], max_calls=2)
    for user in ("u1", "u2", "u3", "u4"):
        second.complete("s", user, 1)
    assert (fake.call_count, second.uncached_calls, second.cache_hits) == (2, 2, 2)


def test_a_failed_call_still_counts_against_the_cap_and_is_not_cached(tmp_path):
    # A call that timed out may have been billed, so a failure is not a free retry.
    llm, fake = make(tmp_path, [LLMUnavailableError("down"), LLMUnavailableError("down"), "late"], max_calls=2)
    for _ in range(2):
        with pytest.raises(LLMUnavailableError):
            llm.complete("s", "u", 1)
    assert llm.uncached_calls == 2 and not (tmp_path / "llm.jsonl").exists()
    with pytest.raises(BudgetExceededError):
        llm.complete("s", "u", 1)
    assert fake.call_count == 2


def test_a_negative_cap_is_refused(tmp_path):
    with pytest.raises(ValueError, match="negative"):
        make(tmp_path, max_calls=-1)


# --- the file ---------------------------------------------------------------------------------------------

def test_each_reply_is_written_as_soon_as_it_arrives(tmp_path):
    llm, _ = make(tmp_path, ["a", RuntimeError("the provider fell over")])
    llm.complete("s", "u1", 1)
    with pytest.raises(RuntimeError):
        llm.complete("s", "u2", 1)
    # The run died, and the first reply is already on disk.
    assert [e["text"] for e in entries(tmp_path / "llm.jsonl")] == ["a"]


def test_the_file_holds_replies_and_token_counts_and_never_the_prompt(tmp_path):
    llm, _ = make(tmp_path, [LLMReply("the reply", 7, 2)])
    llm.complete("SECRET-SYSTEM-TEXT", "SECRET-QUESTION-AND-PASSAGES", 1)
    raw = (tmp_path / "llm.jsonl").read_text(encoding="utf-8")
    [entry] = entries(tmp_path / "llm.jsonl")
    assert entry == {"key": cache_key("openai", "gpt-x", "SECRET-SYSTEM-TEXT", "SECRET-QUESTION-AND-PASSAGES", 1),
                     "text": "the reply", "input_tokens": 7, "output_tokens": 2}
    assert "SECRET" not in raw


def test_awkward_text_round_trips_and_stays_one_line_per_entry(tmp_path):
    awkward = 'line one\nline two\r\n{"covered": true}\u2028\x0b\x85 \U0001f600 caf\u00e9 "quoted" \\ back'
    llm, _ = make(tmp_path, [awkward, "plain"])
    llm.complete("s", "u1", 1)
    llm.complete("s", "u2", 1)
    raw = (tmp_path / "llm.jsonl").read_text(encoding="utf-8")
    assert raw.count("\n") == 2 and raw.isascii()  # an entry is exactly one line, whatever the reply contains
    again, fake = make(tmp_path, [])
    assert again.complete("s", "u1", 1).text == awkward and fake.call_count == 0


def test_the_cache_directory_is_created_on_first_write(tmp_path):
    fake = FakeLLM(["a"])
    path = tmp_path / "eval" / ".cache" / "llm.jsonl"
    CachingLLM(fake, path, provider="p", model="m", max_calls=1).complete("s", "u", 1)
    assert path.exists()


def test_a_cache_path_that_cannot_be_used_fails_before_any_paid_call(tmp_path):
    (tmp_path / "blocker").write_text("a file where a directory should be", encoding="utf-8")
    fake = FakeLLM(["a"])
    with pytest.raises(OSError):
        CachingLLM(fake, tmp_path / "blocker" / "llm.jsonl", provider="p", model="m", max_calls=1)
    assert fake.call_count == 0


def test_a_cache_that_stops_being_writable_mid_run_stops_the_run(tmp_path):
    # Loading finds no file and is fine; then the directory's place is taken, as a full or broken disk would.
    fake = FakeLLM(["a"])
    llm = CachingLLM(fake, tmp_path / "later" / "llm.jsonl", provider="p", model="m", max_calls=1)
    (tmp_path / "later").write_text("now a file", encoding="utf-8")
    with pytest.raises(OSError):
        llm.complete("s", "u", 1)


def test_a_cache_path_that_is_a_directory_is_an_error_not_an_empty_cache(tmp_path):
    (tmp_path / "llm.jsonl").mkdir()
    with pytest.raises(OSError):
        CachingLLM(FakeLLM([]), tmp_path / "llm.jsonl", provider="p", model="m", max_calls=1)


def test_blank_lines_in_the_file_are_ignored(tmp_path):
    first, _ = make(tmp_path, ["a"])
    first.complete("s", "u", 1)
    path = tmp_path / "llm.jsonl"
    path.write_text("\n" + path.read_text(encoding="utf-8") + "\n   \n", encoding="utf-8")
    again, fake = make(tmp_path, [])
    assert again.complete("s", "u", 1).text == "a"


def test_the_later_entry_wins_if_a_key_is_stored_twice(tmp_path):
    # Two runs at once can both miss and both append. Either reply is a real one; the later is used.
    key = cache_key("openai", "gpt-x", "s", "u", 1)
    lines = [json.dumps({"key": key, "text": t, "input_tokens": None, "output_tokens": None}) for t in ("old", "new")]
    (tmp_path / "llm.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    llm, _ = make(tmp_path, [])
    assert llm.complete("s", "u", 1).text == "new"


# --- a file this module did not write: refused, with the line number, before any call ---------------------

GOOD = {"key": "k", "text": "t", "input_tokens": 1, "output_tokens": 2}


@pytest.mark.parametrize("bad, reason", [
    ('{"key": "k", "text": "cut off', "not valid JSON"),
    ("not json at all", "not valid JSON"),
    ("[1, 2]", "not a cache entry"),
    ('"just a string"', "not a cache entry"),
    (json.dumps({k: v for k, v in GOOD.items() if k != "text"}), "not a cache entry"),
    (json.dumps({**GOOD, "extra": 1}), "not a cache entry"),
    (json.dumps({**GOOD, "text": 5}), "not a cache entry"),
    (json.dumps({**GOOD, "key": None}), "not a cache entry"),
    (json.dumps({**GOOD, "input_tokens": -1}), "not a cache entry"),
    (json.dumps({**GOOD, "output_tokens": 1.5}), "not a cache entry"),
    (json.dumps({**GOOD, "output_tokens": True}), "not a cache entry"),
    (json.dumps({**GOOD, "input_tokens": "7"}), "not a cache entry"),
])
def test_a_bad_cache_file_is_refused_naming_the_line(tmp_path, bad, reason):
    path = tmp_path / "llm.jsonl"
    path.write_text(json.dumps(GOOD) + "\n" + bad + "\n", encoding="utf-8")
    fake = FakeLLM(["never used"])
    with pytest.raises(CacheError, match=r"line 2") as raised:
        CachingLLM(fake, path, provider="p", model="m", max_calls=5)
    assert reason in str(raised.value) and str(path) in str(raised.value)
    assert fake.call_count == 0


def test_a_cache_file_that_is_not_text_is_refused(tmp_path):
    path = tmp_path / "llm.jsonl"
    path.write_bytes(b"\xff\xfe\x00 not utf-8")
    with pytest.raises(CacheError, match="UTF-8"):
        CachingLLM(FakeLLM([]), path, provider="p", model="m", max_calls=5)


def test_a_cache_error_does_not_quote_the_bad_line(tmp_path):
    path = tmp_path / "llm.jsonl"
    path.write_text('{"key": "SENSITIVE-FRAGMENT', encoding="utf-8")
    with pytest.raises(CacheError) as raised:
        CachingLLM(FakeLLM([]), path, provider="p", model="m", max_calls=5)
    assert "SENSITIVE-FRAGMENT" not in str(raised.value) and raised.value.__cause__ is None
