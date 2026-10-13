"""The evaluation's LLM cache and call budget (R5.9).

`CachingLLM` wraps a real `LLM` and sits where the app would put the provider, so the evaluation runs the same
`answer()` pipeline as the API. It does two things the API does not need:

  * a reply is remembered by what was asked, and a repeat is answered from disk without a call. A re-run, or a
    second config that retrieves the same passages, costs nothing, and the numbers it produces are the numbers
    of the first run: a model's reply varies from call to call, the cached one does not.
  * the number of *uncached* calls is capped (EVAL_MAX_LLM_CALLS). The call that would exceed it is not made.

The cache is a JSON-lines file, `eval/.cache/llm.jsonl` (gitignored). A reply is appended the moment it arrives,
so a run that is stopped, or aborted by the cap or by a provider outage, keeps everything it has already paid
for, and the next run continues from there. It stores replies and token counts only, never the prompt.

The key is `sha256(provider, model, system, user, max_output_tokens)`. `user` holds the question and every
passage, so a different question, different retrieved passages or a different model is a miss. What it does
not cover: `LLM_BASE_URL`, so the same model name behind a different server shares entries; delete the file if
you switch to a server that answers differently under the same name.
"""

import hashlib
import json
from pathlib import Path

from askact.generation.llm import LLM, LLMReply

CACHE_PATH = Path("eval/.cache/llm.jsonl")


class CacheError(Exception):
    """The cache file cannot be read as written by this module. Nothing is guessed: an entry that is skipped
    silently is a paid call repeated silently."""


class BudgetExceededError(Exception):
    """The next call would be one more uncached call than EVAL_MAX_LLM_CALLS allows."""

    def __init__(self, limit: int):
        super().__init__(f"the evaluation needs more than the {limit} uncached LLM calls that EVAL_MAX_LLM_CALLS allows")
        self.limit = limit


def cache_key(provider: str, model: str, system: str, user: str, max_output_tokens: int) -> str:
    # Hashing a JSON list, not a concatenation: ("a", "bc") and ("ab", "c") must not give the same key.
    payload = json.dumps([provider, model, system, user, max_output_tokens])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _token_count(value: object) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _load(path: Path) -> dict[str, LLMReply]:
    """Every cached reply. A missing file is an empty cache; a file that is not what this module wrote is a CacheError."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except UnicodeDecodeError:
        raise CacheError(f"{path} is not a UTF-8 text file; delete it to start a new cache") from None
    replies: dict[str, LLMReply] = {}
    for number, line in enumerate(text.split("\n"), start=1):  # not splitlines(): only "\n" ends an entry
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            raise CacheError(
                f"{path}, line {number}: not valid JSON (an interrupted write?). Delete that line, or the file"
            ) from None
        valid = (
            isinstance(entry, dict)
            and set(entry) == {"key", "text", "input_tokens", "output_tokens"}
            and isinstance(entry["key"], str)
            and isinstance(entry["text"], str)
            and _token_count(entry["input_tokens"])
            and _token_count(entry["output_tokens"])
        )
        if not valid:
            raise CacheError(f"{path}, line {number}: not a cache entry. Delete that line, or the file")
        replies[entry["key"]] = LLMReply(entry["text"], entry["input_tokens"], entry["output_tokens"])
    return replies


class CachingLLM:
    """An `LLM` that answers repeats from the cache and refuses to make more than `max_calls` new calls.

    `provider` and `model` are part of the key, so they are given here (they are settings, which the wrapped
    `LLM` does not expose). Not thread-safe: the evaluation makes its calls one after another.
    """

    def __init__(self, inner: LLM, path: Path, *, provider: str, model: str, max_calls: int):
        if max_calls < 0:
            raise ValueError(f"max_calls cannot be negative, got {max_calls}")
        self._inner = inner
        self._path = path
        self._provider = provider
        self._model = model
        self._max_calls = max_calls
        self._replies = _load(path)
        self.uncached_calls = 0  # calls attempted on the wrapped LLM in this run, whether or not they succeeded
        self.cache_hits = 0

    def complete(self, system: str, user: str, max_output_tokens: int) -> LLMReply:
        key = cache_key(self._provider, self._model, system, user, max_output_tokens)
        cached = self._replies.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        if self.uncached_calls >= self._max_calls:
            raise BudgetExceededError(self._max_calls)
        # Counted before the call, and kept if it fails: a call that timed out may still have been billed,
        # so a failure is not a free retry. Failures are not cached.
        self.uncached_calls += 1
        reply = self._inner.complete(system, user, max_output_tokens)
        self._append(key, reply)
        self._replies[key] = reply
        return reply

    def _append(self, key: str, reply: LLMReply) -> None:
        entry = {"key": key, "text": reply.text, "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry) + "\n")  # one write, so a stop cannot leave half an entry plus half another
