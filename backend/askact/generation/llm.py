"""The seam between the app and a language-model provider (R3.9, R3.10, R6.12).

Everything else talks to the `LLM` interface: one method, `complete`. A provider is a class with that
method plus one line in `_PROVIDERS`, so choosing it is an environment variable (`LLM_PROVIDER`), not a
code change. `FakeLLM` is the provider for tests and CI: it needs no network and no key.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from askact.config import Settings


class LLMUnavailableError(Exception):
    """The configured provider cannot be used: not chosen, not supported, or (for real ones) no key or
    model, or the provider refused the call. The API reports it as `llm_unavailable` (R3.10).

    Messages are written for the operator and never contain a key or a request/response body.
    """

    code = "llm_unavailable"


class LLMBadOutputError(Exception):
    """The provider answered, but not with the required structure. The API reports `llm_bad_output` (R3.3).

    The message says what was wrong, never what the reply said: model output is untrusted text and is
    never shown as an answer or written to logs.
    """

    code = "llm_bad_output"


@dataclass(frozen=True)
class LLMReply:
    text: str
    # What the provider says it used, for the spend ledger. None if it did not say: the ledger then
    # keeps its worst-case reservation as the charge (R6.9).
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLM(Protocol):
    def complete(self, system: str, user: str, max_output_tokens: int) -> LLMReply:
        """One chat completion. `max_output_tokens` is a hard cap on the reply (R6.12).

        Raises LLMUnavailableError if the provider cannot be reached or refuses the call.
        """
        ...


@dataclass(frozen=True)
class Call:
    """One request made to a `FakeLLM`, kept so that tests can check what was sent."""

    system: str
    user: str
    max_output_tokens: int


class FakeLLM:
    """Replies from a script, in order, and counts its calls. For tests and CI: no network, no key.

    Each script item is the reply text, a full `LLMReply` (to give token counts), or an exception to
    raise (to simulate a failing provider). A call beyond the end of the script is an error: the test
    made more calls than it expected, and returning something made up would hide that.
    """

    def __init__(self, replies: Sequence[str | LLMReply | Exception] = ()):
        if isinstance(replies, str):
            # FakeLLM("text") would silently script one reply per character.
            raise TypeError("FakeLLM takes a list of replies, not a single string")
        self._script = list(replies)
        self.calls: list[Call] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def complete(self, system: str, user: str, max_output_tokens: int) -> LLMReply:
        self.calls.append(Call(system, user, max_output_tokens))
        if len(self.calls) > len(self._script):
            raise RuntimeError(
                f"FakeLLM was called {len(self.calls)} times but only {len(self._script)} replies were scripted"
            )
        item = self._script[len(self.calls) - 1]
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, LLMReply) else LLMReply(item)


# Provider name (as written in LLM_PROVIDER) -> how to build it. Adding a provider is one line here.
# A fake with an empty script is what LLM_PROVIDER=fake gives: it is there to be replaced by a scripted
# one in tests, and complains if it is called without a script.
_PROVIDERS: dict[str, Callable[[Settings], LLM]] = {
    "fake": lambda settings: FakeLLM(),
}


def get_llm(settings: Settings) -> LLM:
    """The provider chosen by LLM_PROVIDER. Raises LLMUnavailableError, saying what to set, if there is none."""
    supported = ", ".join(sorted(_PROVIDERS))
    name = (settings.llm_provider or "").strip().lower()
    if not name:
        raise LLMUnavailableError(f"LLM_PROVIDER is not set; set it to one of: {supported}")
    factory = _PROVIDERS.get(name)
    if factory is None:
        raise LLMUnavailableError(f"LLM_PROVIDER {settings.llm_provider!r} is not supported; use one of: {supported}")
    return factory(settings)
