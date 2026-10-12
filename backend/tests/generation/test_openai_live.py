"""One real call to the configured provider. Manual and local only: never run in CI, never needs a secret there.

It is marked `slow` (so `pytest` skips it by default) and it skips itself unless both variables are set in the
environment, so it can never fail for want of a key. To run it, from `backend/`:

    LLM_API_KEY=... LLM_MODEL=<a model your account can use> pytest -m slow tests/generation/test_openai_live.py -v

`LLM_BASE_URL` is read too, to point it at an OpenAI-compatible server. The key is read from the environment only
(not from a `.env` file) so that nothing is picked up by accident, and it is never printed: the adapter's errors do
not contain it. The call goes straight to the adapter, not through the spend ledger; it costs a handful of tokens.

What it proves that the mocked tests cannot: that the provider accepts this request shape for this model
(`max_completion_tokens`, a `system` message), that a reply comes back with usage reported, and that the output cap
is actually honoured, which the spend reservation relies on (R6.12).
"""

import os

import pytest

from askact.config import Settings
from askact.generation.llm import get_llm

pytestmark = pytest.mark.slow

# Not tiny on purpose: a reasoning model spends part of this budget on hidden reasoning tokens before it writes any
# visible text, and a very small cap would leave an empty reply for a reason that has nothing to do with the adapter.
CAP = 256


@pytest.mark.skipif(
    not (os.environ.get("LLM_API_KEY") and os.environ.get("LLM_MODEL")),
    reason="set LLM_API_KEY and LLM_MODEL to run the live smoke test",
)
def test_one_real_call_returns_text_and_reports_usage_within_the_cap():
    llm = get_llm(Settings(_env_file=None, llm_provider="openai"))

    reply = llm.complete("You answer with a single short word.", "Say hello.", CAP)

    assert reply.text.strip(), (
        f"the provider returned no text (output tokens: {reply.output_tokens}); if that is close to the cap of {CAP}, "
        "the model may have spent its whole budget on reasoning: try a non-reasoning model"
    )
    assert reply.input_tokens is not None and reply.input_tokens > 0, "usage.prompt_tokens was not reported"
    assert reply.output_tokens is not None and reply.output_tokens > 0, "usage.completion_tokens was not reported"
    assert reply.output_tokens <= CAP, (
        f"the provider reported {reply.output_tokens} output tokens for a cap of {CAP}: "
        "it does not honour max_completion_tokens, so the spend reservation would understate the cost"
    )
