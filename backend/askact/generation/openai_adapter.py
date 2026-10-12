"""The OpenAI provider: Chat Completions over httpx (R3.9, R3.10, R6.12).

API facts below were read from OpenAI's documentation on 2026-10-11, not written from memory:
  * OpenAPI spec (operation createChatCompletion, request/response schemas, bearer auth, error schema):
      https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
  * Error codes (what each HTTP status and `error.code` means):
      https://developers.openai.com/api/docs/guides/error-codes/
  * Response example: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
What they say, and what this module does about it:
  * POST {base}/chat/completions with `Authorization: Bearer <key>`; errors are `{"error": {message, type, param, code}}`.
  * The output cap is `max_completion_tokens`, which counts visible AND reasoning tokens. `max_tokens` is deprecated and
    fails on o-series models, so it is never sent.
  * The reply text is `choices[0].message.content` (null for a refusal); usage is `usage.prompt_tokens` and
    `usage.completion_tokens` (the latter includes reasoning tokens, i.e. what is billed).
  * A 429 can mean a temporary rate limit OR an exhausted balance or spend limit, where retrying cannot help: the
    error code tells them apart (see _OUT_OF_CREDIT).
  * OpenAI now recommends its Responses API for new projects. Chat Completions is still supported and is what the
    OpenAI-compatible servers behind LLM_BASE_URL implement, which is why this adapter uses it.
  * The system prompt is sent with role `system`. The spec says reasoning models take `developer` instead; the only
    statement I found that `system` is converted for them is from OpenAI staff on the community forum, not the docs.
    `system` is also what compatible servers expect. The live smoke test is what confirms it on a given model.

Deliberate choices:
  * No retries. A timed-out call may still have been billed, and a retry would spend again behind the spend guard's
    back; the caller sees a clean `llm_unavailable` instead.
  * No `temperature`, `response_format` or other optional parameters: reasoning models reject some of them, and
    compatible servers support them unevenly. The reply format is enforced by the prompt and the parser instead.
  * Error messages carry the HTTP status and, if it is a short identifier, OpenAI's `error.code`: never the provider's
    free-text message (it can echo request content), never the key, and never the underlying exception's text.
"""

import re
from typing import Any

import httpx

from askact.config import Settings
from askact.generation.llm import LLMReply, LLMUnavailableError

DEFAULT_BASE_URL = "https://api.openai.com/v1"
# Waiting for the whole reply (the request is not streamed), so the read timeout is the real limit on a call.
TIMEOUT = httpx.Timeout(90.0, connect=10.0)

# error.code / error.type values for a 429 that is about money, not speed. From the error-codes guide.
_OUT_OF_CREDIT = {
    "credit_balance_exhausted",
    "insufficient_quota",
    "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
}
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def _count(usage: object, field: str) -> int | None:
    """A token count, or None if the provider did not report one (or reported something that is not a count)."""
    value = usage.get(field) if isinstance(usage, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _error_fields(response: httpx.Response, secret: str) -> tuple[str | None, str | None]:
    """(error.code, error.type) from an error body, each only if it is a short plain identifier.

    "Looks like an identifier" does not make a value safe to print: an API key is made of the same characters,
    and this adapter also talks to arbitrary OpenAI-compatible servers. A value that contains the key, or is a
    fragment of it, is therefore dropped as well.
    """
    try:
        error = response.json().get("error")
    except (ValueError, AttributeError):
        return None, None
    if not isinstance(error, dict):
        return None, None

    def safe(name: str) -> str | None:
        value = error.get(name)
        if not (isinstance(value, str) and _SAFE_IDENTIFIER.fullmatch(value)):
            return None
        if secret in value or (len(value) >= 8 and value in secret):
            return None
        return value

    return safe("code"), safe("type")


def _http_error(response: httpx.Response, secret: str) -> LLMUnavailableError:
    status = response.status_code
    code, kind = _error_fields(response, secret)
    detail = f" ({code})" if code else ""
    if status == 401:
        return LLMUnavailableError(f"the provider rejected the API key (HTTP 401{detail}): check LLM_API_KEY")
    if status == 403:
        return LLMUnavailableError(
            f"the provider refused access (HTTP 403{detail}): the key may lack permission, or this region or IP is not allowed"
        )
    if status == 404:
        return LLMUnavailableError(f"the provider does not know this model or endpoint (HTTP 404{detail}): check LLM_MODEL and LLM_BASE_URL")
    if status == 429:
        if code in _OUT_OF_CREDIT or kind in _OUT_OF_CREDIT:
            return LLMUnavailableError(
                f"the provider account is out of credit or over a spending limit (HTTP 429{detail}): retrying will not help, fix the account first"
            )
        return LLMUnavailableError(f"the provider is rate limiting requests (HTTP 429{detail}): try again shortly")
    if status == 400:
        return LLMUnavailableError(f"the provider rejected the request (HTTP 400{detail}): check LLM_MODEL, and LLM_BASE_URL if it is set")
    if status >= 500:
        return LLMUnavailableError(f"the provider had a problem (HTTP {status}{detail}): try again shortly")
    return LLMUnavailableError(f"unexpected response from the provider (HTTP {status}): check LLM_BASE_URL")


class OpenAIChat:
    """Chat Completions against api.openai.com, or any server that implements the same endpoint.

    A compatible server must honour `max_completion_tokens`: if it ignores the parameter the output cap (R6.12)
    silently does not apply, and the spend reservation assumes it does.
    """

    def __init__(self, *, api_key: str, model: str, base_url: str = DEFAULT_BASE_URL, client: httpx.Client | None = None):
        self._api_key = api_key
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._client = client or httpx.Client(timeout=TIMEOUT)  # follow_redirects stays off: the key is never forwarded

    def __repr__(self) -> str:
        return f"OpenAIChat(model={self.model!r})"  # no key, and no base URL (it could carry credentials)

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAIChat":
        """Build the provider from LLM_API_KEY, LLM_MODEL and optionally LLM_BASE_URL.

        Raises LLMUnavailableError naming every variable that is missing or unusable, and never the key itself.
        """
        key = settings.llm_api_key.get_secret_value().strip() if settings.llm_api_key else ""
        model = (settings.llm_model or "").strip()
        problems = []
        if not key:
            problems.append("LLM_API_KEY is not set")
        elif re.search(r"[\s\x00-\x1f\x7f]", key):
            problems.append("LLM_API_KEY contains spaces or control characters (check it was copied whole)")
        if not model:
            problems.append("LLM_MODEL is not set")
        base_url = (settings.llm_base_url or "").strip() or DEFAULT_BASE_URL
        if not base_url.startswith(("http://", "https://")):
            problems.append("LLM_BASE_URL must start with http:// or https://")
        if problems:
            raise LLMUnavailableError("; ".join(problems))
        return cls(api_key=key, model=model, base_url=base_url)

    def complete(self, system: str, user: str, max_output_tokens: int) -> LLMReply:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": max_output_tokens,
        }
        try:
            response = self._client.post(self._url, json=body, headers={"Authorization": f"Bearer {self._api_key}"})
        except httpx.TimeoutException:
            raise LLMUnavailableError("the provider did not answer in time") from None
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            # Only the type: the message of a transport error can contain a URL or, for a malformed header, the key.
            raise LLMUnavailableError(f"could not reach the provider ({type(exc).__name__}): check LLM_BASE_URL and the network") from None
        if response.status_code != 200:
            raise _http_error(response, self._api_key)
        return self._read_reply(response)

    @staticmethod
    def _read_reply(response: httpx.Response) -> LLMReply:
        not_a_completion = LLMUnavailableError("the provider's reply is not a chat completion: check LLM_BASE_URL")
        try:
            body: Any = response.json()
        except ValueError:
            raise not_a_completion from None
        choices = body.get("choices") if isinstance(body, dict) else None
        first = choices[0] if isinstance(choices, list) and choices else None
        message = first.get("message") if isinstance(first, dict) else None
        if not isinstance(message, dict):
            raise not_a_completion
        content = message.get("content")
        # `content` is null for a refusal, and can be empty when a reasoning model spends the whole token budget
        # thinking. The reply is then just empty text: the parser turns it into llm_bad_output, and the usage below
        # is still reported, because those tokens were billed.
        text = content if isinstance(content, str) else ""
        usage = body.get("usage")
        return LLMReply(text, input_tokens=_count(usage, "prompt_tokens"), output_tokens=_count(usage, "completion_tokens"))
