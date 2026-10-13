"""The HTTP API: `POST /ask` and `GET /health` (R2.8, R3.10, R3.11, R6.5 to R6.11).

The checks on `/ask` run in this order, cheapest and free first, so nothing that costs money happens until
everything that does not has passed:

  1. body size       a streamed cap, before the body is parsed        413 `request_too_large`
  2. rate limit      per client, before FastAPI touches the body      429 `rate_limited` (+ Retry-After)
  3. body shape      JSON with a `question` string                    422 `invalid_request`
  4. question length at most MAX_QUESTION_CHARS                       422 `question_too_long`
  5. retrieve        the active config (hybrid, or hybrid+rerank)     503 `index_unavailable`
  6. relevance gate  below the threshold: not covered, no LLM call
  7. spend reserve   refused if the daily cap would be exceeded       503 `spend_cap_reached`
  8. LLM call, parse and validate citations, reconcile spend          503 `llm_unavailable`, 502 `llm_bad_output`

Steps 1 to 3 are in this order, not the written design's "length, then rate limit", because the length check
needs the parsed body: rate limiting after parsing would let anyone send unlimited malformed requests for free.
The rate limit is middleware and not a FastAPI dependency for a reason found by testing: FastAPI parses the body
while resolving parameters, before it runs dependencies, so a body that is not valid JSON, or not valid UTF-8,
never reaches a dependency at all and would have been unlimited.

Every outcome, including every error, writes exactly one log line (`request_log`), which never contains the
question, the answer, the passages or the client address unless LOG_QUESTIONS is switched on for the question.
Error responses never echo the request or the text of an exception.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from askact.config import Settings
from askact.generation.answer import AnswerResult, SpendGuard, answer
from askact.generation.llm import LLM, get_llm
from askact.guards.ratelimit import Decision, RateLimiter, client_key
from askact.guards.spend import SpendLedger
from askact.index import IndexUnavailableError
from askact.models import RetrievalResult
from askact.request_log import describe_error, log_request
from askact.retrieval import Config, Retriever, app_config

log = logging.getLogger(__name__)

# What each error looks like on the wire: code -> (HTTP status, a fixed message). The message is fixed text and never
# built from the request or from an exception, so an error response cannot echo anything the client or a provider sent.
ERRORS: dict[str, tuple[int, str]] = {
    "request_too_large": (413, "The request body is too large."),
    "rate_limited": (429, "Too many requests. Please wait a moment and try again."),
    "invalid_request": (422, 'The request must be a JSON object with a text field named "question".'),
    "question_too_long": (422, "The question is too long."),
    "index_unavailable": (503, "The search index is not available right now."),
    "not_configured": (503, "Answering is not configured yet."),
    "spend_cap_reached": (503, "The daily usage limit for this demo has been reached. Please try again tomorrow."),
    "llm_unavailable": (503, "The answering service is not available right now."),
    "llm_bad_output": (502, "The answering service returned an unusable reply. Please try again."),
    "internal_error": (500, "Something went wrong."),
}

# Bytes of JSON per character of question. UTF-8 is at most 4 bytes per character and a JSON escape (\u00e9, or a
# surrogate pair for an emoji) can be 12 bytes, so 12 is a safe ceiling; the margin covers the braces and the key.
BYTES_PER_QUESTION_CHAR = 12
BODY_MARGIN_BYTES = 512


def error_response(code: str, headers: dict[str, str] | None = None) -> JSONResponse:
    status, message = ERRORS[code]
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


class ApiError(Exception):
    """Raised inside a request to answer with one of the coded errors in `ERRORS`. (The rate limit does not use it:
    it is refused in the `Admit` middleware, which needs to set `Retry-After` and so builds its own response.)"""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # an unknown field is an error, not silently ignored

    question: str


# --- limiting the size of the body, before anything parses it -----------------------------------------------------

class LimitRequestBody:
    """Refuse a POST whose body is over `limit` bytes, before any of it is parsed.

    FastAPI reads and parses a whole body before the handler runs: a 5 MB body is accepted and held in memory, and
    the question-length check would only run afterwards. This checks the declared Content-Length and, for a body
    sent in chunks (which has none), the bytes actually received, so at most `limit` bytes are ever buffered.
    Raising from inside `receive` does not work: Starlette turns it into a generic 400.
    """

    def __init__(self, app: ASGIApp, limit: int, on_refuse: Callable[[Scope], None]):
        self.app = app
        self.limit = limit
        self.on_refuse = on_refuse

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.limit:
            await self._refuse(scope, receive, send)
            return
        messages: list[Message] = []
        received, more = 0, True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            received += len(message.get("body", b""))
            if received > self.limit:
                await self._refuse(scope, receive, send)
                return
            messages.append(message)
            more = message.get("more_body", False)
        replay = iter(messages)

        async def replayed() -> Message:
            try:
                return next(replay)
            except StopIteration:
                return await receive()

        await self.app(scope, replayed, send)

    async def _refuse(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.on_refuse(scope)
        await error_response("request_too_large")(scope, receive, send)


class Admit:
    """Rate limit `POST /ask` before FastAPI reads or parses the body, and start the request's clock.

    A refusal is logged here, because the endpoint never runs. Other paths and methods (including a GET on /ask,
    which is a 405) pass through untouched and do not use the client's quota.
    """

    def __init__(self, app: ASGIApp, check: Callable[[Scope], int | None]):
        self.app = app
        self.check = check

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == "/ask":
            retry_after = self.check(scope)
            if retry_after is not None:
                await error_response("rate_limited", headers={"Retry-After": str(retry_after)})(scope, receive, send)
                return
        await self.app(scope, receive, send)


# --- what the app is built from ---------------------------------------------------------------------------------------

class IndexState:
    """The retriever, or the reason there isn't one. A missing or stale index must not crash the process: the
    container stays up, `/health` is 503, and `/ask` answers `index_unavailable`. The reason is kept for the
    operator (see `build_services`), never sent to a client."""

    def __init__(self, search: Callable[[str, Config, int], RetrievalResult] | None, error: IndexUnavailableError | None = None):
        self.search = search
        self.error = error

    @property
    def ready(self) -> bool:
        return self.search is not None


class Limiter(Protocol):
    """What the app needs from a rate limiter: one method (`RateLimiter` has it)."""

    def check(self, key: str) -> Decision: ...


@dataclass
class Services:
    """Everything the app needs that is not a setting. Tests pass stubs; production builds the real ones."""

    index: IndexState
    llm: LLM
    spend: SpendGuard
    limiter: Limiter
    clock: Callable[[], float] = time.monotonic


def build_services(settings: Settings) -> Services:
    """The real services, built from settings. Loads the index and the models, so it is slow: call it once at startup.

    A missing or stale index does not stop the process (see `IndexState`). A bad LLM configuration does not either:
    `get_llm` raising would be an unrecoverable start, so it is also deferred to the first request (where it is the
    `llm_unavailable` error). Nothing here is paid for.
    """
    try:
        index = IndexState(Retriever.from_settings(settings).search)
    except IndexUnavailableError as exc:
        # Say it once, at startup, where the operator will see it. Only the kind and the fix: the exception's own
        # message names a filesystem path, which does not belong in a log that might be shipped elsewhere.
        log.error("the search index is %s: every question will be refused until it is rebuilt with `python -m askact.ingest`", exc.kind)
        index = IndexState(None, exc)
    return Services(index=index, llm=_DeferredLLM(settings), spend=SpendLedger.from_settings(settings),
                    limiter=RateLimiter(settings.rate_limit_per_minute))


class _DeferredLLM:
    """Builds the provider on the first call, so a missing key or model is an `llm_unavailable` answer to a request
    and not a crash at startup, and the app (and /health) still come up."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._llm: LLM | None = None

    def complete(self, system: str, user: str, max_output_tokens: int):  # -> LLMReply
        if self._llm is None:
            self._llm = get_llm(self._settings)
        return self._llm.complete(system, user, max_output_tokens)


def build_app() -> FastAPI:
    """The production entry point: `uvicorn --factory askact.api:build_app --no-access-log`. Reads settings from
    the environment, builds the real services, returns the app. Importing this module loads nothing."""
    settings = Settings()
    return create_app(settings, build_services(settings))


def create_app(settings: Settings, services: Services) -> FastAPI:
    """Build the application. Pure wiring: nothing is loaded or connected here, so it is cheap to call in tests."""
    # No public schema or docs UI. redirect_slashes=False: with it on, POST /ask/ answers a 307 to /ask, which is a second
    # path to the endpoint whose first hop the rate limiter (an exact match on "/ask") does not count.
    app = FastAPI(title="Ask the EU AI Act", docs_url=None, redoc_url=None, openapi_url=None, redirect_slashes=False)
    config = app_config(settings)
    max_question = settings.max_question_chars
    body_limit = max_question * BYTES_PER_QUESTION_CHAR + BODY_MARGIN_BYTES

    def finish(outcome: str, started: float, *, question: str | None = None, error: BaseException | None = None,
               input_tokens: int | None = None, output_tokens: int | None = None) -> None:
        """The one log line for a request."""
        error_type = describe_error(error)[1] if error is not None else None
        log_request(
            outcome=outcome, config=config, latency_ms=(services.clock() - started) * 1000,
            input_tokens=input_tokens, output_tokens=output_tokens, error_type=error_type,
            question=question, include_question=settings.log_questions,
        )

    # --- errors: every one has the same shape, none echoes the request or an exception ---------------------------

    def started_of(request: Request) -> float:
        return getattr(request.state, "started", services.clock())

    @app.exception_handler(RequestValidationError)
    async def _invalid_body(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default 422 echoes the offending input back to the caller. This says only what was wrong.
        finish("invalid_request", started_of(request))
        return error_response("invalid_request")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if request.url.path == "/ask" and request.method == "POST" and exc.status_code == 400:
            # The framework could not read the body at all (for instance, it is not valid UTF-8).
            finish("invalid_request", started_of(request))
            return error_response("invalid_request")
        # 404 and 405 are not part of the API's contract and are not /ask requests: a plain error in the same shape.
        status = exc.status_code
        code, message = {404: ("not_found", "Not found."), 405: ("method_not_allowed", "Method not allowed.")}.get(
            status, ("http_error", "The request could not be processed."))
        return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        finish("internal_error", started_of(request), error=exc)
        return error_response("internal_error")

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc.code)

    # --- the middleware stack. The LAST one added is the OUTERMOST, and so sees the request first. ----------------------
    # Added in this order: the rate limit (innermost of the three), the body-size cap, then CORS outermost. CORS has to
    # be outermost because a response generated by an outer layer (the 413 or the 429) never passes through an inner
    # one, and an error without CORS headers looks like a network failure to the browser page instead of a message.

    def admit(scope: Scope) -> int | None:
        """The rate limit for one /ask request. Returns the seconds to wait if it is refused, otherwise None."""
        started = services.clock()
        scope.setdefault("state", {})["started"] = started
        headers = [(k, v) for k, v in scope["headers"]]
        peer = scope["client"][0] if scope.get("client") else None
        lines = [v.decode("latin-1") for k, v in headers if settings.trusted_proxy_header and k == settings.trusted_proxy_header.lower().encode("latin-1")]
        decision = services.limiter.check(client_key(peer, lines))
        if decision.allowed:
            return None
        finish("rate_limited", started)
        return decision.retry_after

    app.add_middleware(Admit, check=admit)

    def refused_too_large(scope: Scope) -> None:
        log_request(outcome="request_too_large", config=config, latency_ms=0)

    app.add_middleware(LimitRequestBody, limit=body_limit, on_refuse=refused_too_large)
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
        expose_headers=["Retry-After"],
    )

    # --- endpoints ---------------------------------------------------------------------------------------------------

    @app.get("/health")
    def health() -> JSONResponse:
        """Liveness plus whether the index is usable. 200 when the index is ready, 503 when it is not, so a
        container health check fails on a missing or stale index. Says nothing else about the deployment."""
        if services.index.ready:
            return JSONResponse({"status": "ok", "index": "ready"})
        return JSONResponse({"status": "unavailable", "index": "unavailable"}, status_code=503)

    @app.post("/ask")
    def ask(body: AskRequest, request: Request) -> JSONResponse:
        started = request.state.started
        question = body.question
        if not question.strip():
            # Nothing to search for. The retriever would raise on it and that would be reported as a server fault (500)
            # for what is the client's mistake, so say so here, before any work is done.
            finish("invalid_request", started)
            raise ApiError("invalid_request")
        if len(question) > max_question:
            finish("question_too_long", started, question=question)
            raise ApiError("question_too_long")
        try:
            if not services.index.ready:
                raise services.index.error or IndexUnavailableError("missing", "the index is not loaded")
            assert services.index.search is not None
            retrieved = services.index.search(question, config, settings.top_k)
            result = answer(question, retrieved, settings, services.llm, services.spend)
        except Exception as exc:
            outcome = "index_unavailable" if isinstance(exc, IndexUnavailableError) else describe_error(exc)[0]
            finish(outcome, started, question=question, error=exc)
            # `outcome` is always one of ERRORS: describe_error only returns known codes, and a test pins that
            # request_log's error outcomes and ERRORS are the same set.
            raise ApiError(outcome) from None
        finish(result.status, started, question=question, input_tokens=result.usage.input_tokens if result.usage else None,
               output_tokens=result.usage.output_tokens if result.usage else None)
        return JSONResponse(_payload(result))

    return app


def _payload(result: AnswerResult) -> dict[str, Any]:
    """The response body of /ask, from the design: status, answer, citations, sources."""
    return {
        "status": result.status,
        "answer": result.answer,
        "citations": [str(c) for c in result.citations],
        "sources": [
            {"section_id": str(s.section_id), "label": s.label, "title": s.title, "text": s.text, "score": s.score}
            for s in result.sources
        ],
    }
