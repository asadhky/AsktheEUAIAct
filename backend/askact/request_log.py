"""The one log line per request, and nothing else the request path should ever write about a user.

Privacy is a property of how the line is built, not of care taken by the caller:

  * The line is assembled from an allowlist. `log_request` has parameters for the outcome, the retrieval config,
    the latency and the token counts, and nowhere to put a question, an answer, a passage or an address. A bug
    elsewhere cannot add one by accident because there is no field for it.
  * The outcome is checked against a closed set, so a caller cannot smuggle request data into the line by
    passing `str(exception)` as the "outcome".
  * The line is one JSON object, ASCII only. A question containing a newline cannot start a second, forged log
    line (the plain-text equivalent can), and U+2028, which some log viewers treat as a line break, is escaped.
  * `LOG_QUESTIONS=true` is the single opt-in, for debugging on your own machine. It adds one extra field,
    `question`, and nothing else; it is off by default and must stay off on the public demo.

For errors, `describe_error` gives the API's error code and the exception's *type name*: never its message,
arguments, cause or traceback, because those can contain request data (a provider's reply, a SQL error naming a
path, an echoed prompt). Nothing in the request path calls `logger.exception()` or passes `exc_info`, for the same
reason: a traceback prints the text of the exception.

The client address is not here at all. The rate limiter keeps client keys in memory only, and the server is run
with `--no-access-log` because uvicorn's access log records the address and the path (Task 22).
"""

import json
import logging
import math
from datetime import UTC, datetime

from askact.retrieval import CONFIGS

# `answered` and `not_covered` are the two successful outcomes; the rest are the error codes of the API table in
# design 3.4. Keep this list in step with it. A closed set, so that a caller cannot pass free text as the outcome.
SUCCESS_OUTCOMES = frozenset({"answered", "not_covered"})
ERROR_OUTCOMES = frozenset({
    "question_too_long",
    "rate_limited",
    "spend_cap_reached",
    "llm_unavailable",
    "llm_bad_output",
    "index_unavailable",
    "not_configured",
    "internal_error",  # an exception nobody planned for: the API answers 500 and logs only its type
})
OUTCOMES = SUCCESS_OUTCOMES | ERROR_OUTCOMES

# With LOG_QUESTIONS=true a question is written out. The API rejects an over-long one (question_too_long), but that
# rejection is itself logged, so without a cut a huge request body would be copied whole into the log.
MAX_LOGGED_QUESTION_CHARS = 1000
TRUNCATION_MARK = "...[truncated]"

log = logging.getLogger("askact.request")


def describe_error(exc: BaseException) -> tuple[str, str]:
    """(outcome code, exception type name) for an exception: never its message, arguments, cause or traceback.

    The code is the exception's own `code` attribute if it has one and it is a known outcome, otherwise
    `internal_error`. Reading `code` is safe: it is a class-level constant on our own errors, and an unknown or
    odd value (say, a library exception that happens to have a `code` attribute) is not trusted.
    """
    code = getattr(type(exc), "code", None)
    outcome = code if isinstance(code, str) and code in ERROR_OUTCOMES else "internal_error"
    return outcome, type(exc).__name__


def _token_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return value


def format_request(
    *,
    outcome: str,
    config: str,
    latency_ms: float,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    error_type: str | None = None,
    question: str | None = None,
    now: datetime | None = None,
) -> str:
    """The log line, as a single-line ASCII JSON object.

    Every value that is not a number comes from a closed set or is checked to be a plain identifier, so the only
    free text the line can ever hold is the optional `question`. `error_type` is only for error outcomes and is the
    exception's type name (see `describe_error`). `question` is added only if it is given, and the caller gives it
    only when LOG_QUESTIONS is true.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome {outcome!r}: expected one of {sorted(OUTCOMES)}")
    if config not in CONFIGS:
        raise ValueError(f"unknown retrieval config {config!r}: expected one of {', '.join(CONFIGS)}")
    if error_type is not None:
        if outcome not in ERROR_OUTCOMES:
            raise ValueError("error_type is only for error outcomes")
        if not (error_type.isidentifier() and error_type.isascii() and len(error_type) <= 100):
            raise ValueError("error_type must be an exception type name")
    if not isinstance(latency_ms, (int, float)) or isinstance(latency_ms, bool) or not math.isfinite(latency_ms) or latency_ms < 0:
        raise ValueError("latency_ms must be a non-negative finite number")
    moment = now if now is not None else datetime.now(UTC)
    if moment.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    fields: dict[str, object] = {
        "ts": moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.astimezone(UTC).microsecond // 1000:03d}Z",
        "event": "ask",
        "outcome": outcome,
        "config": config,
        "latency_ms": round(latency_ms),
        # None (JSON null) means the provider did not report usage, or that no LLM call was made.
        "input_tokens": _token_count(input_tokens, "input_tokens"),
        "output_tokens": _token_count(output_tokens, "output_tokens"),
    }
    if error_type is not None:
        fields["error_type"] = error_type
    if question is not None:
        fields["question"] = (
            question if len(question) <= MAX_LOGGED_QUESTION_CHARS else question[:MAX_LOGGED_QUESTION_CHARS] + TRUNCATION_MARK
        )
    # ensure_ascii: one physical line, whatever the characters. separators: compact.
    return json.dumps(fields, ensure_ascii=True, separators=(",", ":"))


def log_request(
    *,
    outcome: str,
    config: str,
    latency_ms: float,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    error_type: str | None = None,
    question: str | None = None,
    include_question: bool = False,
    now: datetime | None = None,
) -> None:
    """Write the one line for a request. `question` is written only if `include_question` is true (LOG_QUESTIONS)."""
    line = format_request(
        outcome=outcome,
        config=config,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error_type=error_type,
        question=question if include_question else None,
        now=now,
    )
    # The line is an argument, never the template: a "%" in it would otherwise be read as a format directive.
    log.info("%s", line)
