"""Reading the model's reply (R3.2, R3.3).

The reply must be one JSON object `{"answer": str, "citations": [str], "covered": bool}`. A code fence
around it is tolerated, because models add one even when told not to. Anything else, including prose around
the JSON or a field of the wrong type, is `LLMBadOutputError`: the reply is never shown as an answer.

This only checks the *shape*. Whether the citations are real, and what `covered` means for the response,
is decided by the answer pipeline, which knows the retrieved set.
"""

import json
import re

from pydantic import BaseModel, StrictBool, StrictStr, ValidationError

from askact.generation.llm import LLMBadOutputError


class ModelReply(BaseModel):
    answer: StrictStr
    # Kept as the model wrote them. A citation that is not a valid section id, or not in the retrieved set,
    # is dropped by the answer pipeline; it must not make the whole reply unusable.
    citations: list[StrictStr]
    covered: StrictBool  # strict: the string "true" or the number 1 are not booleans


# An opening ``` with an optional language tag, the content, and a closing ``` at the very end. The content
# may span lines; the whitespace around it is left for json.loads, which ignores it.
_FENCE = re.compile(r"\A```[ \t]*(?:json)?(.*?)```\Z", re.DOTALL | re.IGNORECASE)


def parse_reply(text: str) -> ModelReply:
    """The reply as a `ModelReply`, or `LLMBadOutputError` saying what was wrong (never quoting the reply)."""
    stripped = text.strip()
    fenced = _FENCE.match(stripped)
    try:
        data = json.loads(fenced.group(1) if fenced else stripped)
    except json.JSONDecodeError:
        raise LLMBadOutputError("the model's reply is not a single JSON object") from None
    if not isinstance(data, dict):
        raise LLMBadOutputError("the model's reply is JSON but not an object")
    try:
        reply = ModelReply.model_validate(data)
    except ValidationError as exc:
        # Only where the problem is, never the offending value: it is untrusted model output.
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        raise LLMBadOutputError(f"the model's reply does not have the required fields ({problems})") from None
    if reply.covered and not reply.answer.strip():
        raise LLMBadOutputError("the model's reply says the question is covered but gives no answer")
    return reply
