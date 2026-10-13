"""`LimitRequestBody` driven directly as an ASGI app, with the messages a real server sends.

`TestClient` cannot test this middleware properly: it assembles a chunked body itself and hands the app a single
`http.request` message, so the path that reads several messages (`more_body=True`) is never reached through it.
A real server (uvicorn) does send them, so these tests feed the middleware the message sequences directly.
"""

import asyncio

import pytest

from askact.api import LimitRequestBody

LIMIT = 100


class Recorder:
    """A downstream ASGI app: reads the whole body it is given, records it, and answers 200."""

    def __init__(self):
        self.bodies: list[bytes] = []
        self.calls = 0

    async def __call__(self, scope, receive, send):
        self.calls += 1
        body = b""
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break
        self.bodies.append(body)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


def run(messages, headers=None, method="POST", limit=LIMIT):
    """Drive the middleware with `messages` and return (status, body, refused, downstream)."""
    downstream = Recorder()
    refused: list[object] = []
    middleware = LimitRequestBody(downstream, limit=limit, on_refuse=lambda scope: refused.append(scope["path"]))
    scope = {"type": "http", "method": method, "path": "/ask", "headers": headers or []}
    queue = list(messages)
    sent: list[dict] = []
    pulled = {"count": 0}

    async def receive():
        pulled["count"] += 1
        if queue:
            return queue.pop(0)
        await asyncio.sleep(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    start = next((m for m in sent if m["type"] == "http.response.start"), None)
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return (start["status"] if start else None), body, refused, downstream, pulled["count"]


def request(body: bytes, more=False):
    return {"type": "http.request", "body": body, "more_body": more}


def chunks(*sizes):
    return [request(b"x" * size, more=(i < len(sizes) - 1)) for i, size in enumerate(sizes)]


# --- bodies that fit are passed through unchanged -------------------------------------------------------------------

def test_a_small_single_message_body_is_passed_through_whole():
    status, _, refused, downstream, _ = run([request(b'{"question": "hi"}')])
    assert status == 200 and refused == [] and downstream.bodies == [b'{"question": "hi"}']


def test_a_body_in_several_chunks_that_fits_is_reassembled_in_order():
    messages = [request(b'{"question": "', True), request(b"hello", True), request(b' world"}', False)]
    status, _, refused, downstream, _ = run(messages)
    assert status == 200 and refused == [] and downstream.bodies == [b'{"question": "hello world"}']


def test_a_body_of_exactly_the_limit_is_passed_through():
    status, _, refused, downstream, _ = run(chunks(40, 40, 20))
    assert status == 200 and refused == [] and len(downstream.bodies[0]) == LIMIT


def test_an_empty_body_is_passed_through():
    status, _, _, downstream, _ = run([request(b"")])
    assert status == 200 and downstream.bodies == [b""]


def test_a_message_with_no_body_key_counts_as_empty():
    status, _, _, downstream, _ = run([{"type": "http.request", "more_body": False}])
    assert status == 200 and downstream.bodies == [b""]


# --- bodies that do not fit are refused, however they are delivered --------------------------------------------------

def test_a_single_message_over_the_limit_is_refused_and_never_reaches_the_app():
    status, body, refused, downstream, _ = run([request(b"x" * (LIMIT + 1))])
    assert status == 413 and refused == ["/ask"] and downstream.calls == 0
    assert b"request_too_large" in body


def test_several_chunks_that_are_each_small_but_together_too_big_are_refused():
    # 30 x 4 = 120 > 100: the bug "count only the latest chunk" passes this, as every chunk is under the limit.
    status, _, refused, downstream, _ = run(chunks(30, 30, 30, 30))
    assert status == 413 and refused == ["/ask"] and downstream.calls == 0


def test_the_total_is_what_counts_not_the_largest_chunk():
    status, _, _, downstream, _ = run(chunks(60, 60))
    assert status == 413 and downstream.calls == 0


def test_one_byte_over_in_the_last_chunk_is_refused():
    status, _, _, downstream, _ = run(chunks(50, 50, 1))
    assert status == 413 and downstream.calls == 0


def test_the_refusal_comes_without_waiting_for_the_rest_of_the_body():
    # A body that never ends: after the limit is crossed the middleware must stop reading.
    endless = [request(b"x" * 30, True) for _ in range(1000)]
    status, _, _, downstream, pulled = run(endless)
    assert status == 413 and pulled <= 5          # it read 4 chunks (120 bytes), not 1000
    assert downstream.calls == 0


def test_a_later_chunk_is_not_lost_when_the_body_fits():
    # The bug "forget the later chunks" would deliver only the first message.
    status, _, _, downstream, _ = run(chunks(10, 10, 10, 10, 10))
    assert status == 200 and len(downstream.bodies[0]) == 50


def test_more_body_defaulting_to_false_ends_the_body():
    status, _, _, downstream, _ = run([{"type": "http.request", "body": b"abc"}])
    assert status == 200 and downstream.bodies == [b"abc"]


# --- a declared Content-Length ------------------------------------------------------------------------------------------

def test_a_declared_length_over_the_limit_is_refused_without_reading_any_of_the_body():
    status, _, refused, downstream, pulled = run([request(b"tiny")], headers=[(b"content-length", b"10000000")])
    assert status == 413 and refused == ["/ask"] and downstream.calls == 0
    assert pulled == 0      # the declaration alone was enough: not a byte was read


def test_a_declared_length_at_the_limit_is_not_refused_on_the_declaration():
    status, _, _, downstream, _ = run([request(b"x" * LIMIT)], headers=[(b"content-length", str(LIMIT).encode())])
    assert status == 200 and len(downstream.bodies[0]) == LIMIT


def test_a_declared_length_one_over_the_limit_is_refused():
    status, _, _, downstream, pulled = run([request(b"x")], headers=[(b"content-length", str(LIMIT + 1).encode())])
    assert status == 413 and pulled == 0 and downstream.calls == 0


def test_a_lying_declared_length_cannot_get_a_big_body_past_the_cap():
    status, _, _, downstream, _ = run(chunks(60, 60), headers=[(b"content-length", b"10")])
    assert status == 413 and downstream.calls == 0


@pytest.mark.parametrize("value", [b"abc", b"", b"-5", b"1.5", b"0x10", b" 7", b"\xff"])
def test_a_content_length_that_is_not_a_plain_number_is_ignored_not_a_crash(value):
    status, _, _, downstream, _ = run([request(b"ok")], headers=[(b"content-length", value)])
    assert status == 200 and downstream.bodies == [b"ok"]


def test_a_non_numeric_content_length_does_not_let_an_oversized_body_through():
    status, _, _, downstream, _ = run(chunks(60, 60), headers=[(b"content-length", b"abc")])
    assert status == 413 and downstream.calls == 0


# --- which requests it applies to --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS", "DELETE", "PUT"])
def test_it_only_looks_at_post_requests(method):
    status, _, refused, downstream, pulled = run(chunks(60, 60), method=method)
    assert status == 200 and refused == [] and downstream.calls == 1 and pulled >= 1


def test_it_leaves_non_http_scopes_alone():
    downstream = Recorder()
    called = []

    async def inner(scope, receive, send):
        called.append(scope["type"])

    middleware = LimitRequestBody(inner, limit=LIMIT, on_refuse=lambda scope: None)
    asyncio.run(middleware({"type": "lifespan"}, None, None))     # type: ignore[arg-type]
    asyncio.run(middleware({"type": "websocket", "method": "POST"}, None, None))     # type: ignore[arg-type]
    assert called == ["lifespan", "websocket"] and downstream.calls == 0


# --- the client going away ----------------------------------------------------------------------------------------------------

def test_a_client_that_disconnects_part_way_is_dropped_quietly():
    status, _, refused, downstream, _ = run([request(b"abc", True), {"type": "http.disconnect"}])
    assert status is None and refused == [] and downstream.calls == 0       # no response to nobody, and nothing run


def test_a_client_that_disconnects_before_sending_anything_is_dropped_quietly():
    status, _, refused, downstream, _ = run([{"type": "http.disconnect"}])
    assert status is None and refused == [] and downstream.calls == 0


# --- what the refusal looks like ----------------------------------------------------------------------------------------------

def test_the_refusal_is_the_documented_error_shape():
    import json

    status, body, _, _, _ = run([request(b"x" * (LIMIT + 1))])
    parsed = json.loads(body)
    assert status == 413 and set(parsed) == {"error"} and set(parsed["error"]) == {"code", "message"}
    assert parsed["error"]["code"] == "request_too_large"


def test_the_refusal_hook_is_called_exactly_once_per_refused_request():
    refused_count = []
    downstream = Recorder()
    middleware = LimitRequestBody(downstream, limit=LIMIT, on_refuse=lambda scope: refused_count.append(1))
    queue = chunks(60, 60)

    async def receive():
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    async def send(message):
        pass

    asyncio.run(middleware({"type": "http", "method": "POST", "path": "/ask", "headers": []}, receive, send))
    assert refused_count == [1]


def test_the_limit_can_be_any_size():
    assert run(chunks(10, 10), limit=20)[0] == 200
    assert run(chunks(10, 11), limit=20)[0] == 413
    assert run(chunks(500, 500), limit=1000)[0] == 200
    assert run(chunks(500, 501), limit=1000)[0] == 413
