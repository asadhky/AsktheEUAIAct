"""Per-client rate limiting for the public demo (R6.5).

Two small, framework-free pieces, so the API layer only has to wire them together:

  client_key(peer, forwarded)   who is asking: the client's IP address, as a canonical string
  RateLimiter(limit).check(key) may this client make another request now, or how long must it wait

State is in memory and per process, so the backend must run a single worker (the README lists this as a
limitation). It is a cost guard, not a security boundary: the daily spend cap (Task 16) is the hard limit,
and this keeps one client from using the whole cap, or the whole machine, by itself.
"""

import ipaddress
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

WINDOW_SECONDS = 60.0  # the limit is "per minute" (RATE_LIMIT_PER_MINUTE)
UNKNOWN_CLIENT = "unknown"  # the key when there is no address at all: those clients share one bucket


def _address_key(text: str) -> str | None:
    """The canonical key for an IP address given as text, or None if the text is not an IP address.

    IPv6 clients are keyed by their /64: one host normally owns a whole /64, so keying on the full address
    would let any IPv6 client change address for every request and never be limited. An IPv4-mapped IPv6
    address (::ffff:1.2.3.4) is the same client as the plain IPv4 address.
    """
    try:
        address = ipaddress.ip_address(text.strip())
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        return str(ipaddress.IPv6Network((int(address), 64), strict=False))  # int(): any %scope is dropped
    return str(address)


def client_key(peer: str | None, forwarded: Sequence[str] = ()) -> str:
    """The key that identifies the client of a request (R6.5).

    `peer` is the socket peer address. `forwarded` holds every line of the configured TRUSTED_PROXY_HEADER
    (empty if none is configured or the request had none). The client is the LAST entry of that header: it is
    the one our own proxy appended, while earlier entries are whatever the client chose to send. This assumes
    exactly one trusted proxy in front of the backend, and that the backend cannot be reached except through it.

    Only the LAST header line is looked at: a client can send its own header line, and a proxy may add a second
    line instead of extending the first one, so "the first value" could be the spoofed one.

    If the header is unusable (missing, blank, or its last entry is not an IP address) the socket address is used.
    Behind a proxy that makes every client share the proxy's bucket: visibly wrong, but it is the strict
    direction, never a way around the limit.
    """
    if forwarded:
        key = _address_key(forwarded[-1].rsplit(",", 1)[-1])
        if key is not None:
            return key
    if not peer:
        return UNKNOWN_CLIENT
    # A real socket peer is an IP address. Anything else came from the server itself (a test client's host name,
    # say), not from the client, so it can safely be used as it is.
    return _address_key(peer) or peer


@dataclass(frozen=True)
class Decision:
    allowed: bool
    retry_after: int  # whole seconds until the client may try again (the Retry-After header); 0 if allowed


class RateLimiter:
    """A sliding window: at most `limit` requests per client in any `window` seconds.

    Each client has a list of the times of its recent allowed requests. A request is allowed if fewer than
    `limit` of those fall inside the window, and is then added to the list. A refused request is not recorded,
    so a client that keeps retrying does not push its own wait further out.

    Memory: a client holds at most `limit` timestamps, and clients with none left inside the window are dropped
    (when they next appear, and by a sweep once per window), so memory follows the number of clients seen in the
    last window and does not grow with the number ever seen.
    """

    def __init__(self, limit: int, *, window: float = WINDOW_SECONDS, clock: Callable[[], float] = time.monotonic):
        if limit < 1:
            raise ValueError(f"the limit must be at least 1, got {limit}")
        if not window > 0:
            raise ValueError(f"the window must be positive, got {window}")
        self._limit = limit
        self._window = window
        # Monotonic: unaffected by the system clock being changed. Injectable so tests never have to sleep.
        self._clock = clock
        self._requests: dict[str, deque[float]] = {}
        self._last_sweep = clock()
        self._lock = threading.Lock()  # the API runs sync endpoints in a thread pool

    def check(self, key: str) -> Decision:
        """Record a request from `key` if it is within the limit; say how long to wait if it is not."""
        with self._lock:
            now = self._clock()
            cutoff = now - self._window  # a request exactly `window` seconds old is already outside the window
            if now - self._last_sweep >= self._window:
                self._sweep(cutoff)
                self._last_sweep = now
            recent = self._requests.get(key)
            if recent is not None:
                self._expire(recent, cutoff)
            if recent and len(recent) >= self._limit:
                # The oldest request in the window leaves it first; that is when a slot opens.
                wait = recent[0] + self._window - now
                return Decision(allowed=False, retry_after=max(1, math.ceil(wait)))
            if recent is None:
                recent = self._requests[key] = deque()
            recent.append(now)
            return Decision(allowed=True, retry_after=0)

    @staticmethod
    def _expire(recent: deque[float], cutoff: float) -> None:
        while recent and recent[0] <= cutoff:
            recent.popleft()

    def _sweep(self, cutoff: float) -> None:
        """Drop clients whose requests have all left the window."""
        for key in list(self._requests):
            recent = self._requests[key]
            self._expire(recent, cutoff)
            if not recent:
                del self._requests[key]
