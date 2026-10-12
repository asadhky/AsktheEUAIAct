import inspect
import math
import threading
import time

import pytest

from askact.guards.ratelimit import UNKNOWN_CLIENT, WINDOW_SECONDS, Decision, RateLimiter, client_key


class Clock:
    """A clock the test moves by hand, so no test ever sleeps."""

    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def limiter(limit: int, **options) -> tuple[RateLimiter, Clock]:
    clock = Clock()
    return RateLimiter(limit, clock=clock, **options), clock


# =====================================================================================================
# RateLimiter
# =====================================================================================================

def test_requests_up_to_the_limit_are_allowed_and_the_next_one_is_refused():
    rl, _ = limiter(3)
    assert [rl.check("a").allowed for _ in range(3)] == [True, True, True]
    assert rl.check("a").allowed is False


def test_an_allowed_decision_has_no_wait():
    rl, _ = limiter(2)
    assert rl.check("a") == Decision(allowed=True, retry_after=0)


def test_a_refusal_says_how_many_whole_seconds_to_wait_for_the_oldest_request_to_leave_the_window():
    rl, clock = limiter(3)
    for _ in range(3):  # requests at t=0, 1, 2
        rl.check("a")
        clock.advance(1)
    clock.advance(-0.5)  # now t=2.5
    decision = rl.check("a")
    # The oldest request (t=0) leaves the window at t=60, which is 57.5 s away: rounded up to 58.
    assert decision == Decision(allowed=False, retry_after=58)


def test_the_wait_is_never_less_than_one_second_even_when_a_slot_opens_in_a_moment():
    rl, clock = limiter(1)
    rl.check("a")
    clock.advance(59.9999)
    assert rl.check("a") == Decision(allowed=False, retry_after=1)


@pytest.mark.parametrize("offset", [0.0, 0.3, 1.0, 17.25, 40.9, 59.99])
def test_retry_after_is_exact_waiting_that_long_gets_the_client_in_and_one_second_less_does_not(offset):
    rl, clock = limiter(2)
    rl.check("a")
    clock.advance(offset)
    rl.check("a")
    decision = rl.check("a")
    assert decision.allowed is False and decision.retry_after >= 1

    clock.advance(decision.retry_after - 1)
    assert rl.check("a").allowed is False   # one second early: still refused, so Retry-After is not too generous
    clock.advance(1)
    assert rl.check("a").allowed is True    # the full Retry-After: a slot is free, so it is not too short either


def test_several_requests_can_leave_the_window_at_once():
    rl, clock = limiter(5)
    for _ in range(5):
        rl.check("a")                  # five requests at t=0
    clock.advance(WINDOW_SECONDS)
    assert [rl.check("a").allowed for _ in range(5)] == [True] * 5   # all five slots are free again
    assert rl.check("a").allowed is False


def test_one_check_drops_every_request_that_has_left_the_window_not_just_the_oldest():
    rl, clock = limiter(3, window=100)
    for _ in range(3):
        rl.check("a")                       # three requests at t=0
    clock.advance(99)
    rl.check("keepalive")                   # t=99: a sweep is not due yet (it needs a full window)
    clock.advance(1)                        # t=100: all three are exactly one window old
    assert rl.check("a").allowed is True
    # By now the single check above must have dropped all three old requests, leaving only the new one.
    assert len(rl._requests["a"]) == 1


def test_retry_after_follows_a_custom_window():
    rl, clock = limiter(1, window=10)
    rl.check("a")
    clock.advance(3)
    assert rl.check("a") == Decision(allowed=False, retry_after=7)   # 10 - 3, not 60 - 3


def test_retry_after_is_at_least_one_even_when_floating_point_rounds_the_wait_to_zero():
    # A real case found by search: a request one representable step inside the window makes
    # `oldest + window - now` come out as exactly 0.0, which would send "Retry-After: 0" and invite an
    # immediate retry loop.
    now = 2107.0512450573415
    oldest = 2047.0512450573417
    assert oldest + 60.0 - now == 0.0 and oldest > now - 60.0   # the premise: still inside the window, yet no wait
    clock = Clock(oldest)
    rl = RateLimiter(1, clock=clock)
    rl.check("a")
    clock.now = now
    assert rl.check("a") == Decision(allowed=False, retry_after=1)


def test_the_window_slides_rather_than_resetting_all_at_once():
    rl, clock = limiter(2)
    rl.check("a")           # t=0
    clock.advance(30)
    rl.check("a")           # t=30
    clock.advance(29)
    assert rl.check("a").allowed is False      # t=59: both still inside the window
    clock.advance(1)
    assert rl.check("a").allowed is True       # t=60: the t=0 request has left; the t=30 one has not
    assert rl.check("a").allowed is False      # window now holds t=30 and t=60
    clock.advance(30)
    assert rl.check("a").allowed is True       # t=90: t=30 has left


def test_a_request_leaves_the_window_exactly_window_seconds_later():
    rl, clock = limiter(1)
    rl.check("a")
    clock.advance(59.999)
    assert rl.check("a").allowed is False
    clock.advance(0.001)
    assert rl.check("a").allowed is True


def test_refused_requests_are_not_recorded_so_hammering_does_not_extend_the_wait():
    rl, clock = limiter(1)
    rl.check("a")
    for _ in range(20):
        clock.advance(2)
        assert rl.check("a").allowed is False
    clock.advance(20)   # t=60 in total
    assert rl.check("a").allowed is True


def test_different_clients_have_independent_limits():
    rl, _ = limiter(1)
    assert rl.check("a").allowed is True
    assert rl.check("a").allowed is False
    assert rl.check("b").allowed is True
    assert rl.check("b").allowed is False
    assert rl.check("c").allowed is True


def test_a_limit_of_one_allows_exactly_one_request_per_window():
    rl, clock = limiter(1)
    assert [rl.check("a").allowed for _ in range(3)] == [True, False, False]
    clock.advance(WINDOW_SECONDS)
    assert rl.check("a").allowed is True


def test_the_window_length_can_be_changed():
    rl, clock = limiter(1, window=10)
    rl.check("a")
    clock.advance(9.9)
    assert rl.check("a").allowed is False
    clock.advance(0.1)
    assert rl.check("a").allowed is True


def test_the_window_is_a_minute_by_default_and_the_clock_is_monotonic():
    assert WINDOW_SECONDS == 60.0
    signature = inspect.signature(RateLimiter.__init__).parameters
    assert signature["window"].default == WINDOW_SECONDS
    assert signature["clock"].default is time.monotonic  # unaffected by the system clock being changed


def test_the_real_clock_works_without_injection():
    rl = RateLimiter(2)
    assert [rl.check("a").allowed for _ in range(3)] == [True, True, False]


@pytest.mark.parametrize("limit", [0, -1, -100])
def test_a_limit_below_one_is_refused(limit):
    with pytest.raises(ValueError, match="at least 1"):
        RateLimiter(limit)


@pytest.mark.parametrize("window", [0, -1, -0.5, math.nan])
def test_a_window_that_is_not_positive_is_refused(window):
    with pytest.raises(ValueError, match="window must be positive"):
        RateLimiter(5, window=window)


# --- memory ----------------------------------------------------------------------------------

def test_a_client_never_holds_more_timestamps_than_the_limit_however_hard_it_hammers():
    rl, clock = limiter(5)
    for _ in range(2000):
        rl.check("a")
        clock.advance(0.001)
    assert len(rl._requests["a"]) == 5


def test_clients_that_have_gone_quiet_are_dropped_by_the_sweep():
    rl, clock = limiter(5)
    for n in range(300):
        rl.check(f"client-{n}")
    assert len(rl._requests) == 300
    clock.advance(WINDOW_SECONDS * 2)
    rl.check("newcomer")  # the sweep runs on the first check after a full window
    assert set(rl._requests) == {"newcomer"}


def test_the_sweep_runs_about_once_per_window_not_on_every_request():
    rl, clock = limiter(5)
    sweeps = []
    real_sweep = rl._sweep
    rl._sweep = lambda cutoff: (sweeps.append(clock.now), real_sweep(cutoff))[1]
    for _ in range(50):                      # 50 requests spread over 25 seconds: less than one window
        rl.check("a")
        clock.advance(0.5)
    assert sweeps == []
    clock.advance(WINDOW_SECONDS)            # a full window since the limiter was created
    for _ in range(50):
        rl.check("a")
        clock.advance(0.001)
    assert len(sweeps) == 1                  # one sweep, then none for the next window
    clock.advance(WINDOW_SECONDS)
    rl.check("a")
    assert len(sweeps) == 2


def test_the_sweep_runs_exactly_when_a_full_window_has_passed_not_a_moment_before():
    rl, clock = limiter(5)
    sweeps = []
    real_sweep = rl._sweep
    rl._sweep = lambda cutoff: (sweeps.append(clock.now), real_sweep(cutoff))[1]
    clock.advance(WINDOW_SECONDS - 0.001)
    rl.check("a")
    assert sweeps == []
    clock.advance(0.001)
    rl.check("a")
    assert len(sweeps) == 1


def test_the_sweep_does_not_run_before_a_window_has_passed():
    rl, clock = limiter(5)
    for n in range(10):
        rl.check(f"client-{n}")
    clock.advance(WINDOW_SECONDS - 1)
    rl.check("newcomer")
    assert len(rl._requests) == 11


def test_the_sweep_keeps_clients_that_are_still_inside_the_window_and_their_counts():
    rl, clock = limiter(1)
    clock.advance(50)
    rl.check("busy")        # t=50
    clock.advance(50)       # t=100: a sweep is due (100 - 0 >= 60) and the cutoff is 40
    rl.check("other")
    assert "busy" in rl._requests
    assert rl.check("busy").allowed is False   # its t=50 request still counts; the sweep did not forget it
    clock.advance(10)
    assert rl.check("busy").allowed is True    # t=110: it has left the window


def test_a_client_that_comes_back_after_its_window_expired_starts_fresh():
    rl, clock = limiter(1)
    rl.check("a")
    clock.advance(WINDOW_SECONDS)
    assert rl.check("a").allowed is True
    assert len(rl._requests["a"]) == 1


def test_memory_does_not_grow_with_the_number_of_clients_ever_seen():
    rl, clock = limiter(2)
    for window_number in range(20):
        for n in range(100):
            rl.check(f"w{window_number}-c{n}")
        clock.advance(WINDOW_SECONDS + 1)
    rl.check("last")
    assert len(rl._requests) < 150   # about one window's worth of clients, not 2000


# --- threads (the API runs sync endpoints in a thread pool) ------------------------------------------

def test_concurrent_requests_from_one_client_never_exceed_the_limit():
    rl, _ = limiter(25)  # a fixed clock: every request falls in the same window
    results: list[bool] = []
    results_lock = threading.Lock()

    def worker():
        mine = [rl.check("a").allowed for _ in range(100)]
        with results_lock:
            results.extend(mine)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 800 and sum(results) == 25


def test_the_state_is_only_touched_while_the_lock_is_held():
    # Threads under CPython's GIL rarely expose a missing lock, so check it directly: record whether the lock
    # is held at the moment the clock is read and the request table is changed.
    rl, clock = limiter(5)
    held_during = []

    class Table(dict):
        def __setitem__(self, key, value):
            held_during.append(("write", rl._lock.locked()))
            super().__setitem__(key, value)

    rl._requests = Table()
    real_clock = rl._clock
    rl._clock = lambda: (held_during.append(("clock", rl._lock.locked())), real_clock())[1]
    rl.check("a")
    assert held_during and all(held for _, held in held_during), held_during
    assert {kind for kind, _ in held_during} == {"clock", "write"}


def test_the_lock_is_released_after_every_check_including_a_refusal():
    rl, _ = limiter(1)
    rl.check("a")
    rl.check("a")                 # refused: returns early from inside the lock
    assert rl._lock.locked() is False


def test_the_lock_is_released_if_the_clock_raises():
    def broken():
        raise RuntimeError("clock failed")

    rl, _ = limiter(1)   # built with a working clock: the constructor reads the clock too
    rl._clock = broken
    with pytest.raises(RuntimeError):
        rl.check("a")
    assert rl._lock.locked() is False


# =====================================================================================================
# client_key
# =====================================================================================================

def test_without_a_proxy_header_the_client_is_the_socket_peer():
    assert client_key("203.0.113.7") == "203.0.113.7"
    assert client_key("203.0.113.7", []) == "203.0.113.7"
    assert client_key("203.0.113.7", ()) == "203.0.113.7"


def test_the_client_is_the_last_entry_of_the_proxy_header_not_an_earlier_one():
    # "client-supplied, ..., appended by our proxy": only the last entry is trustworthy.
    assert client_key("10.0.0.1", ["6.6.6.6, 7.7.7.7, 198.51.100.9"]) == "198.51.100.9"


def test_a_single_entry_header_is_that_entry():
    assert client_key("10.0.0.1", ["198.51.100.9"]) == "198.51.100.9"


def test_the_header_wins_over_the_peer_when_it_is_usable():
    assert client_key("10.0.0.1", ["198.51.100.9"]) != client_key("10.0.0.1")


def test_entries_are_trimmed():
    assert client_key("10.0.0.1", ["  6.6.6.6 ,   198.51.100.9   "]) == "198.51.100.9"


def test_with_several_header_lines_the_last_entry_of_the_last_line_is_used():
    # A client can send its own X-Forwarded-For line, and a proxy may add a second line instead of extending
    # the first one. Taking "the first value" would hand the client its own bucket.
    assert client_key("10.0.0.1", ["6.6.6.6", "198.51.100.9"]) == "198.51.100.9"
    assert client_key("10.0.0.1", ["spoofed, 5.5.5.5", "6.6.6.6, 198.51.100.9"]) == "198.51.100.9"
    assert client_key("10.0.0.1", ["1.1.1.1", "2.2.2.2", "198.51.100.9"]) == "198.51.100.9"


def test_spoofed_earlier_entries_do_not_change_the_clients_key():
    keys = {client_key("10.0.0.1", [f"{n}.{n}.{n}.{n}, 198.51.100.9"]) for n in range(1, 50)}
    assert keys == {"198.51.100.9"}


@pytest.mark.parametrize("header", [
    [""], ["   "], ["unknown"], ["not-an-ip"], ["6.6.6.6, not-an-ip"], ["198.51.100.9,"], ["198.51.100.9, "],
    ["198.51.100.9:8080"], ["[2001:db8::1]"], ["999.1.1.1"], ["1.2.3"], ["1.2.3.4.5"], ["<script>"], ["198.51.100.9;6.6.6.6"],
])
def test_an_unusable_header_falls_back_to_the_socket_peer(header):
    assert client_key("10.0.0.1", header) == "10.0.0.1"


def test_the_fallback_is_the_same_whatever_garbage_the_client_sent():
    # A client must not be able to choose its bucket by sending junk: all junk lands in the peer's bucket.
    keys = {client_key("10.0.0.1", [junk]) for junk in ("x", "y", "zzz", "", "1.2.3", "1.1.1.1.1")}
    assert keys == {"10.0.0.1"}


def test_when_the_header_is_unusable_and_there_is_no_peer_the_client_is_unknown():
    assert client_key(None, ["garbage"]) == UNKNOWN_CLIENT
    assert client_key("", ["garbage"]) == UNKNOWN_CLIENT
    assert client_key(None) == UNKNOWN_CLIENT
    assert client_key("") == UNKNOWN_CLIENT


def test_a_peer_that_is_not_an_ip_address_is_used_as_it_is():
    # What a test client reports. It came from the server, not the client, so it cannot be spoofed.
    assert client_key("testclient") == "testclient"


# --- IPv6 -------------------------------------------------------------------------------------

def test_ipv6_clients_are_keyed_by_their_slash_64_so_changing_address_does_not_escape_the_limit():
    a = client_key("2001:db8:abcd:12::1")
    b = client_key("2001:db8:abcd:12:ffff:ffff:ffff:ffff")
    c = client_key("2001:DB8:ABCD:12:0:0:0:5")
    assert a == b == c


def test_different_ipv6_networks_are_different_clients():
    assert client_key("2001:db8:abcd:12::1") != client_key("2001:db8:abcd:13::1")
    assert client_key("2001:db8:abcd:12::1") != client_key("2001:db8:abce:12::1")


def test_ipv6_in_the_proxy_header_is_keyed_the_same_way():
    assert client_key("10.0.0.1", ["6.6.6.6, 2001:db8:abcd:12::77"]) == client_key("2001:db8:abcd:12::1")


def test_an_ipv6_scope_id_is_ignored():
    assert client_key("fe80::1%eth0") == client_key("fe80::1")


def test_an_ipv4_mapped_ipv6_address_is_the_same_client_as_the_ipv4_address():
    assert client_key("::ffff:203.0.113.7") == client_key("203.0.113.7") == "203.0.113.7"
    assert client_key("10.0.0.1", ["::ffff:198.51.100.9"]) == "198.51.100.9"


def test_ipv4_clients_are_not_grouped():
    assert client_key("203.0.113.7") != client_key("203.0.113.8")


def test_ipv4_and_ipv6_keys_cannot_collide():
    assert client_key("0.0.0.1") != client_key("::1")


# --- the two together ------------------------------------------------------------------------------

def test_a_client_that_spoofs_earlier_header_entries_still_hits_one_limit():
    rl, _ = limiter(3)
    allowed = [rl.check(client_key("10.0.0.1", [f"{n}.{n}.{n}.{n}, 198.51.100.9"])).allowed for n in range(1, 11)]
    assert allowed == [True] * 3 + [False] * 7


def test_two_real_clients_behind_the_proxy_have_separate_limits():
    rl, _ = limiter(1)
    assert rl.check(client_key("10.0.0.1", ["198.51.100.1"])).allowed is True
    assert rl.check(client_key("10.0.0.1", ["198.51.100.1"])).allowed is False
    assert rl.check(client_key("10.0.0.1", ["198.51.100.2"])).allowed is True


def test_a_broken_proxy_header_puts_everyone_in_the_proxys_bucket_which_is_strict_not_open():
    rl, _ = limiter(2)
    decisions = [rl.check(client_key("10.0.0.1", ["no-ip-here"])).allowed for _ in range(5)]
    assert decisions == [True, True, False, False, False]
