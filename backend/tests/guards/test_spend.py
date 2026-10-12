import json
import logging
import math
import sqlite3
import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsError

from askact.config import ModelPrice, Settings
from askact.generation.answer import SpendGuard, SpendRefusedError, answer
from askact.generation.llm import FakeLLM, LLMReply
from askact.guards.spend import SpendLedger, _cap_micro, _cost_micro, estimate_input_tokens
from askact.models import Chunk, RetrievalResult, ScoredChunk, SectionId

# Prices chosen so that the arithmetic is easy to do by hand: micro-dollars per token = USD per million tokens.
PRICE = ModelPrice(input_per_mtok=1.0, output_per_mtok=4.0)
MODEL = "test-model"
NOON = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self, now: datetime = NOON):
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def ledger(tmp_path, cap_usd=1.0, prices=None, model=MODEL, clock=None, name="spend.sqlite3") -> SpendLedger:
    return SpendLedger(tmp_path / name, cap_usd, model, {MODEL: PRICE} if prices is None else prices, clock=clock or Clock())


def rows(path: Path) -> dict[str, int]:
    with sqlite3.connect(path) as db:
        return dict(db.execute("SELECT day, micro_usd FROM spend").fetchall())


def prompt(chars: int) -> tuple[str, str]:
    """A system and user prompt of `chars` ASCII characters in total: chars/3 estimated input tokens, rounded up."""
    return "s" * (chars // 2), "u" * (chars - chars // 2)


def refused(call) -> str:
    with pytest.raises(SpendRefusedError) as info:
        call()
    assert info.value.code == "spend_cap_reached"
    return str(info.value)


# 300 chars -> 100 estimated input tokens; 100 * 1.0 + 50 * 4.0 = 300 micro-dollars
SYSTEM, USER = prompt(300)
RESERVED = 300


# =====================================================================================================
# estimating and pricing
# =====================================================================================================

@pytest.mark.parametrize("chars, tokens", [(0, 0), (1, 1), (2, 1), (3, 1), (4, 2), (300, 100), (301, 101), (9000, 3000)])
def test_the_estimate_is_the_byte_length_divided_by_three_rounded_up(chars, tokens):
    assert estimate_input_tokens("a" * chars, "") == tokens


def test_the_estimate_counts_the_system_prompt_and_the_user_prompt_together():
    assert estimate_input_tokens("a" * 30, "b" * 30) == 20


def test_the_estimate_reserves_more_for_text_where_a_character_costs_more_than_a_token_fraction():
    # 'é' is 2 bytes, a CJK character 3 and an emoji 4: each reserves more than the same number of ASCII characters.
    ascii_tokens = estimate_input_tokens("a" * 30, "")
    assert estimate_input_tokens("é" * 30, "") == 20 > ascii_tokens == 10
    assert estimate_input_tokens("日" * 30, "") == 30
    assert estimate_input_tokens("😀" * 30, "") == 40


@pytest.mark.parametrize("price, tokens_in, tokens_out, expected", [
    (ModelPrice(input_per_mtok=1.0, output_per_mtok=4.0), 100, 50, 300),          # exact
    (ModelPrice(input_per_mtok=0.15, output_per_mtok=0.60), 1000, 1000, 750),     # 150 + 600
    (ModelPrice(input_per_mtok=0.15, output_per_mtok=0.60), 1, 1, 1),             # 0.75 rounds UP to 1
    (ModelPrice(input_per_mtok=0.15, output_per_mtok=0.0), 1, 0, 1),              # 0.15 rounds UP to 1
    (ModelPrice(input_per_mtok=0.0, output_per_mtok=0.0), 5000, 5000, 0),         # a free model costs nothing
    (ModelPrice(input_per_mtok=2.5, output_per_mtok=10.0), 0, 0, 0),
])
def test_cost_is_tokens_times_price_in_micro_dollars_rounded_up(price, tokens_in, tokens_out, expected):
    assert _cost_micro(price, tokens_in, tokens_out) == expected


@pytest.mark.parametrize("price, tokens, exact", [(1.1, 100, 110), (1.1, 50, 55), (1.1, 200, 220), (1.1, 90, 99)])
def test_an_exact_product_is_not_rounded_up_by_floating_point_error(price, tokens, exact):
    # Each of these is a real case where floats get it wrong: 1.1 * 100 is 110.00000000000001, and a ceiling of
    # that is 111, an overcharge on every call at that price. The ledger does the sum in Decimal.
    assert math.ceil(price * tokens) == exact + 1, "the float trap this test guards against is not there any more"
    assert _cost_micro(ModelPrice(input_per_mtok=price, output_per_mtok=0.0), tokens, 0) == exact
    assert _cost_micro(ModelPrice(input_per_mtok=0.0, output_per_mtok=price), 0, tokens) == exact


def test_a_product_that_really_is_fractional_still_rounds_up():
    assert _cost_micro(ModelPrice(input_per_mtok=1.1, output_per_mtok=0.0), 101, 0) == 112   # 111.1 -> 112


def test_input_and_output_use_their_own_prices():
    price = ModelPrice(input_per_mtok=1.0, output_per_mtok=100.0)
    assert _cost_micro(price, 10, 0) == 10
    assert _cost_micro(price, 0, 10) == 1000


def test_small_prices_in_scientific_notation_are_exact():
    assert _cost_micro(ModelPrice(input_per_mtok=1e-05, output_per_mtok=0.0), 100_000, 0) == 1
    assert _cost_micro(ModelPrice(input_per_mtok=2.5e-05, output_per_mtok=0.0), 100_000, 0) == 3   # 2.5, rounded up


@pytest.mark.parametrize("tokens_in, tokens_out", [(-1, 0), (0, -1), (-5, -5), (True, 0), (0, False), (1.5, 0), (0, "3"), (None, 0)])
def test_token_counts_that_are_not_non_negative_integers_are_refused(tokens_in, tokens_out):
    with pytest.raises(ValueError, match="non-negative integer"):
        _cost_micro(PRICE, tokens_in, tokens_out)


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_a_price_that_is_not_finite_is_refused_by_the_ledger_even_if_it_got_past_validation(bad):
    # model_construct skips validation, as a settings object built in code could.
    price = ModelPrice.model_construct(input_per_mtok=bad, output_per_mtok=1.0)
    with pytest.raises((ValueError, ArithmeticError)):
        _cost_micro(price, 10, 10)


@pytest.mark.parametrize("cap, micro", [(1.0, 1_000_000), (0.0, 0), (0.5, 500_000), (0.000001, 1), (0.0000019, 1), (0.0000009, 0), (12.34, 12_340_000)])
def test_the_cap_is_converted_to_micro_dollars_rounding_down(cap, micro):
    assert _cap_micro(cap) == micro


@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_a_cap_that_is_not_finite_is_refused(bad):
    with pytest.raises((ValueError, ArithmeticError)):
        _cap_micro(bad)


# =====================================================================================================
# reserving (R6.7, R6.8)
# =====================================================================================================

def test_a_reservation_adds_the_worst_case_cost_to_todays_total(tmp_path):
    led = ledger(tmp_path)
    led.reserve(SYSTEM, USER, 50)
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": RESERVED}


def test_reservations_accumulate(tmp_path):
    led = ledger(tmp_path)
    for _ in range(4):
        led.reserve(SYSTEM, USER, 50)
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": 4 * RESERVED}


def test_the_reservation_covers_the_estimated_input_and_the_full_output_cap(tmp_path):
    led = ledger(tmp_path)
    led.reserve(SYSTEM, USER, 200)   # 100 in * 1.0 + 200 out * 4.0
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": 100 + 800}


def test_a_reservation_that_fits_exactly_is_allowed(tmp_path):
    led = ledger(tmp_path, cap_usd=3 * RESERVED / 1_000_000)
    for _ in range(3):
        led.reserve(SYSTEM, USER, 50)
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": 3 * RESERVED}


def test_the_reservation_that_would_exceed_the_cap_by_one_micro_dollar_is_refused(tmp_path):
    led = ledger(tmp_path, cap_usd=(3 * RESERVED - 1) / 1_000_000)
    led.reserve(SYSTEM, USER, 50)
    led.reserve(SYSTEM, USER, 50)
    message = refused(lambda: led.reserve(SYSTEM, USER, 50))
    assert "daily spend cap has been reached" in message and "tomorrow" in message
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": 2 * RESERVED}   # the refusal reserved nothing


def test_a_single_call_bigger_than_the_whole_cap_is_refused_on_an_empty_ledger(tmp_path):
    led = ledger(tmp_path, cap_usd=(RESERVED - 1) / 1_000_000)
    refused(lambda: led.reserve(SYSTEM, USER, 50))
    # The refusal rolled the whole transaction back, including creating the table: nothing at all was written.
    with sqlite3.connect(tmp_path / "spend.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM sqlite_master").fetchone() == (0,)


def test_a_refused_reservation_does_not_block_a_smaller_one_that_fits(tmp_path):
    led = ledger(tmp_path, cap_usd=500 / 1_000_000)
    refused(lambda: led.reserve(SYSTEM, USER, 200))        # 900 > 500
    led.reserve(SYSTEM, USER, 50)                           # 300 <= 500
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": RESERVED}


def test_a_cap_of_zero_refuses_every_priced_call(tmp_path):
    refused(lambda: ledger(tmp_path, cap_usd=0.0).reserve(SYSTEM, USER, 50))


def test_a_free_model_is_allowed_even_with_a_cap_of_zero(tmp_path):
    free = {MODEL: ModelPrice(input_per_mtok=0.0, output_per_mtok=0.0)}
    ledger(tmp_path, cap_usd=0.0, prices=free).reserve(SYSTEM, USER, 50)
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": 0}


def test_the_ledger_is_a_spend_guard(tmp_path):
    guard: SpendGuard = ledger(tmp_path)
    assert guard.reserve(SYSTEM, USER, 50) is not None


# =====================================================================================================
# reconciling (R6.7, R6.9)
# =====================================================================================================

def test_reconciling_down_hands_back_the_unused_part_of_the_reservation(tmp_path):
    led = ledger(tmp_path)
    led.reserve(SYSTEM, USER, 50).reconcile(40, 5)    # 40*1 + 5*4 = 60 instead of 300
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": 60}


def test_reconciling_up_charges_the_real_cost_even_above_the_reservation(tmp_path):
    # The estimate is not a bound (see estimate_input_tokens), and money already spent cannot be un-spent.
    led = ledger(tmp_path)
    led.reserve(SYSTEM, USER, 50).reconcile(1000, 50)  # 1000 + 200 = 1200 instead of 300
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": 1200}


def test_reconciling_to_exactly_the_reservation_changes_nothing(tmp_path):
    led = ledger(tmp_path)
    led.reserve(SYSTEM, USER, 50).reconcile(100, 50)
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": RESERVED}


def test_reconciling_only_replaces_its_own_reservation_not_the_others(tmp_path):
    led = ledger(tmp_path)
    first = led.reserve(SYSTEM, USER, 50)
    led.reserve(SYSTEM, USER, 50)
    first.reconcile(10, 10)          # 10 + 40 = 50 instead of 300
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": 50 + RESERVED}


def test_zero_tokens_is_a_valid_report_and_costs_nothing(tmp_path):
    led = ledger(tmp_path)
    led.reserve(SYSTEM, USER, 50).reconcile(0, 0)
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": 0}


def test_the_cost_of_a_reconciliation_is_rounded_up_like_a_reservation(tmp_path):
    prices = {MODEL: ModelPrice(input_per_mtok=0.15, output_per_mtok=0.60)}
    led = ledger(tmp_path, prices=prices)
    led.reserve(SYSTEM, USER, 50).reconcile(1, 1)   # 0.75 -> 1
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": 1}


@pytest.mark.parametrize("tokens", [(None, None), (100, None), (None, 50)])
def test_when_usage_is_missing_or_only_half_reported_the_reservation_stands(tmp_path, tokens):
    led = ledger(tmp_path)
    led.reserve(SYSTEM, USER, 50).reconcile(*tokens)
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": RESERVED}


@pytest.mark.parametrize("tokens", [(None, None), (100, None), (None, 50)])
def test_missing_usage_is_normal_and_is_not_logged_as_a_failure(tmp_path, caplog, tokens):
    led = ledger(tmp_path)
    reservation = led.reserve(SYSTEM, USER, 50)
    with caplog.at_level(logging.DEBUG, logger="askact.guards.spend"):
        reservation.reconcile(*tokens)
    assert caplog.records == []


def test_when_reconciliation_never_happens_the_reservation_stands(tmp_path):
    ledger(tmp_path).reserve(SYSTEM, USER, 50)   # the process "dies" here: no reconcile
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": RESERVED}


def test_a_reservation_can_only_be_settled_once(tmp_path):
    led = ledger(tmp_path)
    reservation = led.reserve(SYSTEM, USER, 50)
    reservation.reconcile(10, 10)
    reservation.reconcile(500, 500)    # ignored
    reservation.reconcile(None, None)  # ignored
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": 50}


def test_a_settlement_that_reports_no_usage_still_counts_as_settled(tmp_path):
    led = ledger(tmp_path)
    reservation = led.reserve(SYSTEM, USER, 50)
    reservation.reconcile(None, None)
    reservation.reconcile(10, 10)      # a later, different report cannot change what was charged
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": RESERVED}


@pytest.mark.parametrize("tokens", [(-1, 5), (5, -1), (-100, -100)])
def test_negative_usage_cannot_act_as_a_credit(tmp_path, tokens):
    led = ledger(tmp_path)
    led.reserve(SYSTEM, USER, 50).reconcile(*tokens)
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": RESERVED}   # the reservation stands


def test_reconciling_when_the_days_row_has_vanished_records_the_actual_cost_and_never_goes_negative(tmp_path):
    led = ledger(tmp_path)
    reservation = led.reserve(SYSTEM, USER, 50)
    with sqlite3.connect(tmp_path / "spend.sqlite3") as db:
        db.execute("DELETE FROM spend")
    reservation.reconcile(10, 10)
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": 50}


# =====================================================================================================
# the day (R6.10)
# =====================================================================================================

def test_a_new_day_starts_at_zero(tmp_path):
    clock = Clock()
    led = ledger(tmp_path, cap_usd=RESERVED / 1_000_000, clock=clock)
    led.reserve(SYSTEM, USER, 50)
    refused(lambda: led.reserve(SYSTEM, USER, 50))      # today is full
    clock.now += timedelta(days=1)
    led.reserve(SYSTEM, USER, 50)                        # tomorrow is empty
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": RESERVED, "2026-10-13": RESERVED}


@pytest.mark.parametrize("moment, day", [
    (datetime(2026, 10, 12, 23, 59, 59, 999999, tzinfo=UTC), "2026-10-12"),
    (datetime(2026, 10, 13, 0, 0, 0, tzinfo=UTC), "2026-10-13"),
    (datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC), "2026-01-01"),
    (datetime(2026, 12, 31, 23, 59, 59, tzinfo=UTC), "2026-12-31"),
])
def test_the_day_changes_exactly_at_midnight_utc(tmp_path, moment, day):
    ledger(tmp_path, clock=Clock(moment)).reserve(SYSTEM, USER, 50)
    assert list(rows(tmp_path / "spend.sqlite3")) == [day]


def test_the_day_is_the_utc_day_whatever_timezone_the_clock_uses(tmp_path):
    plus_five = timezone(timedelta(hours=5))
    minus_eight = timezone(timedelta(hours=-8))
    # 02:00 on the 13th in +05:00 is 21:00 on the 12th in UTC; 20:00 on the 12th in -08:00 is 04:00 on the 13th in UTC.
    # Each is checked on its own: the two days swapped would give the same pair of rows.
    ledger(tmp_path, clock=Clock(datetime(2026, 10, 13, 2, 0, tzinfo=plus_five)), name="a.sqlite3").reserve(SYSTEM, USER, 50)
    assert rows(tmp_path / "a.sqlite3") == {"2026-10-12": RESERVED}
    ledger(tmp_path, clock=Clock(datetime(2026, 10, 12, 20, 0, tzinfo=minus_eight)), name="b.sqlite3").reserve(SYSTEM, USER, 50)
    assert rows(tmp_path / "b.sqlite3") == {"2026-10-13": RESERVED}


def test_a_clock_without_a_timezone_is_refused_rather_than_guessed(tmp_path):
    led = ledger(tmp_path, clock=Clock(datetime(2026, 10, 12, 12, 0)))   # naive
    refused(lambda: led.reserve(SYSTEM, USER, 50))
    assert not (tmp_path / "spend.sqlite3").exists()


def test_a_call_that_straddles_midnight_is_settled_on_the_day_it_was_reserved(tmp_path):
    clock = Clock(datetime(2026, 10, 12, 23, 59, 58, tzinfo=UTC))
    led = ledger(tmp_path, clock=clock)
    reservation = led.reserve(SYSTEM, USER, 50)
    clock.now = datetime(2026, 10, 13, 0, 0, 3, tzinfo=UTC)      # the call took five seconds
    reservation.reconcile(10, 10)
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": 50}   # yesterday's row changed; no row for today


def test_a_reservation_made_just_after_midnight_does_not_touch_yesterday(tmp_path):
    clock = Clock(datetime(2026, 10, 12, 23, 59, 58, tzinfo=UTC))
    led = ledger(tmp_path, clock=clock)
    led.reserve(SYSTEM, USER, 50)
    clock.now = datetime(2026, 10, 13, 0, 0, 1, tzinfo=UTC)
    led.reserve(SYSTEM, USER, 50)
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": RESERVED, "2026-10-13": RESERVED}


def test_the_real_clock_is_used_by_default_and_is_utc(tmp_path):
    before = datetime.now(UTC).strftime("%Y-%m-%d")
    SpendLedger(tmp_path / "s.sqlite3", 1.0, MODEL, {MODEL: PRICE}).reserve(SYSTEM, USER, 50)
    after = datetime.now(UTC).strftime("%Y-%m-%d")
    [day] = rows(tmp_path / "s.sqlite3")
    assert day in {before, after}   # the date can only differ if midnight UTC passed during this test


# =====================================================================================================
# persistence (R6.10)
# =====================================================================================================

def test_the_total_survives_a_new_ledger_object_on_the_same_file(tmp_path):
    ledger(tmp_path).reserve(SYSTEM, USER, 50)
    reopened = ledger(tmp_path, cap_usd=2 * RESERVED / 1_000_000)   # as after a restart of the process or container
    reopened.reserve(SYSTEM, USER, 50)
    refused(lambda: reopened.reserve(SYSTEM, USER, 50))             # the first call before the restart still counts
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": 2 * RESERVED}


def test_a_restart_cannot_be_used_to_reset_the_cap_within_the_day(tmp_path):
    cap = RESERVED / 1_000_000
    ledger(tmp_path, cap_usd=cap).reserve(SYSTEM, USER, 50)
    for _ in range(3):
        refused(lambda: ledger(tmp_path, cap_usd=cap).reserve(SYSTEM, USER, 50))


def test_the_database_directory_is_created_if_it_does_not_exist(tmp_path):
    led = SpendLedger(tmp_path / "a" / "b" / "spend.sqlite3", 1.0, MODEL, {MODEL: PRICE}, clock=Clock())
    led.reserve(SYSTEM, USER, 50)
    assert rows(tmp_path / "a" / "b" / "spend.sqlite3") == {"2026-10-12": RESERVED}


def test_the_table_is_created_on_first_use_and_an_existing_one_is_reused(tmp_path):
    ledger(tmp_path).reserve(SYSTEM, USER, 50)
    ledger(tmp_path).reserve(SYSTEM, USER, 50)
    with sqlite3.connect(tmp_path / "spend.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM sqlite_master WHERE name = 'spend'").fetchone() == (1,)


def test_a_reservation_is_on_disk_before_the_call_so_another_connection_sees_it(tmp_path):
    ledger(tmp_path).reserve(SYSTEM, USER, 50)
    other = sqlite3.connect(tmp_path / "spend.sqlite3")   # no commit by us: the ledger must already have committed
    assert other.execute("SELECT micro_usd FROM spend").fetchone() == (RESERVED,)


# =====================================================================================================
# failing closed (R6.8, R6.9)
# =====================================================================================================

def test_a_model_with_no_price_is_refused_and_the_database_is_not_touched(tmp_path):
    led = ledger(tmp_path, model="other-model")
    message = refused(lambda: led.reserve(SYSTEM, USER, 50))
    assert "no price is configured" in message and "LLM_PRICE_TABLE" in message
    assert not (tmp_path / "spend.sqlite3").exists()


@pytest.mark.parametrize("model", [None, "", "   "])
def test_no_model_at_all_is_refused(tmp_path, model):
    refused(lambda: ledger(tmp_path, model=model).reserve(SYSTEM, USER, 50))


def test_an_empty_price_table_refuses_everything(tmp_path):
    refused(lambda: ledger(tmp_path, prices={}).reserve(SYSTEM, USER, 50))


def test_the_model_name_is_matched_exactly_apart_from_surrounding_spaces(tmp_path):
    ledger(tmp_path, model=f"  {MODEL}  ").reserve(SYSTEM, USER, 50)            # trimmed, like the adapter does
    refused(lambda: ledger(tmp_path, model=MODEL.upper()).reserve(SYSTEM, USER, 50))
    refused(lambda: ledger(tmp_path, model=MODEL + "-2").reserve(SYSTEM, USER, 50))


def test_a_database_path_that_cannot_be_created_is_refused(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, where a directory is needed")
    led = SpendLedger(blocker / "spend.sqlite3", 1.0, MODEL, {MODEL: PRICE}, clock=Clock())
    message = refused(lambda: led.reserve(SYSTEM, USER, 50))
    assert "ledger cannot be used" in message


def test_a_database_path_that_is_a_directory_is_refused(tmp_path):
    (tmp_path / "spend.sqlite3").mkdir()
    refused(lambda: ledger(tmp_path).reserve(SYSTEM, USER, 50))


def test_a_file_that_is_not_a_database_is_refused_and_left_untouched(tmp_path):
    path = tmp_path / "spend.sqlite3"
    path.write_bytes(b"this is not a sqlite database at all" * 20)
    before = path.read_bytes()
    refused(lambda: ledger(tmp_path).reserve(SYSTEM, USER, 50))
    assert path.read_bytes() == before


def test_a_path_with_a_null_byte_is_refused(tmp_path):
    led = SpendLedger(Path(str(tmp_path / "spend") + "\x00.sqlite3"), 1.0, MODEL, {MODEL: PRICE}, clock=Clock())
    refused(lambda: led.reserve(SYSTEM, USER, 50))


def test_a_write_that_fails_leaves_the_total_unchanged_and_refuses_the_call(tmp_path):
    # A trigger makes every update fail, which simulates a database that can be read but not written.
    ledger(tmp_path).reserve(SYSTEM, USER, 50)
    with sqlite3.connect(tmp_path / "spend.sqlite3") as db:
        db.execute("CREATE TRIGGER no_writes BEFORE UPDATE ON spend BEGIN SELECT RAISE(ABORT, 'read only'); END")
        db.execute("CREATE TRIGGER no_inserts BEFORE INSERT ON spend BEGIN SELECT RAISE(ABORT, 'read only'); END")
    refused(lambda: ledger(tmp_path).reserve(SYSTEM, USER, 50))
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": RESERVED}


def test_a_database_locked_by_another_writer_is_refused_after_the_wait(tmp_path, monkeypatch):
    monkeypatch.setattr("askact.guards.spend.BUSY_TIMEOUT_SECONDS", 0.05)
    ledger(tmp_path).reserve(SYSTEM, USER, 50)
    blocker = sqlite3.connect(tmp_path / "spend.sqlite3", isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")          # holds the write lock
    try:
        refused(lambda: ledger(tmp_path).reserve(SYSTEM, USER, 50))
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    ledger(tmp_path).reserve(SYSTEM, USER, 50)  # and it recovers once the lock is released
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": 2 * RESERVED}


@pytest.mark.parametrize("stored", ["lots", -5, 1.5, None])
def test_a_stored_total_that_is_not_a_non_negative_integer_is_refused_not_guessed(tmp_path, stored):
    path = tmp_path / "spend.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE spend (day TEXT PRIMARY KEY, micro_usd)")   # no type or NOT NULL: allows any value
        db.execute("INSERT INTO spend VALUES ('2026-10-12', ?)", (stored,))
    refused(lambda: ledger(tmp_path).reserve(SYSTEM, USER, 50))


def test_a_table_with_the_wrong_columns_is_refused(tmp_path):
    with sqlite3.connect(tmp_path / "spend.sqlite3") as db:
        db.execute("CREATE TABLE spend (something TEXT)")
    refused(lambda: ledger(tmp_path).reserve(SYSTEM, USER, 50))


def test_a_total_too_large_for_sqlite_is_refused(tmp_path):
    path = tmp_path / "spend.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE spend (day TEXT PRIMARY KEY, micro_usd INTEGER NOT NULL)")
        db.execute("INSERT INTO spend VALUES ('2026-10-12', ?)", (2**63 - 1,))   # the largest value SQLite can store
    # The cap is high enough that the cap check would pass, so only the overflow can stop this call.
    refused(lambda: ledger(tmp_path, cap_usd=1e30).reserve(SYSTEM, USER, 50))
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT micro_usd FROM spend").fetchone() == (2**63 - 1,)   # untouched


@pytest.mark.parametrize("bad_price", [
    ModelPrice.model_construct(input_per_mtok=float("inf"), output_per_mtok=1.0),
    ModelPrice.model_construct(input_per_mtok=1.0, output_per_mtok=float("nan")),
])
def test_a_non_finite_price_refuses_the_call(tmp_path, bad_price):
    refused(lambda: ledger(tmp_path, prices={MODEL: bad_price}).reserve(SYSTEM, USER, 50))


@pytest.mark.parametrize("bad_cap", [float("inf"), float("nan")])
def test_a_non_finite_cap_refuses_the_call_instead_of_allowing_everything(tmp_path, bad_cap):
    refused(lambda: ledger(tmp_path, cap_usd=bad_cap).reserve(SYSTEM, USER, 50))


def test_the_refusal_does_not_chain_the_underlying_error_or_repeat_its_text(tmp_path):
    (tmp_path / "spend.sqlite3").mkdir()
    with pytest.raises(SpendRefusedError) as info:
        ledger(tmp_path).reserve(SYSTEM, USER, 50)
    error = info.value
    assert error.__cause__ is None and (error.__context__ is None or error.__suppress_context__)
    assert "unable to open" not in str(error) and "sqlite" not in str(error).lower()


def test_a_missing_price_is_logged_so_the_operator_can_find_out_why(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="askact.guards.spend"):
        refused(lambda: ledger(tmp_path, model="other-model").reserve(SYSTEM, USER, 50))
    assert "no entry in LLM_PRICE_TABLE" in caplog.text and "other-model" not in caplog.text


def test_a_failed_write_closes_the_connection_and_leaves_no_transaction_open(tmp_path, monkeypatch):
    opened = []
    real_connect = sqlite3.connect

    def spy(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr("askact.guards.spend.sqlite3.connect", spy)
    ledger(tmp_path).reserve(SYSTEM, USER, 50)
    with sqlite3.connect(tmp_path / "spend.sqlite3") as db:
        db.execute("CREATE TRIGGER no_writes BEFORE UPDATE ON spend BEGIN SELECT RAISE(ABORT, 'read only'); END")
        db.execute("CREATE TRIGGER no_inserts BEFORE INSERT ON spend BEGIN SELECT RAISE(ABORT, 'read only'); END")
    opened.clear()
    refused(lambda: ledger(tmp_path).reserve(SYSTEM, USER, 50))
    [connection] = opened
    with pytest.raises(sqlite3.ProgrammingError):    # "Cannot operate on a closed database"
        connection.execute("SELECT 1")


def test_a_refused_reservation_closes_its_connection_too(tmp_path, monkeypatch):
    opened = []
    real_connect = sqlite3.connect
    monkeypatch.setattr("askact.guards.spend.sqlite3.connect", lambda *a, **k: opened.append(real_connect(*a, **k)) or opened[-1])
    led = ledger(tmp_path, cap_usd=RESERVED / 1_000_000)
    led.reserve(SYSTEM, USER, 50)
    refused(lambda: led.reserve(SYSTEM, USER, 50))
    assert len(opened) == 2
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


def test_a_failure_part_way_leaves_the_transaction_rolled_back_not_half_committed(tmp_path):
    # The table is created inside the transaction; a refusal must undo that too, or a half-made database is left.
    led = ledger(tmp_path, cap_usd=(RESERVED - 1) / 1_000_000)
    refused(lambda: led.reserve(SYSTEM, USER, 50))
    with sqlite3.connect(tmp_path / "spend.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM sqlite_master").fetchone() == (0,)


def test_a_failure_while_reserving_is_logged_by_type_only_and_the_message_has_no_path(tmp_path, caplog):
    (tmp_path / "spend.sqlite3").mkdir()
    with caplog.at_level(logging.WARNING, logger="askact.guards.spend"):
        message = refused(lambda: ledger(tmp_path).reserve(SYSTEM, USER, 50))
    assert str(tmp_path) not in message
    assert str(tmp_path) not in caplog.text and "OperationalError" in caplog.text


def test_refusals_do_not_leave_a_database_connection_or_lock_behind(tmp_path):
    cap = RESERVED / 1_000_000
    led = ledger(tmp_path, cap_usd=cap)
    led.reserve(SYSTEM, USER, 50)
    for _ in range(50):
        refused(lambda: led.reserve(SYSTEM, USER, 50))
    other = sqlite3.connect(tmp_path / "spend.sqlite3", isolation_level=None, timeout=0.1)
    other.execute("BEGIN IMMEDIATE")      # would fail if a refused call had left the write lock held
    other.execute("ROLLBACK")
    other.close()


def test_the_text_of_a_ledger_error_never_reaches_the_refusal_or_the_log(tmp_path, caplog):
    ledger(tmp_path).reserve(SYSTEM, USER, 50)
    with sqlite3.connect(tmp_path / "spend.sqlite3") as db:
        # INSERT OR REPLACE is a delete plus an insert, so it is an insert trigger that fires, not an update one
        db.execute("CREATE TRIGGER t BEFORE INSERT ON spend BEGIN SELECT RAISE(ABORT, 'DISTINCTIVE-DB-MESSAGE'); END")
    with caplog.at_level(logging.DEBUG, logger="askact.guards.spend"):
        message = refused(lambda: ledger(tmp_path).reserve(SYSTEM, USER, 50))
    assert "DISTINCTIVE-DB-MESSAGE" not in message and "DISTINCTIVE-DB-MESSAGE" not in caplog.text
    assert "IntegrityError" in caplog.text    # but the kind of failure is there for the operator


def test_a_failed_reconciliation_leaves_the_reservation_standing_and_does_not_raise(tmp_path, caplog):
    led = ledger(tmp_path)
    reservation = led.reserve(SYSTEM, USER, 50)
    with sqlite3.connect(tmp_path / "spend.sqlite3") as db:
        db.execute("CREATE TRIGGER no_writes BEFORE INSERT ON spend BEGIN SELECT RAISE(ABORT, 'read only'); END")
    with caplog.at_level(logging.WARNING, logger="askact.guards.spend"):
        reservation.reconcile(10, 10)
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": RESERVED}
    assert "reservation stands" in caplog.text and "read only" not in caplog.text


# =====================================================================================================
# concurrency (R6.7: "within a single SQLite transaction")
# =====================================================================================================

def run_in_threads(count: int, work) -> list:
    barrier = threading.Barrier(count)
    results = [None] * count

    def worker(i: int):
        barrier.wait()   # release every thread at once
        results[i] = work(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def reserve_outcome(led: SpendLedger) -> str:
    try:
        led.reserve(SYSTEM, USER, 50)
        return "reserved"
    except SpendRefusedError:
        return "refused"


@pytest.mark.parametrize("attempt", range(5))
def test_two_concurrent_reservations_cannot_both_pass_when_only_one_fits(tmp_path, attempt):
    led = ledger(tmp_path, cap_usd=RESERVED / 1_000_000, name=f"spend{attempt}.sqlite3")
    outcomes = run_in_threads(2, lambda i: reserve_outcome(led))
    assert sorted(outcomes) == ["refused", "reserved"]
    assert rows(tmp_path / f"spend{attempt}.sqlite3") == {"2026-10-12": RESERVED}


def test_many_concurrent_reservations_never_exceed_the_cap(tmp_path):
    led = ledger(tmp_path, cap_usd=5 * RESERVED / 1_000_000)
    outcomes = run_in_threads(40, lambda i: reserve_outcome(led))
    assert outcomes.count("reserved") == 5 and outcomes.count("refused") == 35
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": 5 * RESERVED}


def test_concurrent_reservations_and_reconciliations_keep_the_total_exact(tmp_path):
    led = ledger(tmp_path, cap_usd=1.0)

    def work(i: int) -> None:
        for _ in range(5):
            led.reserve(SYSTEM, USER, 50).reconcile(10, 10)    # each call ends up costing exactly 50

    run_in_threads(8, work)
    assert rows(tmp_path / "spend.sqlite3") == {"2026-10-12": 8 * 5 * 50}


def test_separate_ledger_objects_on_one_file_share_the_cap(tmp_path):
    # As two workers (or a restarted worker overlapping the old one) would.
    cap = 3 * RESERVED / 1_000_000
    ledgers = [ledger(tmp_path, cap_usd=cap) for _ in range(6)]
    outcomes = run_in_threads(6, lambda i: reserve_outcome(ledgers[i]))
    assert outcomes.count("reserved") == 3


# =====================================================================================================
# settings and the full path through answer()
# =====================================================================================================

def test_the_ledger_is_built_from_settings(tmp_path):
    settings = Settings(_env_file=None, spend_db_path=tmp_path / "s.sqlite3", daily_spend_cap_usd=0.5, llm_model=MODEL,
                        llm_price_table={MODEL: PRICE})
    SpendLedger.from_settings(settings, clock=Clock()).reserve(SYSTEM, USER, 50)
    assert rows(tmp_path / "s.sqlite3") == {"2026-10-12": RESERVED}
    refused(lambda: SpendLedger.from_settings(settings.model_copy(update={"daily_spend_cap_usd": 0.0001}), clock=Clock()).reserve(SYSTEM, USER, 5000))


def test_a_price_table_from_the_environment_is_used(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", MODEL)
    monkeypatch.setenv("LLM_PRICE_TABLE", json.dumps({MODEL: {"input_per_mtok": 1.0, "output_per_mtok": 4.0}}))
    monkeypatch.setenv("SPEND_DB_PATH", str(tmp_path / "env.sqlite3"))
    SpendLedger.from_settings(Settings(_env_file=None), clock=Clock()).reserve(SYSTEM, USER, 50)
    assert rows(tmp_path / "env.sqlite3") == {"2026-10-12": RESERVED}


@pytest.mark.parametrize("env", [
    {"DAILY_SPEND_CAP_USD": "inf"}, {"DAILY_SPEND_CAP_USD": "1e999"}, {"DAILY_SPEND_CAP_USD": "nan"},
    {"LLM_PRICE_TABLE": '{"m": {"input_per_mtok": 1e999, "output_per_mtok": 1}}'},
    {"LLM_PRICE_TABLE": '{"m": {"input_per_mtok": 1, "output_per_mtok": Infinity}}'},
    {"LLM_PRICE_TABLE": "{not json"},
    {"LLM_PRICE_TABLE": '{"m": {"input_per_mtok": -1, "output_per_mtok": 1}}'},
    {"LLM_PRICE_TABLE": '{"m": {"input_per_mtok": 1}}'},
])
def test_a_bad_cap_or_price_table_stops_the_settings_from_loading_so_no_paid_call_can_start(monkeypatch, env):
    # "An unparsable table refuses": the app cannot even build its settings, so there is nothing to call with.
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises((ValidationError, SettingsError)):
        Settings(_env_file=None)


def scored(section: str, text: str) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(section_id=SectionId.parse(section), ordinal=1, title="T", text=text), score=0.9)


RETRIEVED = RetrievalResult(chunks=[scored("article:5", "passage text " * 20)], relevance=0.9, relevance_kind="cosine")
GOOD_REPLY = json.dumps({"answer": "An answer.", "citations": ["article:5"], "covered": True})


def pipeline_settings(tmp_path, cap: float, max_output_tokens: int = 200) -> Settings:
    return Settings(_env_file=None, spend_db_path=tmp_path / "s.sqlite3", daily_spend_cap_usd=cap, llm_model=MODEL,
                    llm_price_table={MODEL: PRICE}, relevance_threshold_cosine=0.5, max_output_tokens=max_output_tokens)


def test_answer_charges_the_real_cost_of_the_call_through_the_ledger(tmp_path):
    settings = pipeline_settings(tmp_path, cap=1.0)
    llm = FakeLLM([LLMReply(GOOD_REPLY, input_tokens=400, output_tokens=30)])
    result = answer("What is banned?", RETRIEVED, settings, llm, SpendLedger.from_settings(settings, clock=Clock()))
    assert result.status == "answered"
    assert rows(tmp_path / "s.sqlite3") == {"2026-10-12": 400 * 1 + 30 * 4}


def test_answer_with_no_reported_usage_leaves_the_worst_case_reservation_as_the_charge(tmp_path):
    settings = pipeline_settings(tmp_path, cap=1.0, max_output_tokens=200)
    result = answer("What is banned?", RETRIEVED, settings, FakeLLM([GOOD_REPLY]), SpendLedger.from_settings(settings, clock=Clock()))
    assert result.status == "answered"
    [spent] = rows(tmp_path / "s.sqlite3").values()
    assert spent >= 200 * 4   # at least the full output cap was charged


def test_answer_makes_no_llm_call_when_the_ledger_refuses(tmp_path):
    settings = pipeline_settings(tmp_path, cap=0.000001)    # 1 micro-dollar
    llm = FakeLLM([GOOD_REPLY])
    with pytest.raises(SpendRefusedError):
        answer("What is banned?", RETRIEVED, settings, llm, SpendLedger.from_settings(settings, clock=Clock()))
    assert llm.call_count == 0


def test_answer_makes_no_llm_call_for_a_model_without_a_price(tmp_path):
    settings = pipeline_settings(tmp_path, cap=1.0).model_copy(update={"llm_model": "unpriced-model"})
    llm = FakeLLM([GOOD_REPLY])
    with pytest.raises(SpendRefusedError):
        answer("What is banned?", RETRIEVED, settings, llm, SpendLedger.from_settings(settings, clock=Clock()))
    assert llm.call_count == 0


def test_answer_after_a_gate_refusal_spends_nothing(tmp_path):
    settings = pipeline_settings(tmp_path, cap=1.0).model_copy(update={"relevance_threshold_cosine": 0.99})
    llm = FakeLLM([GOOD_REPLY])
    assert answer("What is banned?", RETRIEVED, settings, llm, SpendLedger.from_settings(settings, clock=Clock())).status == "not_covered"
    assert llm.call_count == 0 and not (tmp_path / "s.sqlite3").exists()


def test_a_run_of_answers_is_stopped_by_the_cap_and_each_call_is_charged_its_real_cost(tmp_path):
    # One reservation for this prompt is 671 micro-dollars (471 estimated input tokens + 50 output tokens at 4 each);
    # each call really costs 180 (100 in + 20 out). With a cap of 1000:
    #   call 1: 0 + 671 = 671 fits; reconciled down to 180.
    #   call 2: 180 + 671 = 851 fits; reconciled down to 360.
    #   call 3: 360 + 671 = 1031 does not fit, although the real cost (180) would: the guard is conservative.
    settings = pipeline_settings(tmp_path, cap=0.001, max_output_tokens=50)
    ledger_ = SpendLedger.from_settings(settings, clock=Clock())
    llm = FakeLLM([LLMReply(GOOD_REPLY, 100, 20)] * 50)
    answered = 0
    with pytest.raises(SpendRefusedError):
        for _ in range(50):
            answer("What is banned?", RETRIEVED, settings, llm, ledger_)
            answered += 1
    assert answered == 2 and llm.call_count == 2
    assert rows(tmp_path / "s.sqlite3") == {"2026-10-12": 2 * 180}   # the refused third call left no trace
