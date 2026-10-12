"""The daily spend cap: a SQLite ledger that fails closed (R6.6 to R6.10).

How it works, in the order a paid call meets it:

  reserve   BEFORE the call. Work out the worst-case cost of the call and set it aside against today's total, in
            one SQLite transaction that also checks the cap. If it would not fit, the call is not made.
  reconcile AFTER the call. Replace the reserved worst case with the actual cost from the reported token usage.

Everything fails closed: if the ledger cannot be read or written, the model has no price, or anything unexpected
goes wrong, the call is refused (`SpendRefusedError`) and nothing is spent. The reservation is committed to disk
before the call is made, so if the process dies, the call never reports usage, or reconciling fails, the
reservation simply stands as the amount charged (R6.9).

Money is stored as integer micro-dollars. A price of $P per million tokens is exactly P micro-dollars per token,
so cost = tokens x price, rounded UP. The arithmetic is done in `Decimal`: with floats, 3 x 0.1 is
0.30000000000000004, which would round up to the wrong number.

Each operation opens its own connection. SQLite connections belong to the thread that made them and the API runs
sync endpoints in a thread pool, and `BEGIN IMMEDIATE` takes the database write lock, so concurrent reservations
queue up and the cap check cannot be raced.
"""

import logging
import math
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from askact.config import ModelPrice, Settings
from askact.generation.answer import SpendRefusedError, SpendReservation

log = logging.getLogger(__name__)

MICRO = Decimal(1_000_000)  # micro-dollars per dollar
BUSY_TIMEOUT_SECONDS = 5.0  # how long to wait for another writer; past that the call is refused


def estimate_input_tokens(system: str, user: str) -> int:
    """A deliberate over-estimate of the prompt's token count: UTF-8 bytes divided by 3.

    English averages about 4 characters (bytes) per token, so this reserves roughly a third too much, and the
    difference is handed back when the call is reconciled. Using bytes instead of characters keeps ASCII text the
    same but reserves more for text where one character costs a token or more (Chinese, Japanese, emoji). That
    only matters for the question, which is the one part of the prompt a user writes (at most MAX_QUESTION_CHARS);
    the passages are the Act's English text. It is an estimate, not a bound: a pathological question could still
    cost more than this, and the overshoot is then the difference between the estimate and the real count.
    """
    return math.ceil(len((system + user).encode("utf-8")) / 3)


def _cost_micro(price: ModelPrice, input_tokens: int, output_tokens: int) -> int:
    """Exact cost in micro-dollars, rounded up. Bad inputs are errors rather than numbers.

    Token counts must be non-negative integers: a negative one would make the cost negative, and a negative
    reconciliation would then act as a credit against the day's total. A non-finite price is refused too.
    """
    for count in (input_tokens, output_tokens):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("a token count must be a non-negative integer")
    # repr() gives the shortest string that round-trips, so Decimal sees 0.15 and not 0.1499999999999999944...
    cost = Decimal(repr(price.input_per_mtok)) * input_tokens + Decimal(repr(price.output_per_mtok)) * output_tokens
    # int() of an infinite or NaN Decimal raises (OverflowError / ValueError), so a non-finite price cannot become
    # a cost: it fails here, and `reserve` turns any failure into a refusal. Settings refuses such prices earlier.
    return int(cost.to_integral_value(rounding="ROUND_CEILING"))


def _cap_micro(cap_usd: float) -> int:
    """The cap in micro-dollars, rounded DOWN: a fraction of a micro-dollar is not spendable.

    int() truncates toward zero, which is rounding down for a cap that is never negative. An infinite or NaN cap
    makes int() raise, which `reserve` turns into a refusal: a bad cap can never mean no cap.
    """
    return int(Decimal(repr(cap_usd)) * MICRO)


class SpendLedger:
    """Reserve and reconcile spend against a per-day total stored in SQLite. Implements `SpendGuard`."""

    def __init__(
        self,
        db_path: Path,
        cap_usd: float,
        model: str | None,
        prices: dict[str, ModelPrice],
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self._db_path = db_path
        self._cap_usd = cap_usd
        self._model = (model or "").strip()
        self._prices = prices
        self._clock = clock

    @classmethod
    def from_settings(cls, settings: Settings, **options) -> "SpendLedger":
        return cls(settings.spend_db_path, settings.daily_spend_cap_usd, settings.llm_model, settings.llm_price_table, **options)

    # --- reserve ----------------------------------------------------------------------------------

    def reserve(self, system: str, user: str, max_output_tokens: int) -> SpendReservation:
        """Set aside the worst-case cost of one call, or raise SpendRefusedError (R6.7 to R6.9)."""
        try:
            price = self._prices.get(self._model)
            if price is None:
                # Not an exception from the ledger: this is a configuration the operator has to fix.
                log.warning("spend refused: the model has no entry in LLM_PRICE_TABLE")
                raise SpendRefusedError(
                    "no price is configured for the model (LLM_MODEL and LLM_PRICE_TABLE), so paid calls are refused"
                )
            amount = _cost_micro(price, estimate_input_tokens(system, user), max_output_tokens)
            cap = _cap_micro(self._cap_usd)
            day = self._today()
            with self._transaction() as db:
                total = self._total(db, day)
                if total + amount > cap:
                    raise SpendRefusedError("the daily spend cap has been reached: try again tomorrow (UTC)")
                db.execute("INSERT OR REPLACE INTO spend (day, micro_usd) VALUES (?, ?)", (day, total + amount))
        except SpendRefusedError:
            raise
        except Exception as exc:  # any other failure: fail closed, and say only what kind it was
            log.warning("spend refused: the ledger could not be used (%s)", type(exc).__name__)
            raise SpendRefusedError("the spend ledger cannot be used, so paid calls are refused") from None
        return _Reservation(self, day, amount, price)

    # --- reconcile --------------------------------------------------------------------------------

    def _reconcile(self, day: str, reserved: int, price: ModelPrice, input_tokens: int | None, output_tokens: int | None) -> None:
        if input_tokens is None or output_tokens is None:
            return  # usage not reported (or only half of it): the reservation stands as the charge (R6.9)
        try:
            actual = _cost_micro(price, input_tokens, output_tokens)
            with self._transaction() as db:
                total = self._total(db, day)
                # max(..., 0): if the day's row was reset or lost since the reservation, do not go negative.
                db.execute("INSERT OR REPLACE INTO spend (day, micro_usd) VALUES (?, ?)", (day, max(total - reserved, 0) + actual))
        except Exception as exc:  # the reservation, already on disk, simply stands
            log.warning("spend reconciliation failed (%s); the reservation stands as the charge", type(exc).__name__)

    # --- storage ----------------------------------------------------------------------------------

    def _today(self) -> str:
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("the clock must return a timezone-aware time")
        return now.astimezone(UTC).strftime("%Y-%m-%d")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """A write transaction on a fresh connection: `BEGIN IMMEDIATE` takes the write lock before anything is read."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self._db_path, timeout=BUSY_TIMEOUT_SECONDS, isolation_level=None)  # None: we control BEGIN/COMMIT
        try:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS spend (day TEXT PRIMARY KEY, micro_usd INTEGER NOT NULL)")
            yield db
            db.execute("COMMIT")
        finally:
            # Closing a connection discards a transaction that was not committed, so on any failure (a refusal, an
            # error, the write lock timing out) nothing is kept, including the table created above.
            db.close()

    @staticmethod
    def _total(db: sqlite3.Connection, day: str) -> int:
        row = db.execute("SELECT micro_usd FROM spend WHERE day = ?", (day,)).fetchone()
        if row is None:
            return 0  # a new day starts at zero (R6.10)
        value = row[0]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("the stored total is not a non-negative integer")  # corrupt: refuse rather than guess
        return value


class _Reservation:
    """What `reserve` hands back. Remembers its own day, so a call that straddles midnight is settled on the day it
    was reserved. Settling twice does nothing the second time."""

    def __init__(self, ledger: SpendLedger, day: str, reserved: int, price: ModelPrice):
        self._ledger = ledger
        self._day = day
        self._reserved = reserved
        self._price = price
        self._settled = False

    def reconcile(self, input_tokens: int | None, output_tokens: int | None) -> None:
        if self._settled:
            return
        self._settled = True
        self._ledger._reconcile(self._day, self._reserved, self._price, input_tokens, output_tokens)
