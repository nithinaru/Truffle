"""Append-only, exactly reducible accounting for the $100 shadow account."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Self

from pydantic import Field, field_validator, model_validator

from paper.models import (
    MarketSnapshot,
    NonNegativeDerivedDecimal,
    PaperModel,
    PositiveDecimal,
    ShadowBatchExecuted,
    UtcDatetime,
    decimal_context,
    freeze_decimal_map,
)

SHADOW_SEED_CASH = Decimal("100.00")


class PaperLedgerError(Exception):
    """Base class for deterministic paper-accounting failures."""


class LedgerSequenceError(PaperLedgerError):
    """Raised when an event does not follow the append-only sequence."""


class LedgerIdentifierCollisionError(PaperLedgerError):
    """Raised when an existing event/order/fill ID is reused for new data."""


class LedgerAccountingError(PaperLedgerError):
    """Raised when an atomic event would create impossible long-only state."""


class ShadowMarkRecorded(PaperModel):
    """One market snapshot appended to the exact account event stream."""

    sequence: int = Field(ge=1)
    snapshot: MarketSnapshot

    @property
    def snapshot_id(self) -> str:
        return self.snapshot.snapshot_id


type ShadowLedgerEvent = ShadowBatchExecuted | ShadowMarkRecorded


class ShadowLedgerSnapshot(PaperModel):
    """Fully reduced exact state after a prefix of the event stream."""

    sequence: int = Field(ge=0)
    as_of: UtcDatetime | None = None
    cash: NonNegativeDerivedDecimal
    positions: dict[str, NonNegativeDerivedDecimal]
    marks: dict[str, PositiveDecimal]
    cumulative_fees: NonNegativeDerivedDecimal
    executed_batch_ids: tuple[str, ...] = ()
    executed_order_ids: tuple[str, ...] = ()
    executed_fill_ids: tuple[str, ...] = ()
    recorded_snapshot_ids: tuple[str, ...] = ()
    equity: NonNegativeDerivedDecimal
    gross_exposure: NonNegativeDerivedDecimal

    @field_validator("positions", "marks", mode="after")
    @classmethod
    def _ordered_decimal_maps(cls, value: dict[str, Decimal]) -> dict[str, Decimal]:
        return freeze_decimal_map(value)

    @model_validator(mode="after")
    def _exact_valuation(self) -> Self:
        missing = sorted(set(self.positions) - set(self.marks))
        if missing:
            raise ValueError(f"positions are missing exact marks: {missing}")
        with decimal_context():
            expected_equity = self.cash + sum(
                (quantity * self.marks[ticker] for ticker, quantity in self.positions.items()),
                Decimal("0"),
            )
            expected_gross = sum(
                (
                    abs(quantity * self.marks[ticker])
                    for ticker, quantity in self.positions.items()
                ),
                Decimal("0"),
            )
        if self.equity != expected_equity:
            raise ValueError("equity must equal exact marked cash plus positions")
        if self.gross_exposure != expected_gross:
            raise ValueError("gross exposure must equal exact absolute marked positions")
        return self


def _snapshot(
    *,
    sequence: int,
    as_of: datetime | None,
    cash: Decimal,
    positions: dict[str, Decimal],
    marks: dict[str, Decimal],
    cumulative_fees: Decimal,
    executed_batch_ids: list[str],
    executed_order_ids: list[str],
    executed_fill_ids: list[str],
    recorded_snapshot_ids: list[str],
) -> ShadowLedgerSnapshot:
    with decimal_context():
        active_positions = {
            ticker: quantity
            for ticker, quantity in sorted(positions.items())
            if quantity != Decimal("0")
        }
        ordered_marks = dict(sorted(marks.items()))
        equity = cash + sum(
            (quantity * ordered_marks[ticker] for ticker, quantity in active_positions.items()),
            Decimal("0"),
        )
        gross = sum(
            (
                abs(quantity * ordered_marks[ticker])
                for ticker, quantity in active_positions.items()
            ),
            Decimal("0"),
        )
        return ShadowLedgerSnapshot(
            sequence=sequence,
            as_of=as_of,
            cash=cash,
            positions=active_positions,
            marks=ordered_marks,
            cumulative_fees=cumulative_fees,
            executed_batch_ids=tuple(executed_batch_ids),
            executed_order_ids=tuple(executed_order_ids),
            executed_fill_ids=tuple(executed_fill_ids),
            recorded_snapshot_ids=tuple(recorded_snapshot_ids),
            equity=equity,
            gross_exposure=gross,
        )


def _reduce_shadow_ledger(
    events: tuple[ShadowLedgerEvent, ...] | list[ShadowLedgerEvent],
) -> ShadowLedgerSnapshot:
    """Replay an event prefix into exact state, rejecting any invalid suffix.

    A repeated, byte-identical batch or mark is idempotent.  Reusing its ID
    with different content is an explicit collision, never an overwrite.
    """

    sequence = 0
    as_of: datetime | None = None
    cash = SHADOW_SEED_CASH
    positions: dict[str, Decimal] = {}
    marks: dict[str, Decimal] = {}
    cumulative_fees = Decimal("0")
    batches: dict[str, ShadowBatchExecuted] = {}
    mark_events: dict[str, ShadowMarkRecorded] = {}
    snapshot_contents: dict[str, MarketSnapshot] = {}
    order_ids: set[str] = set()
    fill_ids: set[str] = set()
    ordered_batch_ids: list[str] = []
    ordered_order_ids: list[str] = []
    ordered_fill_ids: list[str] = []
    ordered_snapshot_ids: list[str] = []

    for event in events:
        if isinstance(event, ShadowBatchExecuted):
            prior_batch = batches.get(event.batch_id)
            if prior_batch is not None:
                if prior_batch.canonical_json() == event.canonical_json():
                    continue
                raise LedgerIdentifierCollisionError(
                    f"batch ID {event.batch_id!r} is already bound to a different event"
                )
        else:
            prior_mark = mark_events.get(event.snapshot_id)
            if prior_mark is not None:
                if prior_mark.canonical_json() == event.canonical_json():
                    continue
                raise LedgerIdentifierCollisionError(
                    f"snapshot ID {event.snapshot_id!r} is already bound to a different event"
                )

        event_snapshot = event.snapshot
        prior_snapshot = snapshot_contents.get(event_snapshot.snapshot_id)
        if (
            prior_snapshot is not None
            and prior_snapshot.canonical_json() != event_snapshot.canonical_json()
        ):
            raise LedgerIdentifierCollisionError(
                f"snapshot ID {event_snapshot.snapshot_id!r} is already bound to "
                "different market data"
            )

        expected_sequence = sequence + 1
        if event.sequence != expected_sequence:
            raise LedgerSequenceError(
                f"expected event sequence {expected_sequence}, received {event.sequence}"
            )

        event_time = event_snapshot.as_of
        if as_of is not None and event_time < as_of:
            raise LedgerSequenceError("event timestamps must be non-decreasing")

        if isinstance(event, ShadowBatchExecuted):
            reused_orders = sorted(order_ids.intersection(event.order_ids))
            if reused_orders:
                raise LedgerIdentifierCollisionError(
                    f"order IDs are already executed: {reused_orders}"
                )
            event_fill_ids = tuple(fill.fill_id for fill in event.fills)
            reused_fills = sorted(fill_ids.intersection(event_fill_ids))
            if reused_fills:
                raise LedgerIdentifierCollisionError(
                    f"fill IDs are already executed: {reused_fills}"
                )

            candidate_cash = cash
            candidate_positions = dict(positions)
            candidate_marks = dict(marks)
            incoming_marks = event.snapshot.marks()
            missing_pretrade_marks = sorted(
                ticker
                for ticker, quantity in candidate_positions.items()
                if quantity != Decimal("0") and ticker not in incoming_marks
            )
            if missing_pretrade_marks:
                raise LedgerAccountingError(
                    "execution snapshot is missing held positions: "
                    f"{missing_pretrade_marks}"
                )
            candidate_marks.update(incoming_marks)
            batch_fees = Decimal("0")
            for fill in event.fills:
                current_quantity = candidate_positions.get(fill.ticker, Decimal("0"))
                if fill.side == "buy":
                    candidate_cash -= fill.notional + fill.fee
                    candidate_positions[fill.ticker] = current_quantity + fill.quantity
                else:
                    candidate_cash += fill.notional - fill.fee
                    candidate_positions[fill.ticker] = current_quantity - fill.quantity
                batch_fees += fill.fee

            negative_positions = sorted(
                ticker
                for ticker, quantity in candidate_positions.items()
                if quantity < Decimal("0")
            )
            if negative_positions:
                raise LedgerAccountingError(
                    f"batch would create short positions: {negative_positions}"
                )
            if candidate_cash < Decimal("0"):
                raise LedgerAccountingError(
                    f"batch would create negative cash ({format(candidate_cash, 'f')})"
                )
            missing_posttrade_marks = sorted(
                ticker
                for ticker, quantity in candidate_positions.items()
                if quantity != Decimal("0") and ticker not in incoming_marks
            )
            if missing_posttrade_marks:
                raise LedgerAccountingError(
                    "execution snapshot is missing post-trade positions: "
                    f"{missing_posttrade_marks}"
                )

            cash = candidate_cash
            positions = candidate_positions
            marks = candidate_marks
            cumulative_fees += batch_fees
            batches[event.batch_id] = event
            order_ids.update(event.order_ids)
            fill_ids.update(event_fill_ids)
            ordered_batch_ids.append(event.batch_id)
            ordered_order_ids.extend(event.order_ids)
            ordered_fill_ids.extend(event_fill_ids)
        else:
            incoming_marks = event.snapshot.marks()
            missing_positions = sorted(
                ticker
                for ticker, quantity in positions.items()
                if quantity != Decimal("0") and ticker not in incoming_marks
            )
            if missing_positions:
                raise LedgerAccountingError(
                    "mark snapshot is missing held positions: "
                    f"{missing_positions}"
                )
            marks.update(incoming_marks)
            mark_events[event.snapshot_id] = event

        if prior_snapshot is None:
            snapshot_contents[event_snapshot.snapshot_id] = event_snapshot
            ordered_snapshot_ids.append(event_snapshot.snapshot_id)

        sequence = event.sequence
        as_of = event_time

    return _snapshot(
        sequence=sequence,
        as_of=as_of,
        cash=cash,
        positions=positions,
        marks=marks,
        cumulative_fees=cumulative_fees,
        executed_batch_ids=ordered_batch_ids,
        executed_order_ids=ordered_order_ids,
        executed_fill_ids=ordered_fill_ids,
        recorded_snapshot_ids=ordered_snapshot_ids,
    )


def reduce_shadow_ledger(
    events: tuple[ShadowLedgerEvent, ...] | list[ShadowLedgerEvent],
) -> ShadowLedgerSnapshot:
    """Replay events under Truffle's isolated fixed Decimal context."""

    with decimal_context():
        return _reduce_shadow_ledger(events)


class ShadowLedger:
    """Small append-only wrapper around :func:`reduce_shadow_ledger`."""

    def __init__(self) -> None:
        self._events: list[ShadowLedgerEvent] = []

    @property
    def events(self) -> tuple[ShadowLedgerEvent, ...]:
        return tuple(self._events)

    @property
    def state(self) -> ShadowLedgerSnapshot:
        return reduce_shadow_ledger(self._events)

    def append(self, event: ShadowLedgerEvent) -> ShadowLedgerSnapshot:
        """Validate and atomically append an event; identical retries are no-ops."""

        candidate = reduce_shadow_ledger([*self._events, event])
        if not any(
            type(existing) is type(event)
            and existing.canonical_json() == event.canonical_json()
            for existing in self._events
        ):
            self._events.append(event)
        return candidate

    def execute(self, event: ShadowBatchExecuted) -> ShadowLedgerSnapshot:
        return self.append(event)

    def mark(
        self,
        snapshot: MarketSnapshot,
        *,
        sequence: int | None = None,
    ) -> ShadowLedgerSnapshot:
        next_sequence = self.state.sequence + 1 if sequence is None else sequence
        return self.append(ShadowMarkRecorded(sequence=next_sequence, snapshot=snapshot))


__all__ = [
    "LedgerAccountingError",
    "LedgerIdentifierCollisionError",
    "LedgerSequenceError",
    "PaperLedgerError",
    "SHADOW_SEED_CASH",
    "ShadowLedger",
    "ShadowLedgerEvent",
    "ShadowLedgerSnapshot",
    "ShadowMarkRecorded",
    "reduce_shadow_ledger",
]
