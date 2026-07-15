"""Independent normalized, frictionless comparison ledger.

This ledger is intentionally not a mode of the exact $100 shadow ledger.  It
has its own unit capital, holdings, state, and history so execution rounding,
fees, or accidental mutations in the shadow account cannot contaminate the
ideal comparison curve.
"""

from __future__ import annotations

from datetime import datetime
from decimal import ROUND_DOWN, Decimal
from typing import Self

from pydantic import Field, field_validator, model_validator

from paper.models import (
    DerivedWeightDecimal,
    MarketSnapshot,
    NonNegativeDerivedDecimal,
    PaperModel,
    PositiveDecimal,
    PositiveDerivedDecimal,
    TargetAllocation,
    UtcDatetime,
    decimal_context,
    freeze_decimal_map,
)

IDEAL_SEED_NAV = Decimal("1")


class IdealLedgerError(Exception):
    """Raised when a frictionless comparison update is ambiguous or invalid."""


class IdealLedgerSnapshot(PaperModel):
    """Exact marked state of the isolated unit-capital comparison account."""

    sequence: int = Field(ge=0)
    as_of: UtcDatetime | None = None
    nav: PositiveDerivedDecimal
    cash: NonNegativeDerivedDecimal
    quantities: dict[str, NonNegativeDerivedDecimal]
    marks: dict[str, PositiveDecimal]
    weights: dict[str, DerivedWeightDecimal]
    applied_allocation_ids: tuple[str, ...] = ()

    @field_validator("quantities", "marks", "weights", mode="after")
    @classmethod
    def _ordered_maps(cls, value: dict[str, Decimal]) -> dict[str, Decimal]:
        return freeze_decimal_map(value)

    @model_validator(mode="after")
    def _consistent_value(self) -> Self:
        missing = sorted(set(self.quantities) - set(self.marks))
        if missing:
            raise ValueError(f"ideal positions are missing marks: {missing}")
        with decimal_context():
            expected_nav = self.cash + sum(
                (quantity * self.marks[ticker] for ticker, quantity in self.quantities.items()),
                Decimal("0"),
            )
        if self.nav != expected_nav:
            raise ValueError("ideal NAV must equal exact marked holdings plus cash")
        if set(self.weights) != set(self.quantities):
            raise ValueError("ideal weights and quantities must name the same assets")
        with decimal_context():
            expected_weights = {
                ticker: quantity * self.marks[ticker] / self.nav
                for ticker, quantity in self.quantities.items()
            }
        if self.weights != expected_weights:
            raise ValueError("ideal weights must equal exact marked holdings divided by NAV")
        return self

    @property
    def cash_weight(self) -> Decimal:
        with decimal_context():
            return self.cash / self.nav


def _valued_state(
    *,
    sequence: int,
    as_of: datetime | None,
    cash: Decimal,
    quantities: dict[str, Decimal],
    marks: dict[str, Decimal],
    allocation_ids: list[str],
) -> IdealLedgerSnapshot:
    with decimal_context():
        active_quantities = {
            ticker: quantity
            for ticker, quantity in sorted(quantities.items())
            if quantity != Decimal("0")
        }
        active_marks = {ticker: marks[ticker] for ticker in active_quantities}
        nav = cash + sum(
            (
                quantity * active_marks[ticker]
                for ticker, quantity in active_quantities.items()
            ),
            Decimal("0"),
        )
        if nav <= Decimal("0"):
            raise IdealLedgerError("ideal NAV must remain strictly positive")
        weights = {
            ticker: quantity * active_marks[ticker] / nav
            for ticker, quantity in active_quantities.items()
        }
        return IdealLedgerSnapshot(
            sequence=sequence,
            as_of=as_of,
            nav=nav,
            cash=cash,
            quantities=active_quantities,
            marks=active_marks,
            weights=weights,
            applied_allocation_ids=tuple(allocation_ids),
        )


class IdealLedger:
    """Unit-capital, long-only, fee-free portfolio used as a benchmark."""

    def __init__(self) -> None:
        self._sequence = 0
        self._as_of: datetime | None = None
        self._cash = IDEAL_SEED_NAV
        self._quantities: dict[str, Decimal] = {}
        self._marks: dict[str, Decimal] = {}
        self._allocations: dict[str, tuple[TargetAllocation, MarketSnapshot]] = {}
        self._snapshots: dict[str, MarketSnapshot] = {}
        self._marked_snapshots: dict[str, MarketSnapshot] = {}
        self._allocation_ids: list[str] = []
        self._history: list[IdealLedgerSnapshot] = [
            _valued_state(
                sequence=0,
                as_of=None,
                cash=IDEAL_SEED_NAV,
                quantities={},
                marks={},
                allocation_ids=[],
            )
        ]

    @property
    def state(self) -> IdealLedgerSnapshot:
        return self._history[-1]

    @property
    def history(self) -> tuple[IdealLedgerSnapshot, ...]:
        return tuple(self._history)

    def _require_forward_snapshot(self, snapshot: MarketSnapshot) -> None:
        if self._as_of is not None and snapshot.as_of < self._as_of:
            raise IdealLedgerError("ideal ledger timestamps must be non-decreasing")
        missing = sorted(set(self._quantities) - set(snapshot.marks()))
        if missing:
            raise IdealLedgerError(f"snapshot is missing held ideal assets: {missing}")

    def _register_snapshot(self, snapshot: MarketSnapshot) -> None:
        prior = self._snapshots.get(snapshot.snapshot_id)
        if prior is not None and prior.canonical_json() != snapshot.canonical_json():
            raise IdealLedgerError(
                f"snapshot ID {snapshot.snapshot_id!r} already has different content"
            )

    def mark(self, snapshot: MarketSnapshot) -> IdealLedgerSnapshot:
        """Mark all existing ideal holdings at a supplied local snapshot."""

        self._register_snapshot(snapshot)
        prior_mark = self._marked_snapshots.get(snapshot.snapshot_id)
        if prior_mark is not None:
            if prior_mark.canonical_json() == snapshot.canonical_json():
                return self.state
            raise IdealLedgerError(
                f"snapshot ID {snapshot.snapshot_id!r} already has different content"
            )
        self._require_forward_snapshot(snapshot)
        candidate_marks = dict(self._marks)
        candidate_marks.update(snapshot.marks())
        candidate = _valued_state(
            sequence=self._sequence + 1,
            as_of=snapshot.as_of,
            cash=self._cash,
            quantities=self._quantities,
            marks=candidate_marks,
            allocation_ids=self._allocation_ids,
        )
        self._sequence = candidate.sequence
        self._as_of = candidate.as_of
        self._marks = candidate_marks
        self._snapshots[snapshot.snapshot_id] = snapshot
        self._marked_snapshots[snapshot.snapshot_id] = snapshot
        self._history.append(candidate)
        return candidate

    def apply(
        self,
        target: TargetAllocation,
        snapshot: MarketSnapshot,
    ) -> IdealLedgerSnapshot:
        """Frictionlessly apply target weights at their exact effective instant."""

        if target.effective_at != snapshot.as_of:
            raise IdealLedgerError(
                "target and market snapshot must have the same effective timestamp"
            )

        prior = self._allocations.get(target.allocation_id)
        if prior is not None:
            if (
                prior[0].canonical_json() == target.canonical_json()
                and prior[1].canonical_json() == snapshot.canonical_json()
            ):
                return self.state
            raise IdealLedgerError(
                f"allocation ID {target.allocation_id!r} already has different content"
            )

        self._register_snapshot(snapshot)
        self._require_forward_snapshot(snapshot)
        snapshot_marks = snapshot.marks()
        missing_targets = sorted(set(target.weights) - set(snapshot_marks))
        if missing_targets:
            raise IdealLedgerError(f"snapshot is missing target assets: {missing_targets}")

        with decimal_context():
            marked_nav = self._cash + sum(
                (
                    quantity * snapshot_marks[ticker]
                    for ticker, quantity in self._quantities.items()
                ),
                Decimal("0"),
            )
        if marked_nav <= Decimal("0"):
            raise IdealLedgerError("ideal NAV must remain strictly positive")

        # Round quantities toward zero at high precision, then leave the exact
        # sub-ulp residual in cash.  Therefore the rebalance preserves NAV
        # exactly and can never create leverage through decimal division.
        with decimal_context() as context:
            context.rounding = ROUND_DOWN
            quantities = {
                ticker: (marked_nav * weight) / snapshot_marks[ticker]
                for ticker, weight in target.weights.items()
                if weight != Decimal("0")
            }
        with decimal_context():
            invested = sum(
                (
                    quantity * snapshot_marks[ticker]
                    for ticker, quantity in quantities.items()
                ),
                Decimal("0"),
            )
            cash = marked_nav - invested
        if cash < Decimal("0"):
            raise IdealLedgerError("decimal allocation unexpectedly exceeded ideal NAV")

        next_allocation_ids = [*self._allocation_ids, target.allocation_id]
        candidate = _valued_state(
            sequence=self._sequence + 1,
            as_of=snapshot.as_of,
            cash=cash,
            quantities=quantities,
            marks=snapshot_marks,
            allocation_ids=next_allocation_ids,
        )

        self._sequence = candidate.sequence
        self._as_of = candidate.as_of
        self._cash = cash
        self._quantities = quantities
        self._marks = snapshot_marks
        self._allocations[target.allocation_id] = (target, snapshot)
        self._snapshots[snapshot.snapshot_id] = snapshot
        self._allocation_ids = next_allocation_ids
        self._history.append(candidate)
        return candidate


__all__ = [
    "IDEAL_SEED_NAV",
    "IdealLedger",
    "IdealLedgerError",
    "IdealLedgerSnapshot",
]
