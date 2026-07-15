"""Atomic, typed risk gate for deterministic local shadow execution."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal

from pydantic import Field, field_validator

from paper.ledger import ShadowLedgerSnapshot
from paper.models import (
    ExecutionAssumptions,
    MarketSnapshot,
    NonNegativeDecimal,
    NonNegativeDerivedDecimal,
    OrderBatch,
    PaperModel,
    ShadowBatchExecuted,
    TargetAllocation,
    UtcDatetime,
    decimal_context,
)
from paper.simulator import (
    PaperSimulationError,
    SimulationConfig,
    simulate_shadow_batch,
)


def _canonical_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not symbol:
        raise ValueError("allowed symbols must not be empty")
    return symbol


class ShadowRiskLimits(PaperModel):
    """Hard pre-trade limits for the local $100 shadow account."""

    allowed_symbols: tuple[str, ...] = Field(min_length=1)
    max_snapshot_age_seconds: int = Field(default=300, ge=0)
    max_orders: int = Field(default=20, ge=0)
    max_order_notional: NonNegativeDecimal = Decimal("100.00")
    max_batch_notional: NonNegativeDecimal = Decimal("100.00")
    max_gross_exposure: NonNegativeDecimal = Decimal("100.00")
    minimum_cash: NonNegativeDecimal = Decimal("0")
    simulation: SimulationConfig = SimulationConfig()

    @field_validator("allowed_symbols", mode="after")
    @classmethod
    def _canonical_symbols(cls, symbols: tuple[str, ...]) -> tuple[str, ...]:
        canonical = tuple(sorted(_canonical_symbol(symbol) for symbol in symbols))
        if len(canonical) != len(set(canonical)):
            raise ValueError("allowed_symbols contains duplicates")
        return canonical


class RiskViolation(PaperModel):
    """One stable, machine-readable reason an atomic batch was rejected."""

    code: str
    message: str
    ticker: str | None = None
    order_id: str | None = None


class ShadowRiskApproved(PaperModel):
    """Approval plus the exact local event whose economics were checked."""

    decision: Literal["approved"] = "approved"
    evaluated_at: UtcDatetime
    snapshot_id: str
    batch_id: str
    projected_cash: NonNegativeDerivedDecimal
    projected_gross_exposure: NonNegativeDerivedDecimal
    batch_notional: NonNegativeDerivedDecimal
    simulation: SimulationConfig
    event: ShadowBatchExecuted


class ShadowRiskRejected(PaperModel):
    """Complete deterministic rejection; no ledger mutation has occurred."""

    decision: Literal["rejected"] = "rejected"
    evaluated_at: UtcDatetime | None
    snapshot_id: str
    batch_id: str
    violations: tuple[RiskViolation, ...] = Field(min_length=1)


type ShadowRiskDecision = ShadowRiskApproved | ShadowRiskRejected


def _violation(
    violations: list[RiskViolation],
    code: str,
    message: str,
    *,
    ticker: str | None = None,
    order_id: str | None = None,
) -> None:
    violations.append(
        RiskViolation(
            code=code,
            message=message,
            ticker=ticker,
            order_id=order_id,
        )
    )


def _evaluation_in_utc(evaluated_at: datetime) -> datetime | None:
    if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
        return None
    return evaluated_at.astimezone(UTC)


def _decimal_add(left: Decimal, right: Decimal) -> Decimal:
    with decimal_context():
        return left + right


def _decimal_subtract(left: Decimal, right: Decimal) -> Decimal:
    with decimal_context():
        return left - right


def evaluate_shadow_risk(
    batch: OrderBatch,
    target: TargetAllocation,
    snapshot: MarketSnapshot,
    ledger: ShadowLedgerSnapshot,
    assumptions: ExecutionAssumptions,
    *,
    mode: str,
    evaluated_at: datetime,
    limits: ShadowRiskLimits,
) -> ShadowRiskDecision:
    """Check a complete prospective batch and return approval or rejection.

    All calculations operate on local immutable inputs.  Approval contains the
    exact simulated event reviewed by the gate, preventing execution from
    silently substituting different fills after the checks pass.
    """

    violations: list[RiskViolation] = []
    evaluation_utc = _evaluation_in_utc(evaluated_at)

    if mode != "shadow":
        _violation(violations, "mode_not_shadow", "only mode='shadow' may be approved")
    if evaluation_utc is None:
        _violation(
            violations,
            "invalid_evaluation_time",
            "evaluation time must be timezone-aware",
        )
    else:
        age = evaluation_utc - snapshot.as_of
        if age < timedelta(0):
            _violation(
                violations,
                "snapshot_from_future",
                "market snapshot cannot be later than the evaluation time",
            )
        elif age > timedelta(seconds=limits.max_snapshot_age_seconds):
            _violation(
                violations,
                "stale_snapshot",
                "market snapshot exceeds the configured maximum age",
            )

    if target.effective_at != snapshot.as_of or batch.effective_at != snapshot.as_of:
        _violation(
            violations,
            "timestamp_mismatch",
            "target, batch, and snapshot timestamps must match exactly",
        )
    if batch.allocation_id != target.allocation_id:
        _violation(
            violations,
            "allocation_mismatch",
            "order batch does not identify the supplied target allocation",
        )
    if ledger.as_of is not None and snapshot.as_of < ledger.as_of:
        _violation(
            violations,
            "snapshot_before_ledger",
            "market snapshot cannot predate the reduced ledger state",
        )

    if batch.batch_id in ledger.executed_batch_ids:
        _violation(
            violations,
            "batch_id_reused",
            "batch ID was already executed and cannot be reused or collided",
        )
    reused_order_ids = sorted(
        {order.order_id for order in batch.orders}.intersection(ledger.executed_order_ids)
    )
    for order_id in reused_order_ids:
        _violation(
            violations,
            "order_id_reused",
            "order ID was already executed and cannot be reused or collided",
            order_id=order_id,
        )

    if len(batch.orders) > limits.max_orders:
        _violation(
            violations,
            "too_many_orders",
            f"batch has {len(batch.orders)} orders; limit is {limits.max_orders}",
        )
    canonical_order = tuple(
        sorted(batch.orders, key=lambda order: (order.side != "sell", order.ticker))
    )
    if batch.orders != canonical_order:
        _violation(
            violations,
            "noncanonical_order",
            "orders must be sorted by ticker with every sell before every buy",
        )

    allowed = set(limits.allowed_symbols)
    requested_symbols = sorted(
        {order.ticker for order in batch.orders}
        | {
            ticker
            for ticker, weight in target.weights.items()
            if weight != Decimal("0")
        }
    )
    for ticker in requested_symbols:
        if ticker not in allowed:
            _violation(
                violations,
                "symbol_not_allowed",
                f"symbol {ticker} is outside the configured allow-list",
                ticker=ticker,
            )

    snapshot_symbols = set(snapshot.marks())
    missing_held_marks = sorted(
        ticker
        for ticker, quantity in ledger.positions.items()
        if quantity != Decimal("0") and ticker not in snapshot_symbols
    )
    if missing_held_marks:
        _violation(
            violations,
            "missing_fresh_mark",
            f"snapshot is missing fresh marks for held assets: {missing_held_marks}",
        )

    required_cash = limits.minimum_cash
    if not missing_held_marks:
        with decimal_context():
            pretrade_equity = ledger.cash + sum(
                (
                    quantity * snapshot.quote(ticker).last
                    for ticker, quantity in ledger.positions.items()
                    if quantity != Decimal("0")
                ),
                Decimal("0"),
            )
            required_cash = max(
                limits.minimum_cash,
                pretrade_equity * target.cash_weight,
            )

    event: ShadowBatchExecuted | None = None
    try:
        event = simulate_shadow_batch(
            batch,
            snapshot,
            assumptions,
            sequence=ledger.sequence + 1,
            config=limits.simulation,
        )
    except (PaperSimulationError, ValueError) as exc:
        _violation(violations, "simulation_invalid", str(exc))

    projected_cash = ledger.cash
    projected_gross = Decimal("0")
    batch_notional = Decimal("0")
    if event is not None:
        reused_fill_ids = sorted(
            {fill.fill_id for fill in event.fills}.intersection(ledger.executed_fill_ids)
        )
        for fill_id in reused_fill_ids:
            _violation(
                violations,
                "fill_id_reused",
                f"fill ID {fill_id!r} was already executed and cannot be reused or collided",
            )

        positions = dict(ledger.positions)
        for fill in event.fills:
            batch_notional = _decimal_add(batch_notional, fill.notional)
            if fill.notional > limits.max_order_notional:
                _violation(
                    violations,
                    "order_notional_limit",
                    "simulated order notional exceeds the configured limit",
                    ticker=fill.ticker,
                    order_id=fill.order_id,
                )
            current = positions.get(fill.ticker, Decimal("0"))
            if fill.side == "buy":
                positions[fill.ticker] = _decimal_add(current, fill.quantity)
                projected_cash = _decimal_subtract(
                    projected_cash,
                    _decimal_add(fill.notional, fill.fee),
                )
            else:
                positions[fill.ticker] = _decimal_subtract(current, fill.quantity)
                projected_cash = _decimal_add(
                    projected_cash,
                    _decimal_subtract(fill.notional, fill.fee),
                )

        if batch_notional > limits.max_batch_notional:
            _violation(
                violations,
                "batch_notional_limit",
                "total simulated batch notional exceeds the configured limit",
            )
        short_tickers = sorted(
            ticker for ticker, quantity in positions.items() if quantity < Decimal("0")
        )
        for ticker in short_tickers:
            _violation(
                violations,
                "short_position",
                f"atomic batch would create a short position in {ticker}",
                ticker=ticker,
            )

        missing_marks = sorted(
            ticker
            for ticker, quantity in positions.items()
            if quantity != Decimal("0") and ticker not in snapshot.marks()
        )
        newly_missing_marks = sorted(set(missing_marks) - set(missing_held_marks))
        if newly_missing_marks:
            _violation(
                violations,
                "missing_fresh_mark",
                f"snapshot is missing fresh marks for held assets: {newly_missing_marks}",
            )
        if not missing_marks:
            with decimal_context():
                projected_gross = sum(
                    (
                        abs(quantity * snapshot.quote(ticker).last)
                        for ticker, quantity in positions.items()
                        if quantity != Decimal("0")
                    ),
                    Decimal("0"),
                )
            if projected_gross > limits.max_gross_exposure:
                _violation(
                    violations,
                    "gross_exposure_limit",
                    "fresh-marked gross exposure exceeds the configured limit",
                )

        if projected_cash < required_cash:
            _violation(
                violations,
                "insufficient_cash",
                "atomic batch would spend through the target or configured minimum cash",
            )

    if violations or event is None or evaluation_utc is None:
        return ShadowRiskRejected(
            evaluated_at=evaluation_utc,
            snapshot_id=snapshot.snapshot_id,
            batch_id=batch.batch_id,
            violations=tuple(violations),
        )

    return ShadowRiskApproved(
        evaluated_at=evaluation_utc,
        snapshot_id=snapshot.snapshot_id,
        batch_id=batch.batch_id,
        projected_cash=projected_cash,
        projected_gross_exposure=projected_gross,
        batch_notional=batch_notional,
        simulation=limits.simulation,
        event=event,
    )


def gate_shadow_batch(
    batch: OrderBatch,
    target: TargetAllocation,
    snapshot: MarketSnapshot,
    ledger: ShadowLedgerSnapshot,
    assumptions: ExecutionAssumptions,
    *,
    mode: str,
    evaluated_at: datetime,
    limits: ShadowRiskLimits,
) -> ShadowRiskDecision:
    """Named risk-gate alias for :func:`evaluate_shadow_risk`."""

    return evaluate_shadow_risk(
        batch,
        target,
        snapshot,
        ledger,
        assumptions,
        mode=mode,
        evaluated_at=evaluated_at,
        limits=limits,
    )


__all__ = [
    "RiskViolation",
    "ShadowRiskApproved",
    "ShadowRiskDecision",
    "ShadowRiskLimits",
    "ShadowRiskRejected",
    "evaluate_shadow_risk",
    "gate_shadow_batch",
]
