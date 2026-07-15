"""Deterministic target-to-order planning for the exact shadow account."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from decimal import ROUND_DOWN, Decimal
from typing import Literal

from paper.ledger import ShadowLedgerSnapshot
from paper.models import (
    ExecutionAssumptions,
    MarketSnapshot,
    NonNegativeDecimal,
    OrderBatch,
    OrderIntent,
    PaperModel,
    TargetAllocation,
    decimal_context,
)
from paper.simulator import SimulationConfig, fill_terms


class PaperPlanningError(ValueError):
    """Raised when a target cannot be converted into unambiguous local orders."""


class PlanningConfig(PaperModel):
    """Execution-aware constraints used while creating an order batch."""

    minimum_trade_notional: NonNegativeDecimal = Decimal("0")
    cash_buffer: NonNegativeDecimal = Decimal("0")
    simulation: SimulationConfig = SimulationConfig()


@dataclass(frozen=True, slots=True)
class _Candidate:
    ticker: str
    side: Literal["buy", "sell"]
    quantity: Decimal
    reference_price: Decimal


def _round_quantity(quantity: Decimal, step: Decimal) -> Decimal:
    with decimal_context():
        return (quantity / step).to_integral_value(rounding=ROUND_DOWN) * step


def _hash_id(prefix: str, *parts: str) -> str:
    material = "\x1f".join(parts).encode()
    return f"{prefix}_{hashlib.sha256(material).hexdigest()}"


def _decimal_sum(values: list[Decimal]) -> Decimal:
    with decimal_context():
        return sum(values, Decimal("0"))


def _buy_cost(
    candidate: _Candidate,
    snapshot: MarketSnapshot,
    assumptions: ExecutionAssumptions,
    config: PlanningConfig,
) -> Decimal:
    with decimal_context():
        return -_candidate_cash_change(candidate, snapshot, assumptions, config)


def _temporary_order(candidate: _Candidate) -> OrderIntent:
    return OrderIntent(
        order_id="prospective-order",
        ticker=candidate.ticker,
        side=candidate.side,
        quantity=candidate.quantity,
        reference_price=candidate.reference_price,
    )


def _candidate_cash_change(
    candidate: _Candidate,
    snapshot: MarketSnapshot,
    assumptions: ExecutionAssumptions,
    config: PlanningConfig,
) -> Decimal:
    return fill_terms(
        _temporary_order(candidate),
        snapshot,
        assumptions,
        config=config.simulation,
    ).cash_change


def _meets_minimum(candidate: _Candidate, config: PlanningConfig) -> bool:
    with decimal_context():
        return candidate.quantity * candidate.reference_price >= config.minimum_trade_notional


def _candidate_with_units(
    candidate: _Candidate,
    units: int,
    step: Decimal,
) -> _Candidate | None:
    if units <= 0:
        return None
    with decimal_context():
        quantity = Decimal(units) * step
    return replace(candidate, quantity=quantity)


def _largest_affordable_quantity(
    candidate: _Candidate,
    budget: Decimal,
    *,
    snapshot: MarketSnapshot,
    assumptions: ExecutionAssumptions,
    config: PlanningConfig,
) -> _Candidate | None:
    """Return the largest step-aligned prefix of a buy that fits ``budget``."""

    if budget <= Decimal("0"):
        return None
    step = assumptions.quantity_step
    with decimal_context():
        maximum_units = int(candidate.quantity / step)
    low = 0
    high = maximum_units
    while low < high:
        midpoint = (low + high + 1) // 2
        trial = _candidate_with_units(candidate, midpoint, step)
        assert trial is not None
        cost = _buy_cost(trial, snapshot, assumptions, config)
        if cost <= budget:
            low = midpoint
        else:
            high = midpoint - 1
    result = _candidate_with_units(candidate, low, step)
    if result is None or not _meets_minimum(result, config):
        return None
    return result


def _scale_buys_to_cash(
    buys: list[_Candidate],
    available_cash: Decimal,
    *,
    snapshot: MarketSnapshot,
    assumptions: ExecutionAssumptions,
    config: PlanningConfig,
) -> list[_Candidate]:
    if not buys or available_cash <= Decimal("0"):
        return []

    costs = [
        _buy_cost(candidate, snapshot, assumptions, config)
        for candidate in buys
    ]
    total_cost = _decimal_sum(costs)
    if total_cost <= available_cash:
        return buys

    # Scale all desired buys by one common factor first.  Step rounding is
    # always downward, preserving long-only cash safety.
    with decimal_context():
        ratio = available_cash / total_cost
    scaled: list[_Candidate] = []
    for candidate in buys:
        with decimal_context():
            proportional_quantity = candidate.quantity * ratio
        scaled.append(
            replace(
                candidate,
                quantity=_round_quantity(
                    proportional_quantity,
                    assumptions.quantity_step,
                ),
            )
        )
    scaled = [
        candidate
        for candidate in scaled
        if candidate.quantity > Decimal("0") and _meets_minimum(candidate, config)
    ]

    # Minimum commissions do not scale with quantity.  If they leave the
    # proportional result a few cents too large, trim in reverse canonical
    # ticker order using an exact binary search over quantity steps.
    while scaled:
        scaled_costs = [
            _buy_cost(candidate, snapshot, assumptions, config)
            for candidate in scaled
        ]
        aggregate = _decimal_sum(scaled_costs)
        if aggregate <= available_cash:
            return scaled
        index = len(scaled) - 1
        with decimal_context():
            budget = available_cash - (aggregate - scaled_costs[index])
        trimmed = _largest_affordable_quantity(
            scaled[index],
            budget,
            snapshot=snapshot,
            assumptions=assumptions,
            config=config,
        )
        if trimmed is None:
            scaled.pop(index)
        else:
            scaled[index] = trimmed

    return []


def plan_orders(
    target: TargetAllocation,
    snapshot: MarketSnapshot,
    ledger: ShadowLedgerSnapshot,
    assumptions: ExecutionAssumptions,
    *,
    config: PlanningConfig | None = None,
) -> OrderBatch:
    """Create a canonical sell-first batch from exact marked account state.

    The target is valued against ``snapshot.last`` prices.  Desired deltas are
    rounded down to the configured quantity step, dust is omitted, and buys are
    scaled against conservative simulated prices and fees so the planned batch
    cannot spend through the requested cash buffer.
    """

    planning = config or PlanningConfig()
    if target.effective_at != snapshot.as_of:
        raise PaperPlanningError(
            "target allocation and market snapshot must have the same timestamp"
        )
    if ledger.as_of is not None and snapshot.as_of < ledger.as_of:
        raise PaperPlanningError("planning snapshot cannot predate the ledger state")

    held_tickers = {
        ticker for ticker, quantity in ledger.positions.items() if quantity != Decimal("0")
    }
    target_tickers = {
        ticker for ticker, weight in target.weights.items() if weight != Decimal("0")
    }
    required_tickers = held_tickers | target_tickers
    quote_tickers = {quote.ticker for quote in snapshot.quotes}
    missing = sorted(required_tickers - quote_tickers)
    if missing:
        raise PaperPlanningError(f"snapshot is missing held or target assets: {missing}")

    with decimal_context():
        marked_equity = ledger.cash + sum(
            (
                quantity * snapshot.quote(ticker).last
                for ticker, quantity in ledger.positions.items()
            ),
            Decimal("0"),
        )
    if marked_equity <= Decimal("0"):
        raise PaperPlanningError("marked account equity must be positive")

    sells: list[_Candidate] = []
    buys: list[_Candidate] = []
    for ticker in sorted(required_tickers):
        quote = snapshot.quote(ticker)
        current_quantity = ledger.positions.get(ticker, Decimal("0"))
        with decimal_context():
            desired_value = marked_equity * target.weights.get(ticker, Decimal("0"))
            current_value = current_quantity * quote.last
        if current_value == desired_value:
            continue
        if current_value > desired_value:
            with decimal_context():
                raw_quantity = (current_value - desired_value) / quote.last
            quantity = _round_quantity(raw_quantity, assumptions.quantity_step)
            quantity = min(quantity, current_quantity)
            candidate = _Candidate(ticker, "sell", quantity, quote.bid)
            if quantity > Decimal("0") and _meets_minimum(candidate, planning):
                sells.append(candidate)
        else:
            with decimal_context():
                raw_quantity = (desired_value - current_value) / quote.last
            quantity = _round_quantity(raw_quantity, assumptions.quantity_step)
            candidate = _Candidate(ticker, "buy", quantity, quote.ask)
            if quantity > Decimal("0") and _meets_minimum(candidate, planning):
                buys.append(candidate)

    sell_cash_changes = [
        _candidate_cash_change(candidate, snapshot, assumptions, planning)
        for candidate in sells
    ]
    with decimal_context():
        projected_cash_after_sells = ledger.cash + sum(
            sell_cash_changes,
            Decimal("0"),
        )
    if projected_cash_after_sells < Decimal("0"):
        raise PaperPlanningError("conservative sell fees would create negative cash")
    with decimal_context():
        allocation_cash = marked_equity * target.cash_weight
    required_cash = max(planning.cash_buffer, allocation_cash)
    with decimal_context():
        available_for_buys = max(
            Decimal("0"),
            projected_cash_after_sells - required_cash,
        )
    buys = _scale_buys_to_cash(
        buys,
        available_for_buys,
        snapshot=snapshot,
        assumptions=assumptions,
        config=planning,
    )

    common_material = (
        target.canonical_json(),
        snapshot.canonical_json(),
        ledger.canonical_json(),
        assumptions.canonical_json(),
        planning.canonical_json(),
    )
    orders = tuple(
        OrderIntent(
            order_id=_hash_id(
                "order",
                *common_material,
                candidate.ticker,
                candidate.side,
                format(candidate.quantity, "f"),
                format(candidate.reference_price, "f"),
            ),
            ticker=candidate.ticker,
            side=candidate.side,
            quantity=candidate.quantity,
            reference_price=candidate.reference_price,
        )
        for candidate in [*sells, *buys]
    )
    batch_id = _hash_id(
        "batch",
        *common_material,
        *(order.canonical_json() for order in orders),
    )
    return OrderBatch(
        batch_id=batch_id,
        allocation_id=target.allocation_id,
        effective_at=target.effective_at,
        orders=orders,
    )


__all__ = [
    "PaperPlanningError",
    "PlanningConfig",
    "plan_orders",
]
