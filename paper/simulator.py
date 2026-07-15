"""Deterministic, fully local fill simulation for shadow order batches."""

from __future__ import annotations

import hashlib

from paper.models import (
    DerivedDecimal,
    ExecutionAssumptions,
    Fill,
    MarketSnapshot,
    NonNegativeDecimal,
    OrderBatch,
    OrderIntent,
    PaperModel,
    PositiveDecimal,
    PositiveDerivedDecimal,
    ShadowBatchExecuted,
    SimulationConfig,
    expected_fill_economics,
)


class PaperSimulationError(ValueError):
    """Raised when an order batch cannot be simulated without ambiguity."""


class SimulatedFillTerms(PaperModel):
    """Exact economic terms for one prospective local fill."""

    price: PositiveDecimal
    fee: NonNegativeDecimal
    notional: PositiveDerivedDecimal
    cash_change: DerivedDecimal


def fill_terms(
    order: OrderIntent,
    snapshot: MarketSnapshot,
    assumptions: ExecutionAssumptions,
    *,
    config: SimulationConfig | None = None,
) -> SimulatedFillTerms:
    """Calculate conservative price, fee, and cash movement for one intent."""

    simulation = config or SimulationConfig()
    try:
        quote = snapshot.quote(order.ticker)
    except KeyError as exc:
        raise PaperSimulationError(
            f"snapshot {snapshot.snapshot_id!r} has no quote for {order.ticker}"
        ) from exc

    try:
        economics = expected_fill_economics(order, quote, assumptions, simulation)
        return SimulatedFillTerms(
            price=economics.price,
            fee=economics.fee,
            notional=economics.notional,
            cash_change=economics.cash_change,
        )
    except ValueError as exc:
        raise PaperSimulationError(str(exc)) from exc


def _fill_id(
    *,
    batch: OrderBatch,
    order: OrderIntent,
    snapshot: MarketSnapshot,
    assumptions: ExecutionAssumptions,
    config: SimulationConfig,
    terms: SimulatedFillTerms,
) -> str:
    material = "\x1f".join(
        (
            batch.canonical_json(),
            order.canonical_json(),
            snapshot.canonical_json(),
            assumptions.canonical_json(),
            config.canonical_json(),
            terms.canonical_json(),
        )
    ).encode()
    return f"fill_{hashlib.sha256(material).hexdigest()}"


def simulate_shadow_batch(
    batch: OrderBatch,
    snapshot: MarketSnapshot,
    assumptions: ExecutionAssumptions,
    *,
    sequence: int,
    config: SimulationConfig | None = None,
) -> ShadowBatchExecuted:
    """Create one complete local fill per order without touching a broker.

    The function either returns a fully valid atomic event or raises before an
    event is constructed.  It has no clock, randomness, network, or mutable
    state, so identical inputs produce identical event JSON.
    """

    simulation = config or SimulationConfig()
    if batch.effective_at != snapshot.as_of:
        raise PaperSimulationError(
            "order batch and market snapshot must have the same effective timestamp"
        )

    fills: list[Fill] = []
    for order in batch.orders:
        terms = fill_terms(order, snapshot, assumptions, config=simulation)
        fills.append(
            Fill(
                fill_id=_fill_id(
                    batch=batch,
                    order=order,
                    snapshot=snapshot,
                    assumptions=assumptions,
                    config=simulation,
                    terms=terms,
                ),
                order_id=order.order_id,
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
                price=terms.price,
                fee=terms.fee,
                executed_at=batch.effective_at,
            )
        )

    return ShadowBatchExecuted(
        sequence=sequence,
        batch=batch,
        snapshot=snapshot,
        fills=tuple(fills),
        assumptions=assumptions,
        simulation=simulation,
    )


__all__ = [
    "PaperSimulationError",
    "SimulatedFillTerms",
    "SimulationConfig",
    "fill_terms",
    "simulate_shadow_batch",
]
