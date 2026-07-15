"""Adversarial, hand-accounted tests for the local paper execution layer."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext

import pytest
from pydantic import ValidationError

from paper.ledger import ShadowLedger, ShadowLedgerSnapshot
from paper.models import (
    ExecutionAssumptions,
    Fill,
    MarketSnapshot,
    OrderBatch,
    OrderIntent,
    Quote,
    ShadowBatchExecuted,
    TargetAllocation,
)
from paper.planner import PaperPlanningError, PlanningConfig, plan_orders
from paper.risk import (
    ShadowRiskApproved,
    ShadowRiskLimits,
    ShadowRiskRejected,
    evaluate_shadow_risk,
)
from paper.simulator import (
    PaperSimulationError,
    SimulationConfig,
    simulate_shadow_batch,
)

T0 = datetime(2026, 1, 2, 21, tzinfo=UTC)


def _quote(
    ticker: str,
    *,
    bid: str,
    ask: str,
    last: str,
    at: datetime = T0,
) -> Quote:
    return Quote(ticker=ticker, bid=bid, ask=ask, last=last, as_of=at)


def _snapshot(*quotes: Quote, at: datetime = T0, snapshot_id: str = "close") -> MarketSnapshot:
    return MarketSnapshot(snapshot_id=snapshot_id, as_of=at, quotes=quotes)


def _target(**weights: str) -> TargetAllocation:
    return TargetAllocation(allocation_id="target-1", effective_at=T0, weights=weights)


def _state(
    *,
    cash: str = "100.00",
    positions: dict[str, str] | None = None,
    marks: dict[str, str] | None = None,
    sequence: int = 0,
    as_of: datetime | None = None,
    batch_ids: tuple[str, ...] = (),
    order_ids: tuple[str, ...] = (),
    fill_ids: tuple[str, ...] = (),
) -> ShadowLedgerSnapshot:
    decimal_positions = {
        ticker: Decimal(quantity) for ticker, quantity in (positions or {}).items()
    }
    decimal_marks = {ticker: Decimal(price) for ticker, price in (marks or {}).items()}
    cash_value = Decimal(cash)
    gross = sum(
        (
            quantity * decimal_marks[ticker]
            for ticker, quantity in decimal_positions.items()
        ),
        Decimal("0"),
    )
    return ShadowLedgerSnapshot(
        sequence=sequence,
        as_of=as_of,
        cash=cash_value,
        positions=decimal_positions,
        marks=decimal_marks,
        cumulative_fees="0",
        executed_batch_ids=batch_ids,
        executed_order_ids=order_ids,
        executed_fill_ids=fill_ids,
        equity=cash_value + gross,
        gross_exposure=gross,
    )


def _batch(*orders: OrderIntent, batch_id: str = "batch-1") -> OrderBatch:
    return OrderBatch(
        batch_id=batch_id,
        allocation_id="target-1",
        effective_at=T0,
        orders=orders,
    )


def _order(
    ticker: str,
    side: str,
    quantity: str,
    reference_price: str,
    *,
    order_id: str | None = None,
) -> OrderIntent:
    return OrderIntent(
        order_id=order_id or f"order-{ticker}",
        ticker=ticker,
        side=side,
        quantity=quantity,
        reference_price=reference_price,
    )


def _codes(decision: ShadowRiskRejected) -> set[str]:
    return {violation.code for violation in decision.violations}


def test_simulator_uses_adverse_tick_rounding_and_exact_fees() -> None:
    snapshot = _snapshot(
        _quote("AAA", bid="9.95", ask="10.05", last="10"),
        _quote("BBB", bid="9.95", ask="10.05", last="10"),
    )
    batch = _batch(
        _order("BBB", "sell", "2", "9.95"),
        _order("AAA", "buy", "2", "10.05"),
    )
    assumptions = ExecutionAssumptions(
        slippage_bps="25",
        commission_bps="10",
        minimum_fee="0.01",
        quantity_step="0.001",
    )

    event = simulate_shadow_batch(batch, snapshot, assumptions, sequence=1)

    sell, buy = event.fills
    # 9.95 * (1 - 0.0025) = 9.925125, rounded down to 9.92.
    assert sell.price == Decimal("9.92")
    assert sell.fee == Decimal("0.02")
    # 10.05 * (1 + 0.0025) = 10.075125, rounded up to 10.08.
    assert buy.price == Decimal("10.08")
    assert buy.fee == Decimal("0.03")
    assert event == simulate_shadow_batch(batch, snapshot, assumptions, sequence=1)
    assert event.canonical_json() == simulate_shadow_batch(
        batch, snapshot, assumptions, sequence=1
    ).canonical_json()


@pytest.mark.parametrize(
    ("batch", "snapshot", "assumptions", "match"),
    [
        (
            _batch(_order("AAA", "buy", "1", "10.00")),
            _snapshot(_quote("AAA", bid="9.99", ask="10.01", last="10")),
            ExecutionAssumptions(),
            "reference price",
        ),
        (
            _batch(_order("AAA", "buy", "1.005", "10.01")),
            _snapshot(_quote("AAA", bid="9.99", ask="10.01", last="10")),
            ExecutionAssumptions(quantity_step="0.01"),
            "quantity_step",
        ),
        (
            _batch(_order("AAA", "sell", "1", "9.99")),
            _snapshot(_quote("AAA", bid="9.99", ask="10.01", last="10")),
            ExecutionAssumptions(slippage_bps="10000"),
            "less than 10000",
        ),
    ],
)
def test_simulator_rejects_ambiguous_or_invalid_execution_inputs(
    batch: OrderBatch,
    snapshot: MarketSnapshot,
    assumptions: ExecutionAssumptions,
    match: str,
) -> None:
    with pytest.raises(PaperSimulationError, match=match):
        simulate_shadow_batch(batch, snapshot, assumptions, sequence=1)


def test_simulator_rejects_missing_quote_and_timestamp_mismatch() -> None:
    batch = _batch(_order("AAA", "buy", "1", "10"))
    missing = _snapshot(_quote("BBB", bid="10", ask="10", last="10"))
    with pytest.raises(PaperSimulationError, match="no quote"):
        simulate_shadow_batch(batch, missing, ExecutionAssumptions(), sequence=1)

    later = T0 + timedelta(minutes=1)
    mismatched = _snapshot(
        _quote("AAA", bid="10", ask="10", last="10", at=later),
        at=later,
    )
    with pytest.raises(PaperSimulationError, match="same effective timestamp"):
        simulate_shadow_batch(batch, mismatched, ExecutionAssumptions(), sequence=1)


def test_simulator_preserves_high_scale_derived_notional() -> None:
    batch = _batch(_order("AAA", "buy", "1.000000000000000000", "10.00"))
    snapshot = _snapshot(_quote("AAA", bid="10.00", ask="10.00", last="10.00"))

    event = simulate_shadow_batch(batch, snapshot, ExecutionAssumptions(), sequence=1)

    assert event.fills[0].notional == Decimal("10.00000000000000000000")
    assert event.snapshot == snapshot
    assert event.simulation == SimulationConfig()


def test_executed_event_rejects_fill_economics_not_derived_from_provenance() -> None:
    batch = _batch(_order("AAA", "buy", "1", "10.01"))
    snapshot = _snapshot(_quote("AAA", bid="9.99", ask="10.01", last="10"))
    assumptions = ExecutionAssumptions(slippage_bps="10")
    valid = simulate_shadow_batch(batch, snapshot, assumptions, sequence=1)
    fill = valid.fills[0]
    dishonest_fill = Fill(
        fill_id=fill.fill_id,
        order_id=fill.order_id,
        ticker=fill.ticker,
        side=fill.side,
        quantity=fill.quantity,
        price="10.01",
        fee=fill.fee,
        executed_at=fill.executed_at,
    )

    with pytest.raises(ValidationError, match="does not match persisted"):
        ShadowBatchExecuted(
            sequence=1,
            batch=batch,
            snapshot=snapshot,
            fills=(dishonest_fill,),
            assumptions=assumptions,
            simulation=SimulationConfig(),
        )


def test_planner_marks_exact_state_and_orders_sells_before_buys() -> None:
    state = _state(
        cash="20",
        positions={"AAA": "5", "BBB": "1"},
        marks={"AAA": "99", "BBB": "99"},  # stale marks must not drive target dollars
        sequence=3,
        as_of=T0 - timedelta(days=1),
    )
    snapshot = _snapshot(
        _quote("CCC", bid="9.9", ask="10.1", last="10"),
        _quote("BBB", bid="29.9", ask="30.1", last="30"),
        _quote("AAA", bid="9.9", ask="10.1", last="10"),
    )
    target = _target(AAA="0.2", BBB="0", CCC="0.7")
    assumptions = ExecutionAssumptions(quantity_step="0.1")

    batch = plan_orders(target, snapshot, state, assumptions)

    assert [(order.side, order.ticker) for order in batch.orders] == [
        ("sell", "AAA"),
        ("sell", "BBB"),
        ("buy", "CCC"),
    ]
    assert [order.quantity for order in batch.orders] == [
        Decimal("3.0"),
        Decimal("1.0"),
        Decimal("6.8"),
    ]
    assert [order.reference_price for order in batch.orders] == [
        Decimal("9.9"),
        Decimal("29.9"),
        Decimal("10.1"),
    ]
    assert batch == plan_orders(target, snapshot, state, assumptions)
    assert len({order.order_id for order in batch.orders}) == 3


def test_planner_scales_and_rounds_buys_for_fees_and_cash_buffer() -> None:
    snapshot = _snapshot(_quote("AAA", bid="10", ask="10", last="10"))
    assumptions = ExecutionAssumptions(
        minimum_fee="1",
        quantity_step="1",
    )
    batch = plan_orders(
        _target(AAA="1"),
        snapshot,
        _state(),
        assumptions,
        config=PlanningConfig(cash_buffer="5"),
    )

    assert len(batch.orders) == 1
    assert batch.orders[0].quantity == Decimal("9")
    event = simulate_shadow_batch(batch, snapshot, assumptions, sequence=1)
    ending_cash = Decimal("100") - event.fills[0].notional - event.fills[0].fee
    assert ending_cash == Decimal("9")
    assert ending_cash >= Decimal("5")


def test_planner_preserves_target_cash_remainder_under_spread_cost() -> None:
    snapshot = _snapshot(_quote("AAA", bid="9.9", ask="10.1", last="10"))
    assumptions = ExecutionAssumptions(quantity_step="0.1")
    target = _target(AAA="0.9")

    batch = plan_orders(target, snapshot, _state(), assumptions)
    event = simulate_shadow_batch(batch, snapshot, assumptions, sequence=1)
    ending_cash = Decimal("100") - event.fills[0].notional

    assert batch.orders[0].quantity == Decimal("8.9")
    assert ending_cash == Decimal("10.11")
    assert ending_cash >= Decimal("100") * target.cash_weight


def test_planning_and_simulation_ignore_callers_decimal_context() -> None:
    snapshot = _snapshot(_quote("AAA", bid="3.21", ask="3.23", last="3.22"))
    target = _target(AAA="0.731234567890123456")
    assumptions = ExecutionAssumptions(
        slippage_bps="7.5",
        commission_bps="3.25",
        quantity_step="0.000001",
    )
    normal_batch = plan_orders(target, snapshot, _state(), assumptions)
    normal_event = simulate_shadow_batch(normal_batch, snapshot, assumptions, sequence=1)

    with localcontext() as context:
        context.prec = 6
        constrained_batch = plan_orders(target, snapshot, _state(), assumptions)
        constrained_event = simulate_shadow_batch(
            constrained_batch,
            snapshot,
            assumptions,
            sequence=1,
        )

    assert constrained_batch == normal_batch
    assert constrained_event == normal_event


def test_planner_omits_zero_and_dust_orders() -> None:
    snapshot = _snapshot(
        _quote("AAA", bid="10", ask="10", last="10"),
        _quote("BBB", bid="10", ask="10", last="10"),
    )
    batch = plan_orders(
        _target(AAA="0", BBB="0.01"),
        snapshot,
        _state(positions={"AAA": "0.1"}, marks={"AAA": "10"}),
        ExecutionAssumptions(quantity_step="0.01"),
        config=PlanningConfig(minimum_trade_notional="2"),
    )

    assert batch.orders == ()


def test_planner_requires_complete_fresh_snapshot_and_nonnegative_sell_cash() -> None:
    state = _state(cash="0.10", positions={"AAA": "0.01"}, marks={"AAA": "10"})
    with pytest.raises(PaperPlanningError, match="missing held or target"):
        plan_orders(
            _target(),
            _snapshot(_quote("BBB", bid="10", ask="10", last="10")),
            state,
            ExecutionAssumptions(),
        )

    with pytest.raises(PaperPlanningError, match="negative cash"):
        plan_orders(
            _target(),
            _snapshot(_quote("AAA", bid="10", ask="10", last="10")),
            state,
            ExecutionAssumptions(minimum_fee="1", quantity_step="0.01"),
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PlanningConfig(cash_buffer=1.0),
        lambda: SimulationConfig(price_tick=0.01),
        lambda: ShadowRiskLimits(allowed_symbols=("AAA",), max_batch_notional=100.0),
    ],
)
def test_execution_configs_reject_binary_float_money(factory) -> None:
    with pytest.raises(ValidationError, match="binary floats are not accepted"):
        factory()


def test_risk_gate_approval_contains_exact_reviewed_event_and_fresh_gross() -> None:
    snapshot = _snapshot(_quote("AAA", bid="9.99", ask="10.01", last="10"))
    target = _target(AAA="0.5")
    assumptions = ExecutionAssumptions(
        slippage_bps="10",
        commission_bps="10",
        quantity_step="0.001",
    )
    state = _state()
    batch = plan_orders(target, snapshot, state, assumptions)
    limits = ShadowRiskLimits(
        allowed_symbols=("AAA",),
        max_order_notional="100",
        max_batch_notional="100",
        max_gross_exposure="100",
    )

    decision = evaluate_shadow_risk(
        batch,
        target,
        snapshot,
        state,
        assumptions,
        mode="shadow",
        evaluated_at=T0 + timedelta(seconds=1),
        limits=limits,
    )

    assert isinstance(decision, ShadowRiskApproved)
    assert decision.projected_gross_exposure == batch.orders[0].quantity * Decimal("10")
    assert decision.projected_cash == (
        Decimal("100") - decision.event.fills[0].notional - decision.event.fills[0].fee
    )
    assert decision.snapshot_id == snapshot.snapshot_id
    assert decision.simulation == limits.simulation
    assert state.cash == Decimal("100.00")  # gate is side-effect free
    ledger = ShadowLedger()
    assert ledger.execute(decision.event).cash == decision.projected_cash


def test_risk_gate_rejects_mode_staleness_future_and_naive_clock() -> None:
    snapshot = _snapshot(_quote("AAA", bid="10", ask="10", last="10"))
    target = _target(AAA="0.1")
    batch = _batch(_order("AAA", "buy", "1", "10"))
    limits = ShadowRiskLimits(allowed_symbols=("AAA",), max_snapshot_age_seconds=5)

    stale = evaluate_shadow_risk(
        batch,
        target,
        snapshot,
        _state(),
        ExecutionAssumptions(),
        mode="broker",
        evaluated_at=T0 + timedelta(seconds=6),
        limits=limits,
    )
    assert isinstance(stale, ShadowRiskRejected)
    assert {"mode_not_shadow", "stale_snapshot"} <= _codes(stale)

    future = evaluate_shadow_risk(
        batch,
        target,
        snapshot,
        _state(),
        ExecutionAssumptions(),
        mode="shadow",
        evaluated_at=T0 - timedelta(seconds=1),
        limits=limits,
    )
    assert isinstance(future, ShadowRiskRejected)
    assert "snapshot_from_future" in _codes(future)

    naive = evaluate_shadow_risk(
        batch,
        target,
        snapshot,
        _state(),
        ExecutionAssumptions(),
        mode="shadow",
        evaluated_at=datetime(2026, 1, 2, 21),
        limits=limits,
    )
    assert isinstance(naive, ShadowRiskRejected)
    assert naive.evaluated_at is None
    assert "invalid_evaluation_time" in _codes(naive)


def test_risk_gate_collects_symbol_order_notional_and_exposure_limits() -> None:
    snapshot = _snapshot(
        _quote("AAA", bid="10", ask="10", last="10"),
        _quote("BBB", bid="10", ask="10", last="10"),
    )
    target = _target(AAA="0.2", BBB="0.2")
    # Buy before sell is deliberately noncanonical.
    batch = _batch(
        _order("BBB", "buy", "2", "10"),
        _order("AAA", "sell", "1", "10"),
    )
    decision = evaluate_shadow_risk(
        batch,
        target,
        snapshot,
        _state(cash="100", positions={"AAA": "2"}, marks={"AAA": "10"}),
        ExecutionAssumptions(),
        mode="shadow",
        evaluated_at=T0,
        limits=ShadowRiskLimits(
            allowed_symbols=("AAA",),
            max_orders=1,
            max_order_notional="15",
            max_batch_notional="25",
            max_gross_exposure="15",
        ),
    )

    assert isinstance(decision, ShadowRiskRejected)
    assert {
        "symbol_not_allowed",
        "too_many_orders",
        "noncanonical_order",
        "order_notional_limit",
        "batch_notional_limit",
        "gross_exposure_limit",
    } <= _codes(decision)


def test_risk_gate_rejects_shorts_cash_and_identifier_reuse_atomically() -> None:
    snapshot = _snapshot(
        _quote("AAA", bid="10", ask="10", last="10"),
        _quote("BBB", bid="60", ask="60", last="60"),
    )
    target = _target(AAA="0", BBB="1")
    batch = _batch(
        _order("AAA", "sell", "2", "10", order_id="used-order"),
        _order("BBB", "buy", "2", "60", order_id="new-order"),
        batch_id="used-batch",
    )
    state = _state(
        cash="1",
        positions={"AAA": "1"},
        marks={"AAA": "10"},
        sequence=4,
        as_of=T0,
        batch_ids=("used-batch",),
        order_ids=("used-order",),
    )
    decision = evaluate_shadow_risk(
        batch,
        target,
        snapshot,
        state,
        ExecutionAssumptions(),
        mode="shadow",
        evaluated_at=T0,
        limits=ShadowRiskLimits(
            allowed_symbols=("AAA", "BBB"),
            max_order_notional="1000",
            max_batch_notional="1000",
            max_gross_exposure="1000",
        ),
    )

    assert isinstance(decision, ShadowRiskRejected)
    assert {
        "batch_id_reused",
        "order_id_reused",
        "short_position",
        "insufficient_cash",
    } <= _codes(decision)
    assert state.cash == Decimal("1")
    assert state.positions == {"AAA": Decimal("1")}


def test_risk_gate_requires_matching_target_time_allocation_and_fresh_held_marks() -> None:
    later = T0 + timedelta(minutes=1)
    snapshot = _snapshot(
        _quote("AAA", bid="10", ask="10", last="10", at=later),
        at=later,
    )
    batch = _batch()
    state = _state(
        cash="90",
        positions={"ZZZ": "1"},
        marks={"ZZZ": "10"},
        as_of=T0,
    )
    decision = evaluate_shadow_risk(
        batch,
        _target(),
        snapshot,
        state,
        ExecutionAssumptions(),
        mode="shadow",
        evaluated_at=later,
        limits=ShadowRiskLimits(allowed_symbols=("AAA",)),
    )

    assert isinstance(decision, ShadowRiskRejected)
    assert {"timestamp_mismatch", "missing_fresh_mark"} <= _codes(decision)
