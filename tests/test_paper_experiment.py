"""Hand-accounted tests for the deterministic multi-arm paper replay."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from paper.experiment import (
    PaperScheduleError,
    PaperStepRejectedError,
    run_paper_replay,
)
from paper.ledger import ShadowLedger
from paper.models import (
    ExecutionAssumptions,
    MarketSnapshot,
    Quote,
    SimulationConfig,
    TargetAllocation,
    decimal_context,
)
from paper.planner import PlanningConfig
from paper.provider import LocalReplayProvider
from paper.risk import ShadowRiskLimits


def _snapshot(
    day: int,
    prices: dict[str, str],
    *,
    spread: str = "0.1",
) -> MarketSnapshot:
    as_of = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=day)
    half_spread = Decimal(spread)
    return MarketSnapshot(
        snapshot_id=f"snapshot-{day}",
        as_of=as_of,
        quotes=tuple(
            Quote(
                ticker=ticker,
                bid=Decimal(price) - half_spread,
                ask=Decimal(price) + half_spread,
                last=price,
                as_of=as_of,
            )
            for ticker, price in prices.items()
        ),
    )


@pytest.fixture
def replay() -> LocalReplayProvider:
    return LocalReplayProvider(
        (
            _snapshot(0, {"A": "10", "B": "10", "M": "10"}),
            _snapshot(1, {"A": "12", "B": "10", "M": "11"}),
            _snapshot(2, {"A": "12", "B": "11", "M": "12"}),
        )
    )


@pytest.fixture
def schedule(replay: LocalReplayProvider) -> tuple[TargetAllocation, ...]:
    return (
        TargetAllocation(
            allocation_id="confirmed-a",
            effective_at=replay.snapshots[0].as_of,
            weights={"A": "1"},
        ),
        TargetAllocation(
            allocation_id="confirmed-b",
            effective_at=replay.snapshots[1].as_of,
            weights={"B": "1"},
        ),
    )


@pytest.fixture
def simulation() -> SimulationConfig:
    return SimulationConfig(price_tick="0.01", fee_tick="0.01")


@pytest.fixture
def assumptions() -> ExecutionAssumptions:
    return ExecutionAssumptions(
        slippage_bps="0",
        commission_bps="0",
        minimum_fee="0.10",
        quantity_step="0.1",
    )


def _planning(simulation: SimulationConfig) -> PlanningConfig:
    return PlanningConfig(simulation=simulation)


def _risk(
    simulation: SimulationConfig,
    *,
    allowed_symbols: tuple[str, ...] = ("A", "B", "M"),
    max_orders: int = 10,
) -> ShadowRiskLimits:
    return ShadowRiskLimits(
        allowed_symbols=allowed_symbols,
        max_snapshot_age_seconds=0,
        max_orders=max_orders,
        max_order_notional="1000",
        max_batch_notional="1000",
        max_gross_exposure="1000",
        simulation=simulation,
    )


def _run(
    replay: LocalReplayProvider,
    schedule: tuple[TargetAllocation, ...],
    simulation: SimulationConfig,
    assumptions: ExecutionAssumptions,
):
    return run_paper_replay(
        replay,
        schedule,
        investable_universe=("A", "B"),
        market_symbol="M",
        assumptions=assumptions,
        planning=_planning(simulation),
        risk=_risk(simulation),
        evaluated_at_policy=lambda snapshot: snapshot.as_of,
    )


def test_four_independent_arms_start_with_exact_100_and_unit_ideal(
    replay: LocalReplayProvider,
    schedule: tuple[TargetAllocation, ...],
    simulation: SimulationConfig,
    assumptions: ExecutionAssumptions,
) -> None:
    report = _run(replay, schedule, simulation, assumptions)

    assert report.arms == ("truffle", "equal_weight", "market", "cash")
    assert len(report.curves) == len(replay.snapshots) * 4
    assert [summary.initial_exact_equity for summary in report.summaries] == [Decimal("100.00")] * 4
    assert [summary.initial_ideal_nav for summary in report.summaries] == [Decimal("1")] * 4

    first = {point.arm: point for point in report.curves[:4]}
    assert first["truffle"].exact_positions == {"A": Decimal("9.8")}
    assert first["equal_weight"].exact_positions == {
        "A": Decimal("4.9"),
        "B": Decimal("4.9"),
    }
    assert first["market"].exact_positions == {"M": Decimal("9.8")}
    assert first["cash"].exact_positions == {}
    assert first["cash"].exact_equity == Decimal("100.00")
    assert first["cash"].ideal_nav == Decimal("1")


def test_sell_before_buy_round_trip_and_full_notional_turnover(
    replay: LocalReplayProvider,
    schedule: tuple[TargetAllocation, ...],
    simulation: SimulationConfig,
    assumptions: ExecutionAssumptions,
) -> None:
    report = _run(replay, schedule, simulation, assumptions)
    trade = next(
        record
        for record in report.trades
        if record.arm == "truffle" and record.snapshot_id == "snapshot-1"
    )

    assert [(fill.side, fill.ticker) for fill in trade.fills] == [
        ("sell", "A"),
        ("buy", "B"),
    ]
    assert trade.pretrade_equity == Decimal("118.52")
    assert trade.executed_notional == Decimal("233.78")
    assert trade.fees == Decimal("0.20")
    assert trade.cash == Decimal("0.18")
    assert trade.positions == {"B": Decimal("11.6")}
    with decimal_context():
        expected_turnover = trade.executed_notional / trade.pretrade_equity
    assert trade.turnover == expected_turnover
    assert trade.turnover != expected_turnover / 2


def test_fees_and_spread_drag_exact_account_below_frictionless_ideal(
    replay: LocalReplayProvider,
    schedule: tuple[TargetAllocation, ...],
    simulation: SimulationConfig,
    assumptions: ExecutionAssumptions,
) -> None:
    report = _run(replay, schedule, simulation, assumptions)
    first_truffle = report.curves[0]

    assert first_truffle.exact_equity == Decimal("98.92")
    assert first_truffle.ideal_nav == Decimal("1")
    truffle = report.summaries[0]
    assert truffle.final_exact_equity == Decimal("127.78")
    assert truffle.final_ideal_nav == Decimal("1.32")
    assert truffle.total_fees == Decimal("0.30")
    assert truffle.exact_return == Decimal("0.2778")
    assert truffle.ideal_return == Decimal("0.32")
    assert truffle.exact_minus_ideal_return == Decimal("-0.0422")


def test_cash_baseline_stays_untouched_and_relative_returns_are_explicit(
    replay: LocalReplayProvider,
    schedule: tuple[TargetAllocation, ...],
    simulation: SimulationConfig,
    assumptions: ExecutionAssumptions,
) -> None:
    report = _run(replay, schedule, simulation, assumptions)
    cash_curves = [point for point in report.curves if point.arm == "cash"]
    cash = report.summaries[-1]

    assert all(point.exact_equity == Decimal("100.00") for point in cash_curves)
    assert all(point.exact_cash == Decimal("100.00") for point in cash_curves)
    assert all(point.exact_positions == {} for point in cash_curves)
    assert all(point.ideal_nav == Decimal("1") for point in cash_curves)
    assert all(point.ideal_cash == Decimal("1") for point in cash_curves)
    assert cash.trade_count == 0
    assert cash.exact_return == Decimal("0")
    assert report.relative_returns[0].arm == "truffle"
    assert report.relative_returns[0].benchmark_arm == "cash"
    assert report.relative_returns[0].exact_return_difference == Decimal("0.2778")


def test_arm_state_is_isolated_and_report_maps_are_immutable(
    replay: LocalReplayProvider,
    schedule: tuple[TargetAllocation, ...],
    simulation: SimulationConfig,
    assumptions: ExecutionAssumptions,
) -> None:
    report = _run(replay, schedule, simulation, assumptions)
    second = {point.arm: point for point in report.curves[4:8]}

    assert second["truffle"].exact_positions == {"B": Decimal("11.6")}
    assert set(second["equal_weight"].exact_positions) == {"A", "B"}
    assert set(second["market"].exact_positions) == {"M"}
    assert second["cash"].exact_positions == {}
    with pytest.raises(TypeError, match="cannot be mutated"):
        second["truffle"].exact_positions["A"] = Decimal("999")
    assert "A" in second["equal_weight"].exact_positions


def test_evaluated_at_policy_runs_once_per_dated_step_and_empty_batches_report(
    replay: LocalReplayProvider,
    schedule: tuple[TargetAllocation, ...],
    simulation: SimulationConfig,
    assumptions: ExecutionAssumptions,
) -> None:
    calls: list[str] = []

    def evaluated_at(snapshot: MarketSnapshot) -> datetime:
        calls.append(snapshot.snapshot_id)
        return snapshot.as_of

    report = run_paper_replay(
        replay,
        schedule,
        investable_universe=("A", "B"),
        market_symbol="M",
        assumptions=assumptions,
        planning=_planning(simulation),
        risk=_risk(simulation),
        evaluated_at_policy=evaluated_at,
    )

    assert calls == ["snapshot-0", "snapshot-1"]
    empty_market = next(
        trade
        for trade in report.trades
        if trade.arm == "market" and trade.snapshot_id == "snapshot-1"
    )
    assert empty_market.fills == ()
    assert empty_market.executed_notional == Decimal("0")
    assert empty_market.turnover == Decimal("0")


def test_report_json_and_content_ids_are_deterministic_for_input_order(
    replay: LocalReplayProvider,
    schedule: tuple[TargetAllocation, ...],
    simulation: SimulationConfig,
    assumptions: ExecutionAssumptions,
) -> None:
    first = _run(replay, schedule, simulation, assumptions)
    second = run_paper_replay(
        replay,
        tuple(reversed(schedule)),
        investable_universe=("B", "A"),
        market_symbol="m",
        assumptions=assumptions,
        planning=_planning(simulation),
        risk=_risk(simulation),
        evaluated_at_policy=lambda snapshot: snapshot.as_of,
    )

    assert first.canonical_json() == second.canonical_json()
    assert first.execution_assumptions == assumptions
    assert first.planning_config == _planning(simulation)
    assert first.risk_limits == _risk(simulation)
    assert first.trades[0].evaluated_at == first.trades[0].as_of
    assert first.trades[0].target_weights == {"A": Decimal("1")}
    assert [trade.batch_id for trade in first.trades] == [trade.batch_id for trade in second.trades]
    assert all(trade.batch_id.startswith("batch_") for trade in first.trades)
    assert all(fill.fill_id.startswith("fill_") for trade in first.trades for fill in trade.fills)


@pytest.mark.parametrize(
    ("bad_schedule", "message"),
    [
        (
            lambda replay: (
                TargetAllocation(
                    allocation_id="missing-time",
                    effective_at=replay.snapshots[-1].as_of + timedelta(days=1),
                    weights={"A": "1"},
                ),
            ),
            "no exact replay snapshot",
        ),
        (
            lambda replay: (
                TargetAllocation(
                    allocation_id="duplicate-1",
                    effective_at=replay.snapshots[0].as_of,
                    weights={"A": "1"},
                ),
                TargetAllocation(
                    allocation_id="duplicate-2",
                    effective_at=replay.snapshots[0].as_of,
                    weights={"B": "1"},
                ),
            ),
            "at most one allocation",
        ),
        (
            lambda replay: (
                TargetAllocation(
                    allocation_id="outside-universe",
                    effective_at=replay.snapshots[0].as_of,
                    weights={"M": "1"},
                ),
            ),
            "outside the investable universe",
        ),
    ],
)
def test_schedule_must_exactly_match_replay_and_universe(
    replay: LocalReplayProvider,
    simulation: SimulationConfig,
    assumptions: ExecutionAssumptions,
    bad_schedule,
    message: str,
) -> None:
    with pytest.raises(PaperScheduleError, match=message):
        run_paper_replay(
            replay,
            bad_schedule(replay),
            investable_universe=("A", "B"),
            market_symbol="M",
            assumptions=assumptions,
            planning=_planning(simulation),
            risk=_risk(simulation),
            evaluated_at_policy=lambda snapshot: snapshot.as_of,
        )


def test_mapping_schedule_key_must_equal_allocation_timestamp(
    replay: LocalReplayProvider,
    simulation: SimulationConfig,
    assumptions: ExecutionAssumptions,
) -> None:
    target = TargetAllocation(
        allocation_id="confirmed",
        effective_at=replay.snapshots[0].as_of,
        weights={"A": "1"},
    )
    with pytest.raises(PaperScheduleError, match="exactly match"):
        run_paper_replay(
            replay,
            {replay.snapshots[1].as_of: target},
            investable_universe=("A", "B"),
            market_symbol="M",
            assumptions=assumptions,
            planning=_planning(simulation),
            risk=_risk(simulation),
            evaluated_at_policy=lambda snapshot: snapshot.as_of,
        )


def test_rejection_aborts_step_before_any_arm_executes(
    monkeypatch: pytest.MonkeyPatch,
    replay: LocalReplayProvider,
    simulation: SimulationConfig,
    assumptions: ExecutionAssumptions,
) -> None:
    target = TargetAllocation(
        allocation_id="confirmed",
        effective_at=replay.snapshots[0].as_of,
        weights={"A": "1"},
    )
    executed: list[str] = []
    original_execute = ShadowLedger.execute

    def tracked_execute(self: ShadowLedger, event):
        executed.append(event.batch_id)
        return original_execute(self, event)

    monkeypatch.setattr(ShadowLedger, "execute", tracked_execute)
    with pytest.raises(PaperStepRejectedError) as raised:
        run_paper_replay(
            replay,
            (target,),
            investable_universe=("A", "B"),
            market_symbol="M",
            assumptions=assumptions,
            planning=_planning(simulation),
            # Truffle/A is prospectively approved, then equal-weight/B is rejected.
            risk=_risk(simulation, allowed_symbols=("A", "M")),
            evaluated_at_policy=lambda snapshot: snapshot.as_of,
        )

    assert raised.value.arm == "equal_weight"
    assert [violation.code for violation in raised.value.violations] == ["symbol_not_allowed"]
    assert executed == []


def test_stale_evaluation_is_a_dated_hard_failure(
    replay: LocalReplayProvider,
    simulation: SimulationConfig,
    assumptions: ExecutionAssumptions,
) -> None:
    target = TargetAllocation(
        allocation_id="confirmed",
        effective_at=replay.snapshots[0].as_of,
        weights={"A": "1"},
    )
    with pytest.raises(PaperStepRejectedError) as raised:
        run_paper_replay(
            replay,
            (target,),
            investable_universe=("A", "B"),
            market_symbol="M",
            assumptions=assumptions,
            planning=_planning(simulation),
            risk=_risk(simulation),
            evaluated_at_policy=lambda snapshot: snapshot.as_of + timedelta(seconds=1),
        )

    assert raised.value.arm == "truffle"
    assert "stale_snapshot" in {violation.code for violation in raised.value.violations}
