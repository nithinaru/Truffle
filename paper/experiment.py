"""Deterministic four-arm paper experiment over local replay snapshots."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from typing import Literal

from paper.ideal import IdealLedger
from paper.ledger import ShadowLedger
from paper.models import (
    ExecutionAssumptions,
    MarketSnapshot,
    ShadowBatchExecuted,
    TargetAllocation,
    decimal_context,
)
from paper.planner import PaperPlanningError, PlanningConfig, plan_orders
from paper.provider import LocalReplayProvider
from paper.report import (
    PAPER_ARM_ORDER,
    PaperArm,
    PaperCurvePoint,
    PaperExperimentReport,
    PaperFillRecord,
    PaperTradeRecord,
    summarize_paper_experiment,
)
from paper.risk import (
    RiskViolation,
    ShadowRiskApproved,
    ShadowRiskLimits,
    ShadowRiskRejected,
    gate_shadow_batch,
)

type ConfirmedSchedule = Iterable[TargetAllocation] | Mapping[datetime, TargetAllocation]
type EvaluatedAtPolicy = Callable[[MarketSnapshot], datetime]
type TradedArm = Literal["truffle", "equal_weight", "market"]

_TRADED_ARM_ORDER: tuple[TradedArm, ...] = (
    "truffle",
    "equal_weight",
    "market",
)
_WEIGHT_QUANTUM = Decimal("0.000000000000000001")


class PaperExperimentError(ValueError):
    """Base class for a local paper experiment that cannot be completed."""


class PaperScheduleError(PaperExperimentError):
    """Raised when the confirmed schedule cannot map exactly to replay data."""


class PaperStepRejectedError(PaperExperimentError):
    """Raised when any arm fails the atomic risk gate at a scheduled instant."""

    def __init__(
        self,
        *,
        arm: TradedArm,
        as_of: datetime,
        violations: tuple[RiskViolation, ...],
    ) -> None:
        self.arm = arm
        self.as_of = as_of
        self.violations = violations
        codes = ", ".join(violation.code for violation in violations)
        super().__init__(f"paper step rejected for {arm} at {as_of.isoformat()}: {codes}")


def _canonical_symbol(value: str, *, label: str) -> str:
    symbol = value.strip().upper()
    if not symbol:
        raise PaperScheduleError(f"{label} must not contain empty symbols")
    return symbol


def _canonical_universe(symbols: Iterable[str]) -> tuple[str, ...]:
    canonical = tuple(
        sorted(_canonical_symbol(symbol, label="investable universe") for symbol in symbols)
    )
    if not canonical:
        raise PaperScheduleError("investable universe must not be empty")
    if len(canonical) != len(set(canonical)):
        raise PaperScheduleError("investable universe contains duplicates after canonicalization")
    return canonical


def _utc_schedule_key(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise PaperScheduleError("schedule keys must be datetime instances")
    if value.tzinfo is None or value.utcoffset() is None:
        raise PaperScheduleError("schedule timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _materialize_schedule(schedule: ConfirmedSchedule) -> tuple[TargetAllocation, ...]:
    if isinstance(schedule, Mapping):
        allocations: list[TargetAllocation] = []
        for timestamp, allocation in schedule.items():
            if not isinstance(allocation, TargetAllocation):
                raise PaperScheduleError(
                    "confirmed schedule values must be TargetAllocation objects"
                )
            if _utc_schedule_key(timestamp) != allocation.effective_at:
                raise PaperScheduleError("schedule key must exactly match allocation effective_at")
            allocations.append(allocation)
    else:
        allocations = list(schedule)
        if any(not isinstance(item, TargetAllocation) for item in allocations):
            raise PaperScheduleError("confirmed schedule entries must be TargetAllocation objects")

    allocations.sort(key=lambda allocation: (allocation.effective_at, allocation.allocation_id))
    timestamps = [allocation.effective_at for allocation in allocations]
    if len(timestamps) != len(set(timestamps)):
        raise PaperScheduleError(
            "confirmed schedule must contain at most one allocation per timestamp"
        )
    allocation_ids = [allocation.allocation_id for allocation in allocations]
    if len(allocation_ids) != len(set(allocation_ids)):
        raise PaperScheduleError("confirmed allocation IDs must be unique")
    return tuple(allocations)


def _equal_weights(universe: tuple[str, ...]) -> dict[str, Decimal]:
    with decimal_context():
        common = (Decimal("1") / Decimal(len(universe))).quantize(
            _WEIGHT_QUANTUM,
            rounding=ROUND_DOWN,
        )
        weights = {symbol: common for symbol in universe[:-1]}
        weights[universe[-1]] = Decimal("1") - common * Decimal(len(universe) - 1)
    return weights


def _derived_targets(
    source: TargetAllocation,
    universe: tuple[str, ...],
    market_symbol: str,
) -> dict[TradedArm, TargetAllocation]:
    return {
        "truffle": source,
        "equal_weight": TargetAllocation(
            allocation_id=f"equal_weight:{source.allocation_id}",
            effective_at=source.effective_at,
            weights=_equal_weights(universe),
        ),
        "market": TargetAllocation(
            allocation_id=f"market:{source.allocation_id}",
            effective_at=source.effective_at,
            weights={market_symbol: Decimal("1")},
        ),
    }


def _curve_point(
    arm: PaperArm,
    snapshot: MarketSnapshot,
    exact: ShadowLedger,
    ideal: IdealLedger,
) -> PaperCurvePoint:
    exact_state = exact.state
    ideal_state = ideal.state
    return PaperCurvePoint(
        arm=arm,
        snapshot_id=snapshot.snapshot_id,
        as_of=snapshot.as_of,
        exact_equity=exact_state.equity,
        exact_cash=exact_state.cash,
        exact_positions=dict(exact_state.positions),
        exact_cumulative_fees=exact_state.cumulative_fees,
        ideal_nav=ideal_state.nav,
        ideal_cash=ideal_state.cash,
        ideal_quantities=dict(ideal_state.quantities),
    )


def _trade_record(
    *,
    arm: TradedArm,
    snapshot: MarketSnapshot,
    target: TargetAllocation,
    approval: ShadowRiskApproved,
    pretrade_equity: Decimal,
    exact: ShadowLedger,
) -> PaperTradeRecord:
    event = approval.event
    fills = tuple(
        PaperFillRecord(
            order_id=fill.order_id,
            fill_id=fill.fill_id,
            ticker=fill.ticker,
            side=fill.side,
            quantity=fill.quantity,
            price=fill.price,
            notional=fill.notional,
            fee=fill.fee,
        )
        for fill in event.fills
    )
    with decimal_context():
        executed_notional = sum(
            (fill.notional for fill in event.fills),
            Decimal("0"),
        )
        fees = sum((fill.fee for fill in event.fills), Decimal("0"))
        turnover = (
            Decimal("0")
            if executed_notional == Decimal("0")
            else executed_notional / pretrade_equity
        )
    state = exact.state
    return PaperTradeRecord(
        arm=arm,
        snapshot_id=snapshot.snapshot_id,
        as_of=snapshot.as_of,
        evaluated_at=approval.evaluated_at,
        allocation_id=target.allocation_id,
        batch_id=event.batch_id,
        order_ids=event.order_ids,
        fill_ids=tuple(fill.fill_id for fill in event.fills),
        fills=fills,
        target_weights=dict(target.weights),
        pretrade_equity=pretrade_equity,
        executed_notional=executed_notional,
        turnover=turnover,
        fees=fees,
        posttrade_equity=state.equity,
        cash=state.cash,
        positions=dict(state.positions),
    )


def _check_schedule_scope(
    *,
    allocations: tuple[TargetAllocation, ...],
    provider: LocalReplayProvider,
    universe: tuple[str, ...],
    market_symbol: str,
) -> None:
    provider_symbols = set(provider.symbols)
    required_provider_symbols = set(universe) | {market_symbol}
    missing_provider_symbols = sorted(required_provider_symbols - provider_symbols)
    if missing_provider_symbols:
        raise PaperScheduleError(
            f"replay provider is missing experiment symbols: {missing_provider_symbols}"
        )

    replay_times = {snapshot.as_of for snapshot in provider.snapshots}
    universe_set = set(universe)
    for allocation in allocations:
        if allocation.effective_at not in replay_times:
            raise PaperScheduleError(
                "confirmed allocation timestamp has no exact replay snapshot: "
                f"{allocation.effective_at.isoformat()}"
            )
        outside = sorted(set(allocation.weights) - universe_set)
        if outside:
            raise PaperScheduleError(
                f"allocation {allocation.allocation_id!r} contains symbols outside "
                f"the investable universe: {outside}"
            )


def run_paper_replay(
    provider: LocalReplayProvider,
    confirmed_schedule: ConfirmedSchedule,
    *,
    investable_universe: Iterable[str],
    market_symbol: str,
    assumptions: ExecutionAssumptions,
    planning: PlanningConfig,
    risk: ShadowRiskLimits,
    evaluated_at_policy: EvaluatedAtPolicy,
) -> PaperExperimentReport:
    """Replay four independent arms with no clock, randomness, or network.

    ``confirmed_schedule`` is a trusted compiler output: this function never
    invokes an LLM or silently changes a Truffle target.  Every scheduled
    timestamp must exactly match a provider snapshot.  A snapshot is applied
    as a fresh mark to every ledger before any planning; all three prospective
    batches pass the atomic shadow gate before any batch is executed.  One
    rejection aborts the dated step and the experiment rather than skipping it.
    """

    universe = _canonical_universe(investable_universe)
    canonical_market = _canonical_symbol(market_symbol, label="market symbol")
    allocations = _materialize_schedule(confirmed_schedule)
    _check_schedule_scope(
        allocations=allocations,
        provider=provider,
        universe=universe,
        market_symbol=canonical_market,
    )
    schedule_by_time = {allocation.effective_at: allocation for allocation in allocations}

    exact_ledgers = {arm: ShadowLedger() for arm in PAPER_ARM_ORDER}
    ideal_ledgers = {arm: IdealLedger() for arm in PAPER_ARM_ORDER}
    curves: list[PaperCurvePoint] = []
    trades: list[PaperTradeRecord] = []

    for snapshot in provider.snapshots:
        for arm in PAPER_ARM_ORDER:
            exact_ledgers[arm].mark(snapshot)
            ideal_ledgers[arm].mark(snapshot)

        source_target = schedule_by_time.get(snapshot.as_of)
        if source_target is not None:
            targets = _derived_targets(
                source_target,
                universe,
                canonical_market,
            )
            try:
                evaluated_at = evaluated_at_policy(snapshot)
            except Exception as exc:
                raise PaperExperimentError(
                    f"evaluated_at policy failed at {snapshot.as_of.isoformat()}: {exc}"
                ) from exc

            approvals: dict[TradedArm, ShadowRiskApproved] = {}
            pretrade_equities: dict[TradedArm, Decimal] = {}
            for arm in _TRADED_ARM_ORDER:
                target = targets[arm]
                ledger_state = exact_ledgers[arm].state
                pretrade_equities[arm] = ledger_state.equity
                try:
                    batch = plan_orders(
                        target,
                        snapshot,
                        ledger_state,
                        assumptions,
                        config=planning,
                    )
                except PaperPlanningError as exc:
                    raise PaperExperimentError(
                        f"paper planning failed for {arm} at {snapshot.as_of.isoformat()}: {exc}"
                    ) from exc
                decision = gate_shadow_batch(
                    batch,
                    target,
                    snapshot,
                    ledger_state,
                    assumptions,
                    mode="shadow",
                    evaluated_at=evaluated_at,
                    limits=risk,
                )
                if isinstance(decision, ShadowRiskRejected):
                    raise PaperStepRejectedError(
                        arm=arm,
                        as_of=snapshot.as_of,
                        violations=decision.violations,
                    )
                approvals[arm] = decision

            for arm in _TRADED_ARM_ORDER:
                target = targets[arm]
                event: ShadowBatchExecuted = approvals[arm].event
                exact_ledgers[arm].execute(event)
                ideal_ledgers[arm].apply(target, snapshot)
                trades.append(
                    _trade_record(
                        arm=arm,
                        snapshot=snapshot,
                        target=target,
                        approval=approvals[arm],
                        pretrade_equity=pretrade_equities[arm],
                        exact=exact_ledgers[arm],
                    )
                )

        for arm in PAPER_ARM_ORDER:
            curves.append(
                _curve_point(
                    arm,
                    snapshot,
                    exact_ledgers[arm],
                    ideal_ledgers[arm],
                )
            )

    return summarize_paper_experiment(
        investable_universe=universe,
        market_symbol=canonical_market,
        execution_assumptions=assumptions,
        planning_config=planning,
        risk_limits=risk,
        snapshot_ids=tuple(snapshot.snapshot_id for snapshot in provider.snapshots),
        scheduled_allocation_ids=tuple(allocation.allocation_id for allocation in allocations),
        curves=tuple(curves),
        trades=tuple(trades),
    )


run_paper_experiment = run_paper_replay


__all__ = [
    "ConfirmedSchedule",
    "EvaluatedAtPolicy",
    "PaperExperimentError",
    "PaperScheduleError",
    "PaperStepRejectedError",
    "run_paper_experiment",
    "run_paper_replay",
]
