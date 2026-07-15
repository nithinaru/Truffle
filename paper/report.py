"""Frozen, deterministic reports for local multi-arm paper replays."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from paper.ideal import IDEAL_SEED_NAV
from paper.ledger import SHADOW_SEED_CASH
from paper.models import (
    DerivedDecimal,
    ExecutionAssumptions,
    NonNegativeDecimal,
    NonNegativeDerivedDecimal,
    PaperModel,
    PositiveDecimal,
    PositiveDerivedDecimal,
    UtcDatetime,
    WeightDecimal,
    decimal_context,
    freeze_decimal_map,
)
from paper.planner import PlanningConfig
from paper.risk import ShadowRiskLimits

type PaperArm = Literal["truffle", "equal_weight", "market", "cash"]

PAPER_ARM_ORDER: tuple[PaperArm, ...] = (
    "truffle",
    "equal_weight",
    "market",
    "cash",
)


class PaperCurvePoint(PaperModel):
    """Post-step exact and ideal values for one arm at one market snapshot."""

    arm: PaperArm
    snapshot_id: str
    as_of: UtcDatetime
    exact_equity: NonNegativeDerivedDecimal
    exact_cash: NonNegativeDerivedDecimal
    exact_positions: dict[str, NonNegativeDerivedDecimal]
    exact_cumulative_fees: NonNegativeDerivedDecimal
    ideal_nav: PositiveDerivedDecimal
    ideal_cash: NonNegativeDerivedDecimal
    ideal_quantities: dict[str, NonNegativeDerivedDecimal]

    @field_validator("exact_positions", "ideal_quantities", mode="after")
    @classmethod
    def _ordered_positions(cls, value: dict[str, Decimal]) -> dict[str, Decimal]:
        return freeze_decimal_map(value)


class PaperFillRecord(PaperModel):
    """One hand-auditable simulated fill in canonical batch order."""

    order_id: str
    fill_id: str
    ticker: str
    side: Literal["buy", "sell"]
    quantity: PositiveDecimal
    price: PositiveDecimal
    notional: PositiveDerivedDecimal
    fee: NonNegativeDecimal


class PaperTradeRecord(PaperModel):
    """One scheduled arm rebalance, including an intentionally empty batch."""

    arm: Literal["truffle", "equal_weight", "market"]
    snapshot_id: str
    as_of: UtcDatetime
    evaluated_at: UtcDatetime
    allocation_id: str
    batch_id: str
    order_ids: tuple[str, ...]
    fill_ids: tuple[str, ...]
    fills: tuple[PaperFillRecord, ...]
    target_weights: dict[str, WeightDecimal]
    pretrade_equity: PositiveDerivedDecimal
    executed_notional: NonNegativeDerivedDecimal
    turnover: NonNegativeDerivedDecimal
    fees: NonNegativeDerivedDecimal
    posttrade_equity: NonNegativeDerivedDecimal
    cash: NonNegativeDerivedDecimal
    positions: dict[str, NonNegativeDerivedDecimal]

    @field_validator("positions", "target_weights", mode="after")
    @classmethod
    def _ordered_positions(cls, value: dict[str, Decimal]) -> dict[str, Decimal]:
        return freeze_decimal_map(value)

    @model_validator(mode="after")
    def _consistent_batch_totals(self) -> Self:
        if self.order_ids != tuple(fill.order_id for fill in self.fills):
            raise ValueError("trade order IDs must match fills in canonical order")
        if self.fill_ids != tuple(fill.fill_id for fill in self.fills):
            raise ValueError("trade fill IDs must match fills in canonical order")
        with decimal_context():
            expected_notional = sum(
                (fill.notional for fill in self.fills),
                Decimal("0"),
            )
            expected_fees = sum((fill.fee for fill in self.fills), Decimal("0"))
            expected_turnover = (
                Decimal("0")
                if expected_notional == Decimal("0")
                else expected_notional / self.pretrade_equity
            )
        if self.executed_notional != expected_notional:
            raise ValueError("executed notional must equal the exact fill total")
        if self.fees != expected_fees:
            raise ValueError("trade fees must equal the exact fill total")
        if self.turnover != expected_turnover:
            raise ValueError("turnover must equal notional divided by pretrade equity")
        return self


class PaperArmSummary(PaperModel):
    """Terminal and aggregate results for one fully isolated replay arm."""

    arm: PaperArm
    initial_exact_equity: PositiveDerivedDecimal
    final_exact_equity: NonNegativeDerivedDecimal
    exact_return: DerivedDecimal
    initial_ideal_nav: PositiveDerivedDecimal
    final_ideal_nav: PositiveDerivedDecimal
    ideal_return: DerivedDecimal
    exact_minus_ideal_return: DerivedDecimal
    total_executed_notional: NonNegativeDerivedDecimal
    total_turnover: NonNegativeDerivedDecimal
    total_fees: NonNegativeDerivedDecimal
    trade_count: int = Field(ge=0)


class PaperRelativeReturn(PaperModel):
    """Return difference between two named, independently run arms."""

    arm: PaperArm
    benchmark_arm: PaperArm
    exact_return_difference: DerivedDecimal
    ideal_return_difference: DerivedDecimal


class PaperExperimentReport(PaperModel):
    """Complete deterministic artifact for one offline paper replay."""

    kind: Literal["paper_replay"] = "paper_replay"
    arms: tuple[PaperArm, ...]
    investable_universe: tuple[str, ...]
    market_symbol: str
    execution_assumptions: ExecutionAssumptions
    planning_config: PlanningConfig
    risk_limits: ShadowRiskLimits
    snapshot_ids: tuple[str, ...]
    scheduled_allocation_ids: tuple[str, ...]
    curves: tuple[PaperCurvePoint, ...]
    trades: tuple[PaperTradeRecord, ...]
    summaries: tuple[PaperArmSummary, ...]
    relative_returns: tuple[PaperRelativeReturn, ...]

    @model_validator(mode="after")
    def _canonical_arm_order(self) -> Self:
        if self.arms != PAPER_ARM_ORDER:
            raise ValueError("paper experiment arms must use the canonical arm order")
        if tuple(summary.arm for summary in self.summaries) != PAPER_ARM_ORDER:
            raise ValueError("paper summaries must use the canonical arm order")
        expected_curve_arms = PAPER_ARM_ORDER * len(self.snapshot_ids)
        if tuple(point.arm for point in self.curves) != expected_curve_arms:
            raise ValueError("paper curves must be snapshot-major in canonical arm order")
        return self


def summarize_paper_experiment(
    *,
    investable_universe: tuple[str, ...],
    market_symbol: str,
    execution_assumptions: ExecutionAssumptions,
    planning_config: PlanningConfig,
    risk_limits: ShadowRiskLimits,
    snapshot_ids: tuple[str, ...],
    scheduled_allocation_ids: tuple[str, ...],
    curves: tuple[PaperCurvePoint, ...],
    trades: tuple[PaperTradeRecord, ...],
) -> PaperExperimentReport:
    """Derive summaries and stable relative returns from replay observations."""

    if not snapshot_ids:
        raise ValueError("paper report requires at least one snapshot")

    summaries: list[PaperArmSummary] = []
    for arm in PAPER_ARM_ORDER:
        arm_curves = [point for point in curves if point.arm == arm]
        if len(arm_curves) != len(snapshot_ids):
            raise ValueError(f"arm {arm!r} is missing paper curve observations")
        arm_trades = [trade for trade in trades if trade.arm == arm]
        initial_exact = SHADOW_SEED_CASH
        initial_ideal = IDEAL_SEED_NAV
        final_exact = arm_curves[-1].exact_equity
        final_ideal = arm_curves[-1].ideal_nav
        with decimal_context():
            exact_return = final_exact / initial_exact - Decimal("1")
            ideal_return = final_ideal / initial_ideal - Decimal("1")
            exact_minus_ideal = exact_return - ideal_return
            total_notional = sum(
                (trade.executed_notional for trade in arm_trades),
                Decimal("0"),
            )
            total_turnover = sum(
                (trade.turnover for trade in arm_trades),
                Decimal("0"),
            )
            total_fees = sum((trade.fees for trade in arm_trades), Decimal("0"))
        summaries.append(
            PaperArmSummary(
                arm=arm,
                initial_exact_equity=initial_exact,
                final_exact_equity=final_exact,
                exact_return=exact_return,
                initial_ideal_nav=initial_ideal,
                final_ideal_nav=final_ideal,
                ideal_return=ideal_return,
                exact_minus_ideal_return=exact_minus_ideal,
                total_executed_notional=total_notional,
                total_turnover=total_turnover,
                total_fees=total_fees,
                trade_count=len(arm_trades),
            )
        )

    by_arm = {summary.arm: summary for summary in summaries}
    relative_pairs: tuple[tuple[PaperArm, PaperArm], ...] = (
        ("truffle", "cash"),
        ("equal_weight", "cash"),
        ("market", "cash"),
        ("truffle", "market"),
        ("equal_weight", "market"),
    )
    relative_returns: list[PaperRelativeReturn] = []
    for arm, benchmark in relative_pairs:
        with decimal_context():
            exact_difference = by_arm[arm].exact_return - by_arm[benchmark].exact_return
            ideal_difference = by_arm[arm].ideal_return - by_arm[benchmark].ideal_return
        relative_returns.append(
            PaperRelativeReturn(
                arm=arm,
                benchmark_arm=benchmark,
                exact_return_difference=exact_difference,
                ideal_return_difference=ideal_difference,
            )
        )
    return PaperExperimentReport(
        arms=PAPER_ARM_ORDER,
        investable_universe=investable_universe,
        market_symbol=market_symbol,
        execution_assumptions=execution_assumptions,
        planning_config=planning_config,
        risk_limits=risk_limits,
        snapshot_ids=snapshot_ids,
        scheduled_allocation_ids=scheduled_allocation_ids,
        curves=curves,
        trades=trades,
        summaries=tuple(summaries),
        relative_returns=tuple(relative_returns),
    )


__all__ = [
    "PAPER_ARM_ORDER",
    "PaperArm",
    "PaperArmSummary",
    "PaperCurvePoint",
    "PaperExperimentReport",
    "PaperFillRecord",
    "PaperRelativeReturn",
    "PaperTradeRecord",
    "summarize_paper_experiment",
]
