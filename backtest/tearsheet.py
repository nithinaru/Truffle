"""Frozen, JSON-serializable walk-forward backtest results."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backtest.config import BacktestConfig


class _ResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class CurvePoint(_ResultModel):
    """Normalized wealth curves at one observed close."""

    date: date
    strategy_net: float = Field(ge=0.0)
    strategy_gross: float = Field(ge=0.0)
    equal_weight_net: float = Field(ge=0.0)
    market: float | None = Field(default=None, ge=0.0)


class RebalanceRecord(_ResultModel):
    """One signal, its delayed fill, and subsequent risk realization."""

    signal_date: date
    fill_date: date
    training_start: date
    training_end: date
    signal_weights: dict[str, float]
    fill_pretrade_weights: dict[str, float]
    target_weights: dict[str, float]
    modeled_cvar: float
    realized_cvar: float | None = None
    holding_observations: int = Field(default=0, ge=0)
    decision_turnover: float = Field(ge=0.0)
    realized_turnover: float = Field(ge=0.0)
    transaction_cost_fraction: float = Field(ge=0.0)
    transaction_cost_paid: float = Field(ge=0.0)


class SeriesMetrics(_ResultModel):
    observations: int = Field(ge=1)
    total_return: float
    annualized_return: float
    annualized_volatility: float = Field(ge=0.0)
    sharpe: float | None
    max_drawdown: float = Field(le=0.0)
    realized_cvar: float


class BacktestSummary(_ResultModel):
    strategy: SeriesMetrics
    strategy_gross: SeriesMetrics
    equal_weight: SeriesMetrics
    market: SeriesMetrics | None = None
    total_turnover: float = Field(ge=0.0)
    annualized_turnover: float = Field(ge=0.0)
    total_cost_paid: float = Field(ge=0.0)
    cost_drag: float = Field(ge=0.0)
    holding_weighted_modeled_cvar: float | None = None
    holding_weighted_realized_cvar: float | None = None
    realized_gross_cvar: float
    realized_net_cvar: float
    resolved_execution_cost_bps: float = Field(ge=0.0)
    execution_cost_source: Literal["config_override", "spec_transaction_cost", "none"]


class Tearsheet(_ResultModel):
    """Complete deterministic output of one walk-forward experiment."""

    kind: Literal["tearsheet"] = "tearsheet"
    config: BacktestConfig
    start_date: date
    end_date: date
    curves: tuple[CurvePoint, ...]
    rebalances: tuple[RebalanceRecord, ...]
    summary: BacktestSummary
