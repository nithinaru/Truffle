"""Validated configuration for the walk-forward backtester."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BacktestConfig(BaseModel):
    """Controls historical estimation, execution timing, and accounting.

    A signal always fills one observation later.  Keeping that lag fixed at
    one is intentional: with close-only data, it prevents a strategy from
    both learning from and claiming a fill at the same close.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    lookback_returns: int = Field(
        default=252,
        ge=2,
        description="Trailing simple/log return observations available at each signal.",
    )
    rebalance_frequency: Literal["weekly", "monthly", "quarterly"] = "monthly"
    execution_lag_observations: Literal[1] = 1
    periods_per_year: int = Field(default=252, ge=1)
    cvar_alpha: float = Field(default=0.95, gt=0.0, lt=1.0)
    execution_cost_bps: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Realized proportional cost per unit of L1 turnover. None infers "
            "the sum of TransactionCost nodes in the spec."
        ),
    )
    annual_risk_free_rate: float = Field(default=0.0, gt=-1.0)
