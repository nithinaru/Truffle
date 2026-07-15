"""Small deterministic Python facade over Truffle's numerical engines.

Callers supply a typed portfolio specification (or a mapping that validates as
one) and local pandas market data.  This module does not interpret natural
language, fetch data, contact a model, or provide financial advice.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from backtest.config import BacktestConfig
from backtest.engine import run_backtest as _run_backtest
from backtest.tearsheet import Tearsheet
from core.ir import PortfolioSpec
from core.report import SolutionReport
from core.solve import solve_spec as _solve_spec

type SpecInput = PortfolioSpec | Mapping[str, object]

__all__ = ["SpecInput", "run_walk_forward_backtest", "solve_portfolio"]


def _validated_spec(spec: SpecInput) -> PortfolioSpec:
    """Return a validated IR instance and reject non-IR input explicitly."""
    if not isinstance(spec, (PortfolioSpec, Mapping)):
        raise TypeError("spec must be a PortfolioSpec or a mapping of PortfolioSpec fields")
    # Rebuild existing models too: Pydantic's frozen setting prevents field
    # assignment but cannot make nested lists/dicts immutable.
    payload = spec.model_dump() if isinstance(spec, PortfolioSpec) else spec
    return PortfolioSpec.model_validate(payload)


def solve_portfolio(
    spec: SpecInput,
    prices: pd.DataFrame,
    *,
    sectors: dict[str, str] | None = None,
    benchmarks: dict[str, dict[str, float]] | None = None,
    factors: dict[str, dict[str, float]] | None = None,
    time_limit_s: float | None = None,
    diagnose: bool = False,
    periods_per_year: int = 252,
) -> SolutionReport:
    """Validate ``spec`` and solve it against caller-supplied local prices.

    The returned object is the structured numerical report.  The lower-level
    compiled CVXPY model is intentionally omitted from this simple facade;
    callers that need it can use :func:`core.solve.solve_spec` directly.
    Validation and solver exceptions propagate unchanged.
    """
    validated = _validated_spec(spec)
    _, report = _solve_spec(
        validated,
        prices,
        sectors=sectors,
        benchmarks=benchmarks,
        factors=factors,
        time_limit_s=time_limit_s,
        diagnose=diagnose,
        periods_per_year=periods_per_year,
    )
    return report


def run_walk_forward_backtest(
    spec: SpecInput,
    prices: pd.DataFrame,
    *,
    config: BacktestConfig | None = None,
    sectors: dict[str, str] | None = None,
    benchmarks: dict[str, dict[str, float]] | None = None,
    factors: dict[str, dict[str, float]] | None = None,
    market_prices: pd.Series | None = None,
    time_limit_s: float | None = None,
) -> Tearsheet:
    """Run the no-lookahead engine with a validated spec and local data.

    Timing, accounting, and typed failure behavior are exactly those of
    :func:`backtest.run_backtest`; this facade adds no implicit data access or
    exception recovery.
    """
    validated = _validated_spec(spec)
    return _run_backtest(
        validated,
        prices,
        config=config,
        sectors=sectors,
        benchmarks=benchmarks,
        factors=factors,
        market_prices=market_prices,
        time_limit_s=time_limit_s,
    )
