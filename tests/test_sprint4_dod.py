"""Slice 5 — Sprint-4 definition-of-done integration tests.

End to end on a synthetic 25-name universe:

* "minimize CVaR, at most 15 names, tech under 30%"  -> MILP (HiGHS)
* "minimize variance, at most 10 names"              -> MIQP (SCIP)

Both must solve, select <= K names, route to the correct backend, and report
conditional shadow prices flagged as such. We also assert the grounded
(deterministic) explanation states the conditionality and the selected count,
that a continuous problem is unchanged, and that an over-tight cardinality
raises cleanly.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from agent.explain import template_summary
from agent.grounding import verify
from core.exceptions import InfeasibleError
from core.ir import (
    Box,
    Budget,
    Cardinality,
    GroupCap,
    LongOnly,
    MinCVaR,
    MinVariance,
    PortfolioSpec,
)
from core.solve import solve_spec

SCIP_AVAILABLE = cp.SCIP in cp.installed_solvers()
N_NAMES = 25
TICKERS = [f"T{i:02d}" for i in range(N_NAMES)]
# First 8 are "Tech"; rest split across two other sectors.
SECTORS = {
    t: ("Tech" if i < 8 else "Energy" if i < 16 else "Health")
    for i, t in enumerate(TICKERS)
}


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    # Reproducible correlated-ish random walk; enough rows for estimation.
    rng = np.random.default_rng(7)
    days = 300
    drift = rng.normal(0.0003, 0.0002, size=N_NAMES)
    rets = rng.normal(0.0, 0.012, size=(days, N_NAMES)) + drift
    px = 100.0 * np.cumprod(1.0 + rets, axis=0)
    idx = pd.date_range("2023-01-02", periods=days, freq="B")
    return pd.DataFrame(px, index=idx, columns=TICKERS)


def test_milp_min_cvar_at_most_15_names_tech_under_30(prices: pd.DataFrame) -> None:
    spec = PortfolioSpec(
        universe=TICKERS,
        objective=MinCVaR(cvar_alpha=0.95),
        constraints=[
            Budget(),
            LongOnly(),
            GroupCap(group="Tech", max_weight=0.30),
            Cardinality(max_names=15),
        ],
    )
    _, report = solve_spec(spec, prices, sectors=SECTORS)

    assert report.solver == "HiGHS"  # MILP backend
    assert report.nonzero_names <= 15
    assert report.selected_names is not None
    assert len(report.selected_names) == report.nonzero_names
    assert report.duals_conditional is True
    assert report.optimality_gap is not None

    # Grounded explanation states the conditionality and the selected count.
    narration = template_summary(report)
    assert "conditional" in narration.lower()
    assert f"{report.nonzero_names} names" in narration or f"{report.nonzero_names} of" in narration
    assert verify(narration, report).ok


@pytest.mark.skipif(not SCIP_AVAILABLE, reason="SCIP (MIQP backend) not installed")
def test_miqp_min_variance_at_most_10_names(prices: pd.DataFrame) -> None:
    spec = PortfolioSpec(
        universe=TICKERS,
        objective=MinVariance(),
        constraints=[Budget(), LongOnly(), Cardinality(max_names=10)],
    )
    _, report = solve_spec(spec, prices, sectors=SECTORS)

    assert report.solver == "SCIP"  # MIQP backend
    assert report.nonzero_names <= 10
    assert report.duals_conditional is True
    assert report.selected_names is not None and len(report.selected_names) <= 10
    assert verify(template_summary(report), report).ok


def test_continuous_problem_unchanged_regression(prices: pd.DataFrame) -> None:
    # No cardinality => continuous, Clarabel, ordinary unconditional duals.
    spec = PortfolioSpec(
        universe=TICKERS,
        objective=MinVariance(),
        constraints=[Budget(), LongOnly(), Box(lower=0.0, upper=0.10)],
    )
    _, report = solve_spec(spec, prices)
    assert report.solver == "Clarabel"
    assert report.duals_conditional is False
    assert report.selected_names is None
    assert report.optimality_gap is None


def test_over_tight_cardinality_is_infeasible_cleanly(prices: pd.DataFrame) -> None:
    # max_names=1 but a universe-wide 30% cap with a full budget cannot be
    # satisfied by a single name (max reachable = 0.30 < 1.0): the MILP is
    # infeasible and we raise InfeasibleError cleanly.
    # NOTE: rich *diagnosis* of WHY (the minimal conflicting set) is Sprint 5;
    # here we only assert it fails cleanly rather than crashing.
    spec = PortfolioSpec(
        universe=TICKERS,
        objective=MinCVaR(cvar_alpha=0.95),
        constraints=[
            Budget(total=1.0),
            LongOnly(),
            Box(lower=0.0, upper=0.30),
            Cardinality(max_names=1),
        ],
    )
    with pytest.raises(InfeasibleError):
        solve_spec(spec, prices, sectors=SECTORS)
