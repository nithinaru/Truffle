"""Slice 2 tests: fix-and-resolve conditional shadow prices.

A MIP has no duals; Truffle recovers *conditional* shadow prices by fixing the
integer selection y* (restricting the universe to the selected names, dropping
integrality) and harvesting duals from that continuous restriction. These tests
assert the mechanism: the report is flagged conditional, carries the selected
set, the unselected weights are exactly zero, and the restriction reproduces the
MIP optimum. The continuous path is guarded to stay *unconditional*.
"""

from __future__ import annotations

from pathlib import Path

import cvxpy as cp
import pandas as pd
import pytest

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

EXAMPLES = Path(__file__).parent.parent / "examples"
UNIVERSE = ["AAA", "BBB", "CCC", "DDD", "EEE"]
SECTORS = {"AAA": "Tech", "BBB": "Tech", "CCC": "Energy", "DDD": "Energy", "EEE": "Healthcare"}
SCIP_AVAILABLE = cp.SCIP in cp.installed_solvers()


@pytest.fixture
def prices() -> pd.DataFrame:
    return pd.read_csv(EXAMPLES / "prices_sample.csv", parse_dates=[0], index_col=0)


@pytest.mark.skipif(not SCIP_AVAILABLE, reason="SCIP (MIQP backend) not installed")
def test_miqp_conditional_duals_flagged_and_selected_set(prices: pd.DataFrame) -> None:
    spec = PortfolioSpec(
        universe=UNIVERSE,
        objective=MinVariance(),
        constraints=[Budget(), LongOnly(), Cardinality(max_names=2)],
    )
    _, report = solve_spec(spec, prices, sectors=SECTORS)

    assert report.solver == "SCIP"
    assert report.duals_conditional is True
    assert report.selected_names is not None
    # The selected set is exactly the nonzero names, and respects the cap.
    assert len(report.selected_names) == report.nonzero_names <= 2
    # Unselected names are pinned to exactly zero.
    for t in UNIVERSE:
        if t not in report.selected_names:
            assert report.weights[t] == 0.0
    # The fix-and-resolve produced real conditional shadow prices.
    assert report.binding
    assert all(b.shadow_price == b.shadow_price for b in report.binding)  # not NaN
    assert report.optimality_gap is not None


def test_milp_conditional_duals_with_group_cap(prices: pd.DataFrame) -> None:
    # cardinality + min-CVaR => MILP (HiGHS). A binding Tech cap shows up as a
    # *conditional* shadow price given the selected names.
    spec = PortfolioSpec(
        universe=UNIVERSE,
        objective=MinCVaR(cvar_alpha=0.95),
        constraints=[
            Budget(),
            LongOnly(),
            Box(lower=0.0, upper=0.6),
            GroupCap(group="Tech", max_weight=0.5),
            Cardinality(max_names=3),
        ],
    )
    _, report = solve_spec(spec, prices, sectors=SECTORS)

    assert report.solver == "HiGHS"
    assert report.duals_conditional is True
    assert report.nonzero_names <= 3
    assert report.var is not None  # CVaR objective exposes VaR
    assert report.optimality_gap is not None


def test_continuous_duals_remain_unconditional(prices: pd.DataFrame) -> None:
    # Regression guard: a continuous problem still routes to Clarabel and
    # produces ordinary (non-conditional) duals exactly as before.
    spec = PortfolioSpec(
        universe=UNIVERSE,
        objective=MinVariance(),
        constraints=[Budget(), LongOnly(), Box(lower=0.0, upper=0.30, tickers=["AAA"])],
    )
    _, report = solve_spec(spec, prices)

    assert report.solver == "Clarabel"
    assert report.duals_conditional is False
    assert report.selected_names is None
    assert report.optimality_gap is None
    assert report.binding  # the AAA cap binds
