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
import numpy as np
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
    TransactionCost,
    TurnoverCap,
)
from core.solve import _conditional_spec, solve_spec

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


def test_mip_report_keeps_full_universe_turnover_and_original_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Liquidating an unselected holding must count in fix-and-resolve.

    Starting from 10%/10%/80%, a two-name target under a 20% turnover cap can
    liquidate B and add the same 10% to A. The former restricted-universe pass
    forgot B's 10% liquidation, then re-optimized A/C to 25%/75% and reported a
    target with 30% true turnover. The report must instead retain the original
    MIP's 20%/0%/80% decision and its VaR.
    """
    universe = ["A", "B", "C"]
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0],
            "B": [100.0, 101.0],
            "C": [100.0, 101.0],
        },
        index=pd.date_range("2025-01-01", periods=2),
    )
    scenarios = np.tile(np.array([[0.20, 0.10, 0.0]]), (4, 1))
    monkeypatch.setattr(
        "core.solve.estimate_moments",
        lambda panel, periods_per_year=252: (
            np.zeros(panel.shape[1]),
            np.eye(panel.shape[1]),
        ),
    )
    monkeypatch.setattr("core.solve.historical_scenarios", lambda panel: scenarios)

    current = {"A": 0.10, "B": 0.10, "C": 0.80}
    turnover_id = "turnover"
    spec = PortfolioSpec(
        universe=universe,
        objective=MinCVaR(cvar_alpha=0.50),
        constraints=[
            Budget(),
            LongOnly(),
            TurnoverCap(id=turnover_id, max_turnover=0.20),
            Cardinality(max_names=2),
        ],
        current_weights=current,
    )

    compiled, report = solve_spec(spec, prices)

    mip_weights = dict(
        zip(universe, [float(x) for x in compiled.recovered_weights()], strict=True)
    )
    assert report.weights == pytest.approx(mip_weights, abs=1e-9)
    assert report.weights == pytest.approx({"A": 0.20, "B": 0.0, "C": 0.80}, abs=1e-9)
    true_turnover = sum(abs(report.weights[t] - current[t]) for t in universe)
    assert true_turnover == pytest.approx(0.20, abs=1e-9)
    assert true_turnover <= 0.20 + 1e-9
    assert report.var == pytest.approx(float(compiled.extra_vars["t"].value), abs=1e-9)
    assert all(not item.constraint_id.startswith("__conditional_") for item in report.binding)


def test_conditional_spec_preserves_holdings_costs_and_min_position_floor() -> None:
    spec = PortfolioSpec(
        universe=["A", "B", "C"],
        objective=MinCVaR(cvar_alpha=0.95),
        constraints=[
            Budget(id="budget"),
            LongOnly(id="long"),
            TurnoverCap(id="turnover", max_turnover=0.40),
            TransactionCost(id="cost", bps=12.0),
            Cardinality(id="names", max_names=2, min_position=0.25),
        ],
        current_weights={"A": 0.20, "B": 0.30, "C": 0.50},
    )

    restricted, synthetic_ids = _conditional_spec(spec, [0, 2])

    assert restricted.universe == spec.universe
    assert restricted.current_weights == spec.current_weights
    assert restricted.objective is spec.objective
    assert not any(isinstance(c, Cardinality) for c in restricted.constraints)
    assert {c.id for c in restricted.constraints if c.id not in synthetic_ids} == {
        "budget",
        "long",
        "turnover",
        "cost",
    }
    synthetic = {
        c.id: c
        for c in restricted.constraints
        if c.id in synthetic_ids and isinstance(c, Box)
    }
    zero_fix = next(c for c in synthetic.values() if c.lower == c.upper == 0.0)
    floor = next(c for c in synthetic.values() if c.lower == 0.25)
    assert zero_fix.tickers == ["B"]
    assert floor.tickers == ["A", "C"]
    assert floor.upper == 1.0


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
