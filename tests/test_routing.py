"""Slice 0 tests: the solver router picks the right backend per branch.

The routing decision itself (``route_for``) is pure and tested directly, so we
do not need a paid solver installed to assert the table. ``select_solver`` adds
the availability check on top; we assert the continuous branch end to end and
the actionable error when a MIP backend is missing.
"""

from __future__ import annotations

import cvxpy as cp
import pytest

from core.exceptions import CompilationError, SolverError
from core.ir import (
    Budget,
    LongOnly,
    MaxSharpe,
    MeanVariance,
    MinCVaR,
    MinVariance,
    PortfolioSpec,
)
from core.routing import route_for, select_solver


def test_route_continuous_convex_goes_to_clarabel() -> None:
    choice = route_for("convex", MinVariance())
    assert choice.cp_solver == cp.CLARABEL
    assert choice.name == "Clarabel"
    assert choice.problem_form == "continuous-convex"
    assert choice.is_mip is False


def test_route_mip_linear_objective_goes_to_highs() -> None:
    # cardinality + min-CVaR => MILP.
    choice = route_for("mip", MinCVaR(cvar_alpha=0.95))
    assert choice.cp_solver == cp.HIGHS
    assert choice.name == "HiGHS"
    assert choice.problem_form == "MILP"
    assert choice.is_mip is True


def test_route_mip_quadratic_objective_goes_to_scip() -> None:
    # cardinality + min-variance => MIQP.
    choice = route_for("mip", MinVariance())
    assert choice.cp_solver == cp.SCIP
    assert choice.name == "SCIP"
    assert choice.problem_form == "MIQP"
    assert choice.is_mip is True

    # mean-variance is also quadratic => MIQP.
    assert route_for("mip", MeanVariance(risk_aversion=1.0)).cp_solver == cp.SCIP


def test_route_mip_unsupported_objective_raises() -> None:
    # max_sharpe (change-of-variable) cannot be combined with integrality here.
    with pytest.raises(CompilationError, match="mixed-integer"):
        route_for("mip", MaxSharpe())


def test_select_solver_continuous_spec_is_clarabel() -> None:
    spec = PortfolioSpec(
        universe=["AAA", "BBB", "CCC"],
        objective=MinVariance(),
        constraints=[Budget(), LongOnly()],
    )
    assert spec.problem_class == "convex"
    assert select_solver(spec).name == "Clarabel"


def test_select_solver_missing_backend_raises_actionable_error(monkeypatch) -> None:
    # Simulate SCIP being uninstalled: select_solver must raise a SolverError
    # naming the missing solver and how to install it — never a silent fallback.
    # Force the route to SCIP (Cardinality lands in Slice 1, so we patch the
    # pure router rather than build a real MIP spec here) and hide SCIP from the
    # installed-solver list.
    import core.routing as routing  # noqa: PLC0415

    monkeypatch.setattr(cp, "installed_solvers", lambda: ["CLARABEL", "HIGHS"])
    monkeypatch.setattr(
        routing, "route_for", lambda *_a, **_k: routing.SolverChoice(cp.SCIP, "SCIP", "MIQP")
    )
    spec = PortfolioSpec(universe=["AAA", "BBB"], objective=MinVariance())
    with pytest.raises(SolverError, match="pyscipopt"):
        select_solver(spec)
