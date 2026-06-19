"""End-to-end solve helper: spec + prices → (CompiledProblem, SolutionReport).

This shared helper sits in core/ so both the deterministic CLI ``solve``
command and the chat loop can drive a solve through one code path. Keeping
solve plumbing in one place avoids divergence between the two entrypoints
(e.g. one swapping a solver while the other doesn't).
"""

from __future__ import annotations

import time

import cvxpy as cp
import numpy as np
import pandas as pd

from core.compiler import compile_spec
from core.duals import harvest_duals
from core.exceptions import InfeasibleError, SolverError, UnboundedError
from core.ir import Box, Budget, Constraint, LongOnly, MinCVaR, PortfolioSpec
from core.report import SolutionReport, build_report
from data.estimation import estimate_moments
from data.scenarios import historical_scenarios

SOLVER_NAME = "Clarabel"


def human_name_for(c: Constraint) -> str:
    """Plain-English label for a constraint, used in the binding-report."""
    if isinstance(c, Budget):
        return "the budget constraint"
    if isinstance(c, LongOnly):
        return "the long-only constraint"
    if isinstance(c, Box):
        if c.tickers and len(c.tickers) == 1:
            return f"the {c.tickers[0]} position cap"
        if c.tickers:
            return f"the position cap on {', '.join(c.tickers)}"
        return "the universe-wide position bounds"
    return c.id  # fallback


def solve_spec(spec: PortfolioSpec, prices: pd.DataFrame) -> tuple[object, SolutionReport]:
    """Estimate moments, compile, solve, harvest duals, build a SolutionReport.

    Args:
        spec: validated ``PortfolioSpec`` whose ``universe`` is a subset of
            ``prices.columns``.
        prices: price panel; columns must include every ticker in
            ``spec.universe``.

    Returns:
        ``(compiled, report)`` — the compiled problem (for callers that want
        to inspect raw cvxpy state) and the structured report ready for
        narration.

    Raises:
        InfeasibleError, UnboundedError, SolverError: typed status-error
            mapping. Caller decides how to surface to the user.
    """
    if not set(spec.universe).issubset(set(prices.columns)):
        missing = sorted(set(spec.universe) - set(prices.columns))
        raise SolverError(f"Prices CSV is missing universe tickers: {missing}.")
    panel = prices[spec.universe]
    mu, sigma = estimate_moments(panel)
    scenarios = historical_scenarios(panel) if isinstance(spec.objective, MinCVaR) else None
    # Single-shot pre-trade vector; absent holdings default to 0.0 ("from cash").
    w_prev = np.asarray(spec.w_prev_vector(), dtype=float)

    compiled = compile_spec(spec, mu=mu, sigma=sigma, scenarios=scenarios, w_prev=w_prev)
    start = time.perf_counter()
    try:
        compiled.problem.solve(solver=cp.CLARABEL)
    except cp.SolverError as e:
        raise SolverError(f"Clarabel failed: {e}") from e
    elapsed_ms = 1000.0 * (time.perf_counter() - start)

    status = compiled.problem.status
    if status in {"infeasible", "infeasible_inaccurate"}:
        raise InfeasibleError(
            f"Solver reports the problem is infeasible (status: {status!r}). "
            "Sprint 3 will run an elastic relaxation to identify the conflicting set."
        )
    if status in {"unbounded", "unbounded_inaccurate"}:
        raise UnboundedError(
            f"Solver reports the problem is unbounded (status: {status!r}). "
            "Check that you have a budget constraint and finite expected returns."
        )
    if status not in {"optimal", "optimal_inaccurate"}:
        raise SolverError(f"Solver returned non-optimal status: {status!r}.")

    weights = dict(zip(spec.universe, [float(x) for x in compiled.weights.value], strict=True))
    duals = harvest_duals(compiled)
    var = (
        float(compiled.extra_vars["t"].value)
        if isinstance(spec.objective, MinCVaR)
        else None
    )
    human_names = {c.id: human_name_for(c) for c in spec.constraints}
    report = build_report(
        weights=weights,
        objective_kind=spec.objective.kind,
        objective_value=float(compiled.problem.value),
        solver=SOLVER_NAME,
        solve_time_ms=elapsed_ms,
        status=status,
        duals=duals,
        constraint_human_names=human_names,
        var=var,
    )
    return compiled, report
