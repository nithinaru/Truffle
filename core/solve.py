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
from core.ir import (
    Box,
    Budget,
    Constraint,
    CVaRLimit,
    FactorExposure,
    GroupCap,
    LongOnly,
    MinCVaR,
    PortfolioSpec,
    TrackingErrorCap,
    TransactionCost,
    TurnoverCap,
)
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
    if isinstance(c, GroupCap):
        return f"the {c.group} group cap"
    if isinstance(c, TurnoverCap):
        return "the turnover cap"
    if isinstance(c, TransactionCost):
        return "the transaction-cost penalty"
    if isinstance(c, CVaRLimit):
        return f"the CVaR limit (α={c.alpha:g})"
    if isinstance(c, TrackingErrorCap):
        return f"the tracking-error cap vs {c.benchmark}"
    if isinstance(c, FactorExposure):
        return f"the {c.factor} factor-exposure limit"
    return c.id  # fallback


def _align_named_vectors(
    named: dict[str, dict[str, float]] | None, universe: list[str]
) -> dict[str, np.ndarray] | None:
    """Align ``{name -> {ticker -> value}}`` maps to universe-ordered arrays.

    Tickers absent from a given map default to 0.0. Returns ``None`` when no
    maps were supplied so the compiler keeps its "input not provided" errors.
    """
    if not named:
        return None
    out: dict[str, np.ndarray] = {}
    for name, by_ticker in named.items():
        out[name] = np.array([float(by_ticker.get(t, 0.0)) for t in universe], dtype=float)
    return out


def solve_spec(
    spec: PortfolioSpec,
    prices: pd.DataFrame,
    *,
    sectors: dict[str, str] | None = None,
    benchmarks: dict[str, dict[str, float]] | None = None,
    factors: dict[str, dict[str, float]] | None = None,
) -> tuple[object, SolutionReport]:
    """Estimate moments, compile, solve, harvest duals, build a SolutionReport.

    Args:
        spec: validated ``PortfolioSpec`` whose ``universe`` is a subset of
            ``prices.columns``.
        prices: price panel; columns must include every ticker in
            ``spec.universe``.
        sectors: ``{ticker -> group}`` mapping for ``GroupCap``.
        benchmarks: ``{name -> {ticker -> weight}}`` for tracking-error nodes;
            aligned to the universe (missing tickers -> 0.0).
        factors: ``{name -> {ticker -> loading}}`` for ``FactorExposure``.

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
    # Scenarios are needed by the MinCVaR objective *and* by any CVaRLimit node.
    needs_scenarios = isinstance(spec.objective, MinCVaR) or any(
        isinstance(c, CVaRLimit) for c in spec.constraints
    )
    scenarios = historical_scenarios(panel) if needs_scenarios else None
    # Single-shot pre-trade vector; absent holdings default to 0.0 ("from cash").
    w_prev = np.asarray(spec.w_prev_vector(), dtype=float)

    compiled = compile_spec(
        spec,
        mu=mu,
        sigma=sigma,
        scenarios=scenarios,
        w_prev=w_prev,
        sectors=sectors,
        benchmark_weights=_align_named_vectors(benchmarks, spec.universe),
        factor_loadings=_align_named_vectors(factors, spec.universe),
    )
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
