"""End-to-end solve helper: spec + prices → (CompiledProblem, SolutionReport).

This shared helper sits in core/ so both the deterministic CLI ``solve``
command and the chat loop can drive a solve through one code path. Keeping
solve plumbing in one place avoids divergence between the two entrypoints
(e.g. one swapping a solver while the other doesn't).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from core.compile_context import CompiledProblem
from core.compiler import compile_spec
from core.duals import harvest_duals
from core.exceptions import InfeasibleError, SolverError, UnboundedError
from core.ir import (
    Box,
    Budget,
    Cardinality,
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
from core.routing import select_solver
from data.estimation import estimate_moments
from data.inputs import align_named
from data.scenarios import historical_scenarios


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
    if isinstance(c, Cardinality):
        return f"the cardinality limit (≤ {c.max_names} names)"
    return c.id  # fallback


@dataclass
class _ProblemInputs:
    """The numeric inputs a compile needs, bundled so the continuous restriction
    can be assembled from index-subselected copies of the originals."""

    mu: np.ndarray
    sigma: np.ndarray
    scenarios: np.ndarray | None
    w_prev: np.ndarray
    sectors: dict[str, str] | None
    benchmarks: dict[str, dict[str, float]] | None
    factors: dict[str, dict[str, float]] | None


@dataclass
class _Conditional:
    """Result of the fix-and-resolve continuous restriction."""

    weights: dict[str, float]
    duals: dict[str, float]
    var: float | None


def _scenarios_if_needed(spec: PortfolioSpec, panel: pd.DataFrame) -> np.ndarray | None:
    # Scenarios are needed by the MinCVaR objective *and* by any CVaRLimit node.
    needs = isinstance(spec.objective, MinCVaR) or any(
        isinstance(c, CVaRLimit) for c in spec.constraints
    )
    return historical_scenarios(panel) if needs else None


def _compile(spec: PortfolioSpec, inputs: _ProblemInputs) -> CompiledProblem:
    return compile_spec(
        spec,
        mu=inputs.mu,
        sigma=inputs.sigma,
        scenarios=inputs.scenarios,
        w_prev=inputs.w_prev,
        sectors=inputs.sectors,
        benchmark_weights=align_named(inputs.benchmarks, spec.universe, label="Benchmark"),
        factor_loadings=align_named(inputs.factors, spec.universe, label="Factor"),
    )


def _run_solver(
    problem: cp.Problem, cp_solver: str, name: str, *, time_limit_s: float | None = None
) -> float:
    """Solve ``problem`` with ``cp_solver``; return wall-clock ms.

    A ``time_limit_s`` is forwarded to the MIP backends using their own
    parameter names (HiGHS ``time_limit``; SCIP ``limits/time``). Clarabel
    ignores it (the continuous path never passes one).
    """
    kwargs: dict[str, object] = {}
    if time_limit_s is not None:
        if cp_solver == cp.SCIP:
            kwargs["scip_params"] = {"limits/time": float(time_limit_s)}
        elif cp_solver == cp.HIGHS:
            kwargs["time_limit"] = float(time_limit_s)
    start = time.perf_counter()
    try:
        problem.solve(solver=cp_solver, **kwargs)
    except cp.SolverError as e:
        raise SolverError(f"{name} failed: {e}") from e
    return 1000.0 * (time.perf_counter() - start)


def _map_status(status: str) -> None:
    if status in {"infeasible", "infeasible_inaccurate"}:
        raise InfeasibleError(
            f"Solver reports the problem is infeasible (status: {status!r}). "
            "Cardinality combined with position/sector caps can make genuinely "
            "conflicting requests; a Sprint-5 diagnoser will identify the "
            "minimal conflicting set."
        )
    if status in {"unbounded", "unbounded_inaccurate"}:
        raise UnboundedError(
            f"Solver reports the problem is unbounded (status: {status!r}). "
            "Check that you have a budget constraint and finite expected returns."
        )
    if status not in {"optimal", "optimal_inaccurate"}:
        raise SolverError(f"Solver returned non-optimal status: {status!r}.")


def _var_of(spec: PortfolioSpec, compiled: CompiledProblem) -> float | None:
    if isinstance(spec.objective, MinCVaR):
        return float(compiled.extra_vars["t"].value)
    return None


def _human_names(spec: PortfolioSpec) -> dict[str, str]:
    return {c.id: human_name_for(c) for c in spec.constraints}


def _optimality_gap(problem: cp.Problem) -> float:
    """Best-effort MIP optimality gap from the solver's extra stats.

    At ``optimal`` status the proven gap is ~0 for both backends; we still read
    the real number when the backend exposes it (HiGHS ``mip_gap``; SCIP's
    ``model.getGap()``) so a time-limited solve can report a nonzero gap.
    """
    stats = getattr(problem, "solver_stats", None)
    es = getattr(stats, "extra_stats", None)
    if es is None:
        return 0.0
    gap = getattr(es, "mip_gap", None)  # HiGHS (HighsInfo)
    if gap is not None:
        return float(gap)
    if isinstance(es, dict):  # SCIP
        model = es.get("model")
        if model is not None:
            try:
                return float(model.getGap())
            except Exception:
                return 0.0
    return 0.0


def _selected_indices(compiled: CompiledProblem) -> list[int]:
    """Indices the MIP selected (binary ``y_i`` rounded to 1)."""
    y = compiled.extra_vars.get("y")
    if y is None or y.value is None:
        raise SolverError("MIP solve did not populate the cardinality selection vector y.")
    vals = np.asarray(y.value, dtype=float).ravel()
    return [i for i, v in enumerate(vals) if v > 0.5]


def _restrict_spec(
    spec: PortfolioSpec, selected: list[str], sectors: dict[str, str] | None
) -> PortfolioSpec:
    """Build the continuous restriction: universe = selected names, integrality
    dropped, every other constraint carried over with ids preserved so duals map
    back to the original IR.

    Constraints are adjusted only where the smaller universe demands it: per-
    ticker Box bounds are intersected with the selection (dropped if empty), and
    a GroupCap with no selected member is dropped (vacuously satisfied). All ids
    are preserved so :func:`core.duals.harvest_duals` keys back to the original
    constraints.
    """
    selected_set = set(selected)
    new_constraints: list = []
    for c in spec.constraints:
        if isinstance(c, Cardinality):
            continue  # drop integrality — the binaries are now fixed
        if isinstance(c, Box) and c.tickers is not None:
            kept = [t for t in c.tickers if t in selected_set]
            if not kept:
                continue
            new_constraints.append(c.model_copy(update={"tickers": kept}))
            continue
        if isinstance(c, GroupCap):
            if not any((sectors or {}).get(t) == c.group for t in selected):
                continue
            new_constraints.append(c)
            continue
        new_constraints.append(c)
    return PortfolioSpec(
        universe=list(selected),
        objective=spec.objective,
        constraints=new_constraints,
    )


def _fix_and_resolve(
    spec: PortfolioSpec, selected_idx: list[int], inputs: _ProblemInputs
) -> _Conditional:
    """Re-solve the continuous problem with the integer selection fixed.

    Fixing the binaries at ``y*`` is equivalent to restricting the universe to
    the selected names and dropping integrality (the prompt's sanctioned
    technique). We subselect the *same* mu/sigma/scenarios used by the MIP (not
    a re-estimate), so the restriction reproduces the MIP's continuous optimum
    exactly, then harvest its duals — the **conditional** shadow prices.
    """
    if not selected_idx:
        raise SolverError("The MIP selected no names; cannot form a conditional restriction.")
    selected = [spec.universe[i] for i in selected_idx]
    sub = _ProblemInputs(
        mu=inputs.mu[selected_idx],
        sigma=inputs.sigma[np.ix_(selected_idx, selected_idx)],
        scenarios=None if inputs.scenarios is None else inputs.scenarios[:, selected_idx],
        w_prev=inputs.w_prev[selected_idx],
        sectors=inputs.sectors,
        benchmarks=inputs.benchmarks,
        factors=inputs.factors,
    )
    restricted = _restrict_spec(spec, selected, inputs.sectors)
    compiled = _compile(restricted, sub)
    choice = select_solver(restricted)  # continuous restriction -> Clarabel
    _run_solver(compiled.problem, choice.cp_solver, choice.name)
    _map_status(compiled.problem.status)
    weights = dict(
        zip(restricted.universe, [float(x) for x in compiled.recovered_weights()], strict=True)
    )
    return _Conditional(
        weights=weights,
        duals=harvest_duals(compiled),
        var=_var_of(restricted, compiled),
    )


def solve_spec(
    spec: PortfolioSpec,
    prices: pd.DataFrame,
    *,
    sectors: dict[str, str] | None = None,
    benchmarks: dict[str, dict[str, float]] | None = None,
    factors: dict[str, dict[str, float]] | None = None,
    time_limit_s: float | None = None,
) -> tuple[object, SolutionReport]:
    """Estimate moments, compile, solve, harvest duals, build a SolutionReport.

    Two solve paths, selected by :func:`core.routing.select_solver`:

    * **Continuous convex** (Clarabel): unchanged from Sprints 1–3. Ordinary,
      *unconditional* duals are harvested directly from the solve.
    * **Mixed-integer** (HiGHS for MILP / SCIP for MIQP): an integer program has
      no meaningful dual variables, so after the MIP solve we run a
      *fix-and-resolve* (:func:`_fix_and_resolve`) — fix the binaries at the
      optimal selection ``y*`` (equivalently, restrict the universe to the
      selected names and drop integrality), re-solve the resulting continuous
      problem with Clarabel, and harvest its duals. Those are **conditional**
      shadow prices: valid given the selected name set, not globally. The report
      is flagged ``duals_conditional=True`` and carries ``selected_names``.

    Args:
        spec: validated ``PortfolioSpec`` whose ``universe`` is a subset of
            ``prices.columns``.
        prices: price panel; columns must include every ticker in
            ``spec.universe``.
        sectors: ``{ticker -> group}`` mapping for ``GroupCap``.
        benchmarks: ``{name -> {ticker -> weight}}`` for tracking-error nodes;
            aligned to the universe (missing tickers -> 0.0).
        factors: ``{name -> {ticker -> loading}}`` for ``FactorExposure``.
        time_limit_s: optional wall-clock limit passed to the MIP solver
            (ignored on the continuous path). Wired by the chat loop's
            universe-size guard (Slice 4).

    Returns:
        ``(compiled, report)`` — the compiled problem (for callers that want
        to inspect raw cvxpy state) and the structured report ready for
        narration. On the MIP path ``compiled`` is the *integer* problem.

    Raises:
        InfeasibleError, UnboundedError, SolverError: typed status-error
            mapping. Caller decides how to surface to the user.
    """
    if not set(spec.universe).issubset(set(prices.columns)):
        missing = sorted(set(spec.universe) - set(prices.columns))
        raise SolverError(f"Prices CSV is missing universe tickers: {missing}.")
    panel = prices[spec.universe]
    mu, sigma = estimate_moments(panel)
    scenarios = _scenarios_if_needed(spec, panel)
    # Single-shot pre-trade vector; absent holdings default to 0.0 ("from cash").
    w_prev = np.asarray(spec.w_prev_vector(), dtype=float)
    inputs = _ProblemInputs(
        mu=mu,
        sigma=sigma,
        scenarios=scenarios,
        w_prev=w_prev,
        sectors=sectors,
        benchmarks=benchmarks,
        factors=factors,
    )

    choice = select_solver(spec)
    compiled = _compile(spec, inputs)

    if not choice.is_mip:
        # Continuous convex path (Clarabel) — unchanged from Sprints 1–3, with
        # ordinary (unconditional) duals harvested directly from the solve.
        elapsed_ms = _run_solver(compiled.problem, choice.cp_solver, choice.name)
        _map_status(compiled.problem.status)
        weights = dict(
            zip(spec.universe, [float(x) for x in compiled.recovered_weights()], strict=True)
        )
        report = build_report(
            weights=weights,
            objective_kind=spec.objective.kind,
            objective_value=float(compiled.problem.value),
            solver=choice.name,
            solve_time_ms=elapsed_ms,
            status=compiled.problem.status,
            duals=harvest_duals(compiled),
            constraint_human_names=_human_names(spec),
            var=_var_of(spec, compiled),
        )
        return compiled, report

    # Mixed-integer path. Solve the MIP, read the optimal selection y*, then run
    # the fix-and-resolve to recover *conditional* shadow prices (see below and
    # core.duals). The MIP itself yields no duals.
    elapsed_ms = _run_solver(
        compiled.problem, choice.cp_solver, choice.name, time_limit_s=time_limit_s
    )
    _map_status(compiled.problem.status)
    gap = _optimality_gap(compiled.problem)

    selected_idx = _selected_indices(compiled)
    selected = [spec.universe[i] for i in selected_idx]

    cond = _fix_and_resolve(spec, selected_idx, inputs)

    # Full-universe weights: unselected names are exactly 0; selected names take
    # the continuous-restriction weights (which reproduce the MIP optimum).
    weights = {t: 0.0 for t in spec.universe}
    weights.update(cond.weights)
    report = build_report(
        weights=weights,
        objective_kind=spec.objective.kind,
        objective_value=float(compiled.problem.value),
        solver=choice.name,
        solve_time_ms=elapsed_ms,
        status=compiled.problem.status,
        duals=cond.duals,
        constraint_human_names=_human_names(spec),
        var=cond.var,
        duals_conditional=True,
        selected_names=selected,
        optimality_gap=gap,
    )
    return compiled, report
