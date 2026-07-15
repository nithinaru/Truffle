"""End-to-end solve helper: spec + prices → (CompiledProblem, SolutionReport).

This shared helper sits in core/ so both the deterministic CLI ``solve``
command and the chat loop can drive a solve through one code path. Keeping
solve plumbing in one place avoids divergence between the two entrypoints
(e.g. one swapping a solver while the other doesn't).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from numbers import Integral

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
from data.estimation import DEFAULT_PERIODS_PER_YEAR, estimate_moments
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
    """The numeric inputs shared by the MIP and its continuous restriction."""

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

    duals: dict[str, float]


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
            "Use the opt-in diagnosis path to identify a minimal conflicting "
            "set and any verified repairs."
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


def _synthetic_constraint_id(base: str, used_ids: set[str]) -> str:
    """Return a deterministic id that cannot collide with a user constraint."""

    candidate = base
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def _conditional_spec(
    spec: PortfolioSpec, selected_idx: list[int]
) -> tuple[PortfolioSpec, frozenset[str]]:
    """Fix the MIP selection while retaining the original full-universe math.

    Dropping assets from the universe is not equivalent to fixing their binary
    selectors: doing so also drops their entries from ``w_prev``, turnover, and
    transaction-cost expressions. Instead, keep the complete universe and all
    non-cardinality nodes, pin every unselected weight to zero, and recreate a
    Cardinality ``min_position`` floor for selected names. The added Box rows
    exist only to define the conditional problem and are removed from its dual
    report.
    """
    selected_idx_set = set(selected_idx)
    selected = [ticker for i, ticker in enumerate(spec.universe) if i in selected_idx_set]
    unselected = [ticker for i, ticker in enumerate(spec.universe) if i not in selected_idx_set]

    cardinalities = [c for c in spec.constraints if isinstance(c, Cardinality)]
    if len(cardinalities) != 1:
        raise SolverError(
            "A mixed-integer fix-and-resolve requires exactly one Cardinality constraint."
        )
    cardinality = cardinalities[0]

    new_constraints: list[Constraint] = [
        c for c in spec.constraints if not isinstance(c, Cardinality)
    ]
    used_ids = {c.id for c in spec.constraints}
    synthetic_ids: set[str] = set()

    if unselected:
        constraint_id = _synthetic_constraint_id("__conditional_unselected_zero", used_ids)
        synthetic_ids.add(constraint_id)
        new_constraints.append(
            Box(
                id=constraint_id,
                lower=0.0,
                upper=0.0,
                tickers=unselected,
                elastic=False,
            )
        )

    if cardinality.min_position is not None:
        constraint_id = _synthetic_constraint_id("__conditional_selected_floor", used_ids)
        synthetic_ids.add(constraint_id)
        new_constraints.append(
            Box(
                id=constraint_id,
                lower=cardinality.min_position,
                # LongOnly + Budget(total=1) already imply this upper bound; it
                # is supplied because Box represents a two-sided interval.
                upper=1.0,
                tickers=selected,
                elastic=False,
            )
        )

    return (
        PortfolioSpec(
            universe=list(spec.universe),
            objective=spec.objective,
            constraints=new_constraints,
            current_weights=spec.current_weights,
        ),
        frozenset(synthetic_ids),
    )


def _fix_and_resolve(
    spec: PortfolioSpec, selected_idx: list[int], inputs: _ProblemInputs
) -> _Conditional:
    """Re-solve the full-universe continuous problem at the fixed selection.

    The original MIP remains authoritative for weights, objective value, and
    VaR. This second solve exists only to recover conditional shadow prices.
    Keeping the original universe and pre-trade vector makes liquidation,
    turnover, transaction costs, factor data, and benchmark data identical to
    the MIP; synthetic rows fix the binary selection without leaking into the
    reported duals.
    """
    if not selected_idx:
        raise SolverError("The MIP selected no names; cannot form a conditional restriction.")
    restricted, synthetic_ids = _conditional_spec(spec, selected_idx)
    compiled = _compile(restricted, inputs)
    choice = select_solver(restricted)  # Cardinality removed -> continuous
    _run_solver(compiled.problem, choice.cp_solver, choice.name)
    _map_status(compiled.problem.status)
    duals = harvest_duals(compiled)
    return _Conditional(
        duals={cid: value for cid, value in duals.items() if cid not in synthetic_ids}
    )


def solve_spec(
    spec: PortfolioSpec,
    prices: pd.DataFrame,
    *,
    sectors: dict[str, str] | None = None,
    benchmarks: dict[str, dict[str, float]] | None = None,
    factors: dict[str, dict[str, float]] | None = None,
    time_limit_s: float | None = None,
    diagnose: bool = False,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> tuple[object, SolutionReport]:
    """Estimate moments, compile, solve, harvest duals, build a SolutionReport.

    Two solve paths, selected by :func:`core.routing.select_solver`:

    * **Continuous convex** (Clarabel): unchanged from Sprints 1–3. Ordinary,
      *unconditional* duals are harvested directly from the solve.
    * **Mixed-integer** (HiGHS for MILP / SCIP for MIQP or MISOCP): an integer program has
      no meaningful dual variables, so after the MIP solve we run a
      *fix-and-resolve* (:func:`_fix_and_resolve`) — fix the binaries at the
      optimal selection ``y*`` with full-universe zero/floor rows, drop the
      Cardinality node, re-solve the resulting continuous problem with
      Clarabel, and harvest its duals. Those are **conditional** shadow prices:
      valid given the selected name set, not globally. The original MIP remains
      authoritative for final weights and VaR. The report is flagged
      ``duals_conditional=True`` and carries ``selected_names``.

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
        diagnose: when true, an infeasible solve runs the opt-in elastic/IIS
            pass and attaches its ``ConflictReport`` to ``InfeasibleError``.
            The default remains false because diagnosis requires several
            additional solves and callers must choose that cost explicitly.
        periods_per_year: annualization factor used for estimated means and
            covariance (252 for daily closes, 52 for weekly, 12 for monthly).

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
    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, Integral)
        or periods_per_year < 1
    ):
        raise SolverError("periods_per_year must be a positive integer.")
    annualization = int(periods_per_year)
    mu, sigma = estimate_moments(panel, periods_per_year=annualization)
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

    def _check_status() -> None:
        try:
            _map_status(compiled.problem.status)
        except InfeasibleError as exc:
            if diagnose:
                # Lazy import keeps the ordinary solve path independent of the
                # multi-solve diagnostic machinery and avoids a module cycle.
                from core.diagnose import diagnose_infeasibility  # noqa: PLC0415

                try:
                    exc.conflict_report = diagnose_infeasibility(
                        spec,
                        prices,
                        sectors=sectors,
                        benchmarks=benchmarks,
                        factors=factors,
                        periods_per_year=annualization,
                    )
                except Exception as diagnosis_error:
                    # Diagnosis is an optional, secondary multi-solve pass.  A
                    # missing backend or diagnostic bug must never replace the
                    # solver's primary and already-proven infeasibility result.
                    exc.add_note(
                        "Optional infeasibility diagnosis failed: "
                        f"{type(diagnosis_error).__name__}."
                    )
                    raise exc from diagnosis_error
            raise

    if not choice.is_mip:
        # Continuous convex path (Clarabel) — unchanged from Sprints 1–3, with
        # ordinary (unconditional) duals harvested directly from the solve.
        elapsed_ms = _run_solver(compiled.problem, choice.cp_solver, choice.name)
        _check_status()
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
    _check_status()
    gap = _optimality_gap(compiled.problem)

    selected_idx = _selected_indices(compiled)
    selected = [spec.universe[i] for i in selected_idx]

    cond = _fix_and_resolve(spec, selected_idx, inputs)

    # The MIP is the sole source of the portfolio and objective auxiliaries.
    # Fix-and-resolve exists only for conditional duals; using its weights would
    # make a numerically different continuous solution the reported decision.
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
        duals=cond.duals,
        constraint_human_names=_human_names(spec),
        var=_var_of(spec, compiled),
        duals_conditional=True,
        selected_names=selected,
        optimality_gap=gap,
    )
    return compiled, report
