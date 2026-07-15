"""End-to-end solve helper: spec + prices → (CompiledProblem, SolutionReport).

This shared helper sits in core/ so both the deterministic CLI ``solve``
command and the chat loop can drive a solve through one code path. Keeping
solve plumbing in one place avoids divergence between the two entrypoints
(e.g. one swapping a solver while the other doesn't).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import cvxpy as cp
import numpy as np
import pandas as pd

from core.compile_context import CompiledProblem
from core.compiler import compile_spec
from core.exceptions import DualsUnavailableError, InfeasibleError, SolverError, UnboundedError
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
from core.report_semantics import ReportSemantics, build_report_semantics
from core.routing import select_solver
from core.sensitivity import (
    SensitivityCoverage,
    SensitivityRecord,
    harvest_sensitivities,
    sensitivity_coverage,
    sensitivity_dependency_reasons,
)
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

    weights: np.ndarray | None
    sensitivities: tuple[SensitivityRecord, ...]
    unavailable_reasons: dict[str, str]
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class _MipTermination:
    """Backend-verified meaning of a successful MIP stop."""

    reason: Literal["optimal", "time_limit"]
    optimality_proven: bool


@dataclass(frozen=True)
class _ValidatedMipIncumbent:
    """A complete, finite, feasible mixed-integer primal snapshot."""

    weights: np.ndarray
    selected_indices: tuple[int, ...]


_MIP_INTEGRALITY_TOL = 1e-5
_MIP_FEASIBILITY_TOL = 1e-5
_CONDITIONAL_SUPPORT_REL_TOL = 1e-12


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


def _validate_time_limit(time_limit_s: float | None) -> float | None:
    """Return a finite positive MIP limit, rejecting ambiguous numeric input."""

    if time_limit_s is None:
        return None
    if (
        isinstance(time_limit_s, bool)
        or not isinstance(time_limit_s, Real)
        or not math.isfinite(float(time_limit_s))
        or float(time_limit_s) <= 0.0
    ):
        raise SolverError("time_limit_s must be a finite positive number of seconds.")
    return float(time_limit_s)


def _var_of(spec: PortfolioSpec, compiled: CompiledProblem) -> float | None:
    if isinstance(spec.objective, MinCVaR):
        return float(compiled.extra_vars["t"].value)
    return None


def _human_names(spec: PortfolioSpec) -> dict[str, str]:
    return {c.id: human_name_for(c) for c in spec.constraints}


def _report_semantics(
    spec: PortfolioSpec,
    compiled: CompiledProblem,
    weights: np.ndarray,
    inputs: _ProblemInputs,
    *,
    periods_per_year: int,
) -> ReportSemantics:
    """Build typed metrics from the same authoritative inputs as the solve."""

    try:
        decomposition, metrics = build_report_semantics(
            spec,
            compiled,
            weights,
            inputs.mu,
            inputs.sigma,
            scenarios=inputs.scenarios,
            w_prev=inputs.w_prev,
            benchmark_weights=align_named(
                inputs.benchmarks,
                spec.universe,
                label="Benchmark",
            ),
            periods_per_year=periods_per_year,
        )
        reconstruction_scale = max(
            1.0,
            abs(decomposition.solver_value),
            math.fsum(abs(term.objective_contribution) for term in decomposition.terms),
        )
        if abs(decomposition.reconstruction_error) > 1e-6 * reconstruction_scale:
            raise ValueError(
                "typed objective terms do not reconstruct the solver score "
                f"(error {decomposition.reconstruction_error:g} objective_score)."
            )
        return decomposition, metrics
    except (TypeError, ValueError, OverflowError) as exc:
        raise SolverError(
            "The solver returned a portfolio, but its objective decomposition or "
            f"portfolio metrics could not be validated: {exc}"
        ) from exc


def _solution_sensitivities(
    compiled: CompiledProblem,
    weights: np.ndarray,
    inputs: _ProblemInputs,
    *,
    conditional: bool,
) -> tuple[SensitivityRecord, ...]:
    """Harvest typed rows from the exact numeric state used by a solve."""

    try:
        return harvest_sensitivities(
            compiled,
            weights,
            inputs.w_prev,
            inputs.sigma,
            scenarios=inputs.scenarios,
            sectors=inputs.sectors,
            benchmarks=align_named(
                inputs.benchmarks,
                compiled.spec.universe if compiled.spec is not None else [],
                label="Benchmark",
            ),
            factors=align_named(
                inputs.factors,
                compiled.spec.universe if compiled.spec is not None else [],
                label="Factor",
            ),
            conditional=conditional,
        )
    except DualsUnavailableError as exc:
        raise SolverError(f"Solver sensitivities could not be validated: {exc}") from exc


def _coverage(
    spec: PortfolioSpec,
    sensitivities: tuple[SensitivityRecord, ...],
    *,
    conditional: bool,
    constraint_unavailable_reasons: dict[str, str] | None = None,
    unavailable_reason: str | None = None,
) -> dict[str, SensitivityCoverage]:
    coverage = sensitivity_coverage(spec, sensitivities, conditional)
    for constraint_id, reason in (constraint_unavailable_reasons or {}).items():
        if constraint_id in coverage:
            coverage[constraint_id] = SensitivityCoverage(
                availability="unavailable",
                reason=reason,
            )
    if unavailable_reason is None:
        return coverage
    return {
        constraint_id: SensitivityCoverage(
            availability="unavailable",
            reason=(
                unavailable_reason
                if item.reason is None
                else f"{item.reason} {unavailable_reason}"
            ),
        )
        for constraint_id, item in coverage.items()
    }


def _scip_native_status(problem: cp.Problem) -> str | None:
    """Read SCIP's native status without inferring it from CVXPY wording."""

    stats = getattr(problem, "solver_stats", None)
    es = getattr(stats, "extra_stats", None)
    if not isinstance(es, dict):
        return None
    native = es.get("scip_status")
    if native is not None:
        return str(native).strip().lower()
    model = es.get("model")
    if model is None:
        return None
    try:
        return str(model.getStatus()).strip().lower()
    except Exception:
        return None


def _mip_termination(
    problem: cp.Problem,
    *,
    cp_solver: str,
    time_limit_requested: bool,
) -> _MipTermination:
    """Map a MIP stop only when the backend proves what caused it.

    HiGHS exposes a time stop as CVXPY ``user_limit``. SCIP currently maps its
    native ``timelimit`` status to ``optimal_inaccurate``. Neither generic
    string is safe to accept without a limit supplied by this call and, for
    SCIP, the solver-native status.
    """

    status = problem.status
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

    if cp_solver == cp.SCIP:
        native = _scip_native_status(problem)
        if (
            time_limit_requested
            and status in {"optimal", "optimal_inaccurate", "user_limit"}
            and native == "timelimit"
        ):
            return _MipTermination(reason="time_limit", optimality_proven=False)
        if status == "optimal" and native in {None, "optimal"}:
            return _MipTermination(reason="optimal", optimality_proven=True)
        if status == "optimal_inaccurate" and native == "optimal":
            return _MipTermination(reason="optimal", optimality_proven=True)

    if status == "optimal" and cp_solver != cp.SCIP:
        return _MipTermination(reason="optimal", optimality_proven=True)

    if cp_solver == cp.HIGHS and time_limit_requested and status == "user_limit":
        return _MipTermination(reason="time_limit", optimality_proven=False)

    raise SolverError(
        "Mixed-integer solver returned an unverified termination state: "
        f"CVXPY status {status!r}. No portfolio was reported."
    )


def _reported_mip_gap(problem: cp.Problem) -> float | None:
    """Read a backend relative gap, never manufacturing a zero."""

    stats = getattr(problem, "solver_stats", None)
    es = getattr(stats, "extra_stats", None)
    raw: object | None = None
    scip_infinity: float | None = None
    if es is not None:
        raw = getattr(es, "mip_gap", None)  # HiGHS (HighsInfo)
    if raw is None and isinstance(es, dict):  # SCIP
        model = es.get("model")
        if model is not None:
            try:
                raw = model.getGap()
            except Exception:
                return None
            try:
                scip_infinity = float(model.infinity())
            except Exception:
                scip_infinity = None
    if raw is None:
        return None
    try:
        gap = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SolverError("MIP backend returned a non-numeric optimality gap.") from exc
    if not math.isfinite(gap) or gap < 0.0:
        raise SolverError(
            f"MIP backend returned an invalid relative optimality gap: {gap!r}."
        )
    if (
        scip_infinity is not None
        and math.isfinite(scip_infinity)
        and scip_infinity > 0.0
        and gap >= scip_infinity
    ):
        raise SolverError(
            "SCIP returned its infinity sentinel instead of a finite relative "
            "optimality gap."
        )
    return gap


def _optimality_gap(problem: cp.Problem, termination: _MipTermination) -> float:
    """Return the actual relative gap, with zero inferred only from proof."""

    gap = _reported_mip_gap(problem)
    if gap is not None:
        return gap
    if termination.optimality_proven:
        return 0.0
    raise SolverError(
        "The time-limited MIP returned a candidate incumbent but no finite "
        "backend optimality gap, so Truffle will not report it."
    )


def _validate_mip_incumbent(compiled: CompiledProblem) -> _ValidatedMipIncumbent:
    """Validate the complete mixed-integer primal before it crosses the API.

    A time-limit status can leave CVXPY variables populated with zeros or a
    partially feasible relaxation. Presence of values is therefore not enough:
    every variable must be finite and inside its declared domain, every
    explicit constraint must satisfy a fixed feasibility tolerance, and the
    selection vector must be genuinely integral.
    """

    spec = compiled.spec
    if spec is None:
        raise SolverError("MIP incumbent validation requires the originating spec.")

    for variable in compiled.problem.variables():
        value = variable.value
        if value is None:
            raise SolverError(
                f"MIP solve did not populate variable {variable.name()!r}; no incumbent exists."
            )
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SolverError(
                f"MIP variable {variable.name()!r} is not a numeric primal value."
            ) from exc
        if array.shape != variable.shape or not np.all(np.isfinite(array)):
            raise SolverError(
                f"MIP variable {variable.name()!r} has a missing, malformed, or non-finite "
                "primal value."
            )
        try:
            projected = np.asarray(variable.project(array), dtype=float)
        except Exception as exc:
            raise SolverError(
                f"MIP variable {variable.name()!r} could not be checked against its domain."
            ) from exc
        if projected.shape != array.shape or not np.all(np.isfinite(projected)):
            raise SolverError(
                f"MIP variable {variable.name()!r} has an invalid domain projection."
            )
        if np.max(np.abs(projected - array), initial=0.0) > _MIP_FEASIBILITY_TOL:
            raise SolverError(
                f"MIP variable {variable.name()!r} violates its declared domain."
            )

    objective_value = compiled.problem.value
    try:
        objective = float(objective_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SolverError("MIP solve did not produce a numeric incumbent objective.") from exc
    if not math.isfinite(objective):
        raise SolverError("MIP solve produced a non-finite incumbent objective.")
    try:
        evaluated_objective = float(compiled.problem.objective.value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SolverError(
            "MIP incumbent objective could not be recomputed from its primal variables."
        ) from exc
    if not math.isfinite(evaluated_objective):
        raise SolverError(
            "MIP incumbent objective recomputed to a non-finite value."
        )
    objective_tolerance = _MIP_FEASIBILITY_TOL * max(
        1.0, abs(objective), abs(evaluated_objective)
    )
    if abs(objective - evaluated_objective) > objective_tolerance:
        raise SolverError(
            "MIP backend objective is inconsistent with the returned primal "
            f"variables by {abs(objective - evaluated_objective):.6g}."
        )

    y = compiled.extra_vars.get("y")
    if y is None or y.value is None:
        raise SolverError("MIP solve did not populate the cardinality selection vector y.")
    vals = np.asarray(y.value, dtype=float)
    expected_shape = (len(spec.universe),)
    if vals.shape != expected_shape or not np.all(np.isfinite(vals)):
        raise SolverError(
            "MIP cardinality selection vector y is malformed or non-finite; "
            f"expected shape {expected_shape}, got {vals.shape}."
        )
    rounded = np.rint(vals)
    if (
        np.any(vals < -_MIP_INTEGRALITY_TOL)
        or np.any(vals > 1.0 + _MIP_INTEGRALITY_TOL)
        or np.max(np.abs(vals - rounded), initial=0.0) > _MIP_INTEGRALITY_TOL
    ):
        raise SolverError("MIP selection vector y is not binary within tolerance.")

    try:
        weights = np.asarray(compiled.recovered_weights(), dtype=float)
    except Exception as exc:
        raise SolverError("MIP incumbent weights could not be recovered safely.") from exc
    if weights.shape != expected_shape or not np.all(np.isfinite(weights)):
        raise SolverError(
            "MIP incumbent weights are malformed or non-finite; "
            f"expected shape {expected_shape}, got {weights.shape}."
        )

    for index, constraint in enumerate(compiled.problem.constraints):
        try:
            violation = np.asarray(constraint.violation(), dtype=float)
        except Exception as exc:
            raise SolverError(
                f"MIP incumbent constraint row {index} could not be validated."
            ) from exc
        if not np.all(np.isfinite(violation)):
            raise SolverError(
                f"MIP incumbent constraint row {index} has a non-finite residual."
            )
        worst = float(np.max(np.abs(violation), initial=0.0))
        if worst > _MIP_FEASIBILITY_TOL:
            raise SolverError(
                "MIP solver stopped without a feasible incumbent: constraint row "
                f"{index} violates the model by {worst:.6g} "
                f"(tolerance {_MIP_FEASIBILITY_TOL:.6g})."
            )

    selected = tuple(int(i) for i in np.flatnonzero(rounded.astype(int)))
    if not selected:
        raise SolverError("The MIP incumbent selected no names; no portfolio was reported.")
    return _ValidatedMipIncumbent(weights=weights.copy(), selected_indices=selected)


def _selected_indices(compiled: CompiledProblem) -> list[int]:
    """Compatibility wrapper returning only validated selected indices."""

    return list(_validate_mip_incumbent(compiled).selected_indices)


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


def _unsupported_conditional_constraint_reasons(
    spec: PortfolioSpec,
    selected_idx: list[int],
    inputs: _ProblemInputs,
) -> dict[str, str]:
    """Rows whose coefficients have no support on a selected-name degree of freedom."""

    selected = set(selected_idx)
    unavailable: dict[str, str] = {}
    aligned_factors = align_named(inputs.factors, spec.universe, label="Factor")
    for constraint in spec.constraints:
        if isinstance(constraint, GroupCap):
            has_support = inputs.sectors is not None and any(
                index in selected
                and inputs.sectors.get(ticker) == constraint.group
                for index, ticker in enumerate(spec.universe)
            )
            if not has_support:
                unavailable[constraint.id] = (
                    "The group row has no selected-name coefficient support; all "
                    "affected weights are fixed at zero in the conditional solve."
                )
        elif isinstance(constraint, FactorExposure):
            if aligned_factors is None or constraint.factor not in aligned_factors:
                # Compilation owns the missing-input error. This branch is only
                # defensive for direct calls with a malformed mocked input.
                continue
            loadings = aligned_factors[constraint.factor]
            scale = max(1.0, float(np.max(np.abs(loadings), initial=0.0)))
            selected_loadings = loadings[selected_idx]
            if (
                selected_loadings.size == 0
                or float(np.max(np.abs(selected_loadings), initial=0.0))
                <= _CONDITIONAL_SUPPORT_REL_TOL * scale
            ):
                unavailable[constraint.id] = (
                    "The factor row has no selected-name coefficient support; all "
                    "nonzero loadings are attached to weights fixed at zero in the "
                    "conditional solve."
                )
    return unavailable


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
    weights = np.asarray(compiled.recovered_weights(), dtype=float)
    if compiled.problem.status != "optimal":
        return _Conditional(
            weights=weights,
            sensitivities=(),
            unavailable_reasons={},
            unavailable_reason=(
                "Conditional sensitivities are unavailable because fix-and-resolve "
                f"returned solver status {compiled.problem.status!r}; an exact "
                "'optimal' status is required for authoritative derivatives."
            ),
        )

    unavailable_reasons = {
        constraint_id: reason
        for constraint_id, reason in sensitivity_dependency_reasons(compiled).items()
        if any(constraint.id == constraint_id for constraint in spec.constraints)
    }
    unsupported_reasons = _unsupported_conditional_constraint_reasons(
        spec,
        selected_idx,
        inputs,
    )
    unavailable_reasons.update(unsupported_reasons)
    sensitivities = _solution_sensitivities(
        compiled,
        weights,
        inputs,
        conditional=True,
    )
    selected_tickers = {spec.universe[index] for index in selected_idx}
    unavailable_ids = set(unavailable_reasons)
    return _Conditional(
        weights=weights,
        sensitivities=tuple(
            record
            for record in sensitivities
            if record.constraint_id not in synthetic_ids
            and record.constraint_id not in unavailable_ids
            and not (
                record.kind in {"long_only", "box"}
                and record.row_label not in selected_tickers
            )
        ),
        unavailable_reasons=unavailable_reasons,
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
    time_limit_s = _validate_time_limit(time_limit_s)
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

    def _maybe_diagnose(exc: InfeasibleError) -> None:
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
        raise exc

    def _check_status() -> None:
        try:
            _map_status(compiled.problem.status)
        except InfeasibleError as exc:
            _maybe_diagnose(exc)

    if not choice.is_mip:
        # Continuous convex path (Clarabel) — unchanged from Sprints 1–3, with
        # ordinary (unconditional) duals harvested directly from the solve.
        elapsed_ms = _run_solver(compiled.problem, choice.cp_solver, choice.name)
        _check_status()
        weight_values = np.asarray(compiled.recovered_weights(), dtype=float)
        weights = dict(zip(spec.universe, [float(x) for x in weight_values], strict=True))
        decomposition, metrics = _report_semantics(
            spec,
            compiled,
            weight_values,
            inputs,
            periods_per_year=annualization,
        )
        sensitivity_note: str | None = None
        sensitivity_unavailable_reasons: dict[str, str] = {}
        if compiled.problem.status == "optimal":
            sensitivities = _solution_sensitivities(
                compiled,
                weight_values,
                inputs,
                conditional=False,
            )
            sensitivity_unavailable_reasons = sensitivity_dependency_reasons(compiled)
        else:
            sensitivities = ()
            sensitivity_note = (
                "Sensitivities are unavailable because the continuous solve "
                f"returned status {compiled.problem.status!r}; an exact 'optimal' "
                "status is required for authoritative derivatives."
            )
        report = build_report(
            weights=weights,
            objective_kind=spec.objective.kind,
            objective_value=float(compiled.problem.value),
            solver=choice.name,
            solve_time_ms=elapsed_ms,
            status=compiled.problem.status,
            duals={},
            constraint_human_names=_human_names(spec),
            var=_var_of(spec, compiled),
            objective_decomposition=decomposition,
            metrics=metrics,
            sensitivities=sensitivities,
            sensitivity_coverage=_coverage(
                spec,
                sensitivities,
                conditional=False,
                constraint_unavailable_reasons=sensitivity_unavailable_reasons,
                unavailable_reason=sensitivity_note,
            ),
            sensitivity_note=sensitivity_note,
            termination_reason=(
                "optimal" if compiled.problem.status == "optimal" else "optimal_inaccurate"
            ),
            optimality_proven=compiled.problem.status == "optimal",
            problem_class="convex",
        )
        return compiled, report

    # Mixed-integer path. Solve the MIP, read the optimal selection y*, then run
    # the fix-and-resolve to recover *conditional* shadow prices (see below and
    # core.duals). The MIP itself yields no duals.
    elapsed_ms = _run_solver(
        compiled.problem, choice.cp_solver, choice.name, time_limit_s=time_limit_s
    )
    try:
        termination = _mip_termination(
            compiled.problem,
            cp_solver=choice.cp_solver,
            time_limit_requested=time_limit_s is not None,
        )
    except InfeasibleError as exc:
        _maybe_diagnose(exc)
    incumbent = _validate_mip_incumbent(compiled)
    gap = _optimality_gap(compiled.problem, termination)

    selected_idx = list(incumbent.selected_indices)
    selected = [spec.universe[i] for i in selected_idx]

    # A time-limited incumbent need not be optimal even with its selected names
    # fixed. Re-optimizing that restriction would attach duals from a different
    # portfolio, so time-limited reports deliberately expose no sensitivities.
    cond = (
        _fix_and_resolve(spec, selected_idx, inputs)
        if termination.optimality_proven
        else _Conditional(weights=None, sensitivities=(), unavailable_reasons={})
    )

    sensitivity_note = cond.unavailable_reason
    sensitivities = cond.sensitivities
    if cond.unavailable_reason is not None:
        sensitivities = ()
    elif termination.optimality_proven and (
        cond.weights is None
        or cond.weights.shape != incumbent.weights.shape
        or not np.allclose(
            cond.weights,
            incumbent.weights,
            rtol=1e-6,
            atol=1e-6,
        )
    ):
        # Duals from a different optimizer in a non-unique conditional problem
        # are real, but they are not local facts about the weights this report
        # exposes. Do not attach them to the MIP-authoritative portfolio.
        sensitivities = ()
        sensitivity_note = (
            "Conditional sensitivities are unavailable because fix-and-resolve "
            "returned different portfolio weights from the reported MIP solution."
        )
    elif not termination.optimality_proven:
        sensitivity_note = (
            "Sensitivities are unavailable for a time-limited incumbent because "
            "fix-and-resolve could optimize a different portfolio."
        )

    # The MIP is the sole source of the portfolio and objective auxiliaries.
    # Fix-and-resolve exists only for conditional duals; using its weights would
    # make a numerically different continuous solution the reported decision.
    weights = dict(zip(spec.universe, [float(x) for x in incumbent.weights], strict=True))
    decomposition, metrics = _report_semantics(
        spec,
        compiled,
        incumbent.weights,
        inputs,
        periods_per_year=annualization,
    )
    report = build_report(
        weights=weights,
        objective_kind=spec.objective.kind,
        objective_value=float(compiled.problem.value),
        solver=choice.name,
        solve_time_ms=elapsed_ms,
        status=compiled.problem.status,
        duals={},
        constraint_human_names=_human_names(spec),
        var=_var_of(spec, compiled),
        duals_conditional=bool(sensitivities),
        selected_names=selected,
        optimality_gap=gap,
        objective_decomposition=decomposition,
        metrics=metrics,
        sensitivities=sensitivities,
        sensitivity_coverage=_coverage(
            spec,
            sensitivities,
            conditional=bool(sensitivities),
            constraint_unavailable_reasons=cond.unavailable_reasons,
            unavailable_reason=sensitivity_note,
        ),
        sensitivity_note=sensitivity_note,
        termination_reason=termination.reason,
        optimality_proven=termination.optimality_proven,
        incumbent_validated=True,
        problem_class="mip",
    )
    return compiled, report
