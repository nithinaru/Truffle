"""Deterministic infeasibility diagnosis and verified repair generation.

The diagnostic path deliberately drops the user's objective and solves only
for feasibility.  It first minimizes *normalized* elastic slack, then deletion
filters every feasibility-bearing IR node to obtain a node-level IIS.  Positive
elastic slack is useful repair evidence, but it is not an honest IIS candidate
restriction: hard constraints and zero-slack soft constraints may still be
essential.  Every repair emitted here is applied and re-solved before it is
returned to a caller.
"""

from __future__ import annotations

import time
from collections.abc import Collection
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Literal

import cvxpy as cp
import numpy as np
import pandas as pd

from core.compiler import _cardinality_domain_is_bounded, compile_spec
from core.constraints.cardinality import Cardinality
from core.constraints.cvar_limit import CVaRLimit
from core.constraints.factor_exposure import FactorExposure
from core.constraints.group_cap import GroupCap
from core.constraints.tracking_error_cap import TrackingErrorCap
from core.constraints.transaction_cost import TransactionCost
from core.constraints.turnover_cap import TurnoverCap
from core.exceptions import DiagnosisError, TruffleError
from core.ir import Box, Budget, Constraint, MinCVaR, MinVariance, PortfolioSpec
from core.patch import SpecPatch, apply_patch
from core.report import (
    ConflictEvidence,
    ConflictMember,
    ConflictReport,
    ConstraintSlack,
    ElasticResult,
    GroundValue,
    IISResult,
    Repair,
    RepairChange,
)
from core.routing import select_solver
from core.solve import human_name_for
from data.estimation import DEFAULT_PERIODS_PER_YEAR, estimate_moments
from data.inputs import align_named
from data.scenarios import historical_scenarios

_SOLVED = {"optimal", "optimal_inaccurate"}
_PROVEN_FEASIBLE = {"optimal"}
_PROVEN_INFEASIBLE = {"infeasible"}
_SLACK_ABS_TOL = 1e-7
_SLACK_REL_TOL = 1e-6
_DOMAIN_SNAP_TOL = 1e-9
_MIP_IIS_MAX_CONSTRAINTS = 12


@dataclass(frozen=True, slots=True)
class DiagnosisData:
    """Numeric inputs aligned once and reused by every diagnostic re-solve."""

    mu: np.ndarray
    sigma: np.ndarray
    scenarios: np.ndarray | None
    w_prev: np.ndarray
    sectors: dict[str, str] | None
    benchmark_weights: dict[str, np.ndarray] | None
    factor_loadings: dict[str, np.ndarray] | None


@dataclass(frozen=True, slots=True)
class _BuiltDiagnostic:
    spec: PortfolioSpec
    compiled: object
    problem: cp.Problem
    slack_vars: dict[str, cp.Variable]
    cardinality_domain_valid: bool


@dataclass(frozen=True, slots=True)
class _CardinalityBounds:
    compiled_relaxation: object
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    relaxation_infeasible: bool = False


def prepare_diagnosis_data(
    spec: PortfolioSpec,
    prices: pd.DataFrame,
    *,
    sectors: dict[str, str] | None = None,
    benchmarks: dict[str, dict[str, float]] | None = None,
    factors: dict[str, dict[str, float]] | None = None,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> DiagnosisData:
    """Estimate and align exactly the numeric inputs diagnosis will reuse."""

    missing = sorted(set(spec.universe) - set(prices.columns))
    if missing:
        raise DiagnosisError(f"Prices are missing universe tickers: {missing}.")
    panel = prices[spec.universe]
    if periods_per_year < 1:
        raise DiagnosisError("periods_per_year must be a positive integer.")
    mu, sigma = estimate_moments(panel, periods_per_year=periods_per_year)
    needs_scenarios = isinstance(spec.objective, MinCVaR) or any(
        isinstance(c, CVaRLimit) for c in spec.constraints
    )
    scenarios = historical_scenarios(panel) if needs_scenarios else None
    return DiagnosisData(
        mu=mu,
        sigma=sigma,
        scenarios=scenarios,
        w_prev=np.asarray(spec.w_prev_vector(), dtype=float),
        sectors=sectors,
        benchmark_weights=align_named(benchmarks, spec.universe, label="Benchmark"),
        factor_loadings=align_named(factors, spec.universe, label="Factor"),
    )


def _feasibility_constraint_ids(spec: PortfolioSpec) -> list[str]:
    """Return every user constraint that can affect primal feasibility."""

    return [c.id for c in spec.constraints if not isinstance(c, TransactionCost)]


def _constraint_map(spec: PortfolioSpec) -> dict[str, Constraint]:
    return {c.id: c for c in spec.constraints}


def _proxy_spec(spec: PortfolioSpec, active_ids: Collection[str] | None) -> PortfolioSpec:
    """Replace the objective with a native-weight feasibility placeholder."""

    allowed = None if active_ids is None else set(active_ids)
    constraints = [
        c
        for c in spec.constraints
        if not isinstance(c, TransactionCost) and (allowed is None or c.id in allowed)
    ]
    return PortfolioSpec(
        universe=list(spec.universe),
        objective=MinVariance(),
        constraints=constraints,
        current_weights=spec.current_weights,
    )


def _derive_cardinality_bounds(
    proxy: PortfolioSpec,
    data: DiagnosisData,
) -> _CardinalityBounds:
    """Bound every weight over the active continuous relaxation.

    A diagnostic deletion trial can remove LongOnly or Budget while retaining
    Cardinality.  The production ``[0, 1]`` link is then no longer justified.
    We drop Cardinality, optimize each weight in both directions over the exact
    remaining convex constraints, and use those finite bounds for a two-sided
    selection link.  ``optimal_inaccurate`` and unbounded bound problems are
    deliberately inconclusive.
    """

    relaxation = PortfolioSpec(
        universe=list(proxy.universe),
        objective=MinVariance(),
        constraints=[c for c in proxy.constraints if not isinstance(c, Cardinality)],
        current_weights=proxy.current_weights,
    )
    compiled = compile_spec(
        relaxation,
        mu=data.mu,
        sigma=data.sigma,
        scenarios=data.scenarios,
        w_prev=data.w_prev,
        sectors=data.sectors,
        benchmark_weights=data.benchmark_weights,
        factor_loadings=data.factor_loadings,
    )
    if cp.CLARABEL not in cp.installed_solvers():
        return _CardinalityBounds(compiled_relaxation=compiled)

    lower = np.empty(len(proxy.universe), dtype=float)
    upper = np.empty(len(proxy.universe), dtype=float)
    constraints = list(compiled.problem.constraints)
    for index in range(len(proxy.universe)):
        for direction in ("lower", "upper"):
            objective = (
                cp.Minimize(compiled.weights[index])
                if direction == "lower"
                else cp.Maximize(compiled.weights[index])
            )
            problem = cp.Problem(objective, constraints)
            try:
                problem.solve(solver=cp.CLARABEL)
            except cp.SolverError:
                return _CardinalityBounds(compiled_relaxation=compiled)
            if problem.status in _PROVEN_INFEASIBLE:
                return _CardinalityBounds(
                    compiled_relaxation=compiled,
                    relaxation_infeasible=True,
                )
            if problem.status not in _PROVEN_FEASIBLE or problem.value is None:
                return _CardinalityBounds(compiled_relaxation=compiled)
            value = float(problem.value)
            padding = 1e-7 * max(1.0, abs(value))
            if direction == "lower":
                lower[index] = value - padding
            else:
                upper[index] = value + padding
    return _CardinalityBounds(
        compiled_relaxation=compiled,
        lower=lower,
        upper=upper,
    )


def _compile_proxy(
    spec: PortfolioSpec,
    data: DiagnosisData,
    *,
    active_ids: Collection[str] | None = None,
    relaxed_ids: Collection[str] = (),
):
    proxy = _proxy_spec(spec, active_ids)
    cardinality_domain_valid = _cardinality_domain_is_bounded(proxy)
    cardinality_bounds: tuple[np.ndarray, np.ndarray] | None = None
    if active_ids is not None and not cardinality_domain_valid:
        derived = _derive_cardinality_bounds(proxy, data)
        if derived.relaxation_infeasible:
            # The system without Cardinality is already infeasible, so this is
            # a stronger proof and the continuous compiled constraints suffice.
            return proxy, derived.compiled_relaxation, True
        if derived.lower is not None and derived.upper is not None:
            cardinality_bounds = (derived.lower, derived.upper)
            cardinality_domain_valid = True

    # If finite bounds could not be proven, an unsafe M=1 solve is still useful
    # only when it finds a feasible witness. `_feasibility_status` downgrades an
    # infeasible result to unknown via the validity bit.
    allow_unproven_domain = active_ids is not None and not cardinality_domain_valid
    compiled = compile_spec(
        proxy,
        mu=data.mu,
        sigma=data.sigma,
        scenarios=data.scenarios,
        w_prev=data.w_prev,
        sectors=data.sectors,
        benchmark_weights=data.benchmark_weights,
        factor_loadings=data.factor_loadings,
        relaxed_constraint_ids=frozenset(relaxed_ids),
        _validate_cardinality_preconditions=not allow_unproven_domain,
        _cardinality_weight_bounds=cardinality_bounds,
    )
    return proxy, compiled, cardinality_domain_valid


def _relax_constraint(con: cp.Constraint, slack: cp.Variable) -> list[cp.Constraint]:
    """Relax a compiler-owned residual by one scalar node-level slack."""

    if isinstance(con, cp.constraints.zero.Equality):
        return [con.expr <= slack, -con.expr <= slack]
    if isinstance(con, cp.constraints.nonpos.Inequality):
        return [con.expr <= slack]
    raise DiagnosisError(
        f"Constraint form {type(con).__name__} is not elasticized in this version."
    )


def _build_diagnostic_problem(
    spec: PortfolioSpec,
    data: DiagnosisData,
    *,
    active_ids: Collection[str] | None = None,
    relax_ids: Collection[str] = (),
) -> _BuiltDiagnostic:
    relax = frozenset(relax_ids)
    proxy, compiled, cardinality_domain_valid = _compile_proxy(
        spec, data, active_ids=active_ids, relaxed_ids=relax
    )
    nodes = _constraint_map(proxy)
    unknown = relax - set(nodes)
    if unknown:
        raise DiagnosisError(f"Cannot relax inactive or unknown constraints: {sorted(unknown)}.")

    reverse_named = {id(con): cid for cid, con in compiled.constraint_objs.items()}
    slack_vars: dict[str, cp.Variable] = {}
    constraints: list[cp.Constraint] = []
    penalties: list[cp.Expression] = []

    for con in compiled.problem.constraints:
        cid = reverse_named.get(id(con))
        if cid not in relax:
            constraints.append(con)
            continue
        node = nodes[cid]
        if not node.is_elastic:
            raise DiagnosisError(f"Constraint {cid!r} is not relaxable.")
        slack = cp.Variable(nonneg=True, name=f"elastic_{cid}")
        slack_vars[cid] = slack
        constraints.extend(_relax_constraint(con, slack))
        penalties.append(slack / node.natural_slack_scale())

    missing_named = relax - set(slack_vars)
    if missing_named:
        raise DiagnosisError(
            f"Relaxable constraints were not registered by the compiler: {sorted(missing_named)}."
        )

    objective_expr: cp.Expression = cp.Constant(0.0)
    for penalty in penalties:
        objective_expr = objective_expr + penalty
    problem = cp.Problem(cp.Minimize(objective_expr), constraints)
    return _BuiltDiagnostic(proxy, compiled, problem, slack_vars, cardinality_domain_valid)


def _diagnostic_solver(spec: PortfolioSpec) -> tuple[str, str]:
    """Route the dropped-objective feasibility form by its actual character."""

    if spec.problem_class != "mip":
        return cp.CLARABEL, "Clarabel"
    # Everything is linear except the natural-unit tracking-error SOC.  Small
    # ordinary cardinality diagnoses therefore use HiGHS rather than paying for
    # the original objective's MIQP solver.
    if any(isinstance(c, TrackingErrorCap) for c in spec.constraints):
        return cp.SCIP, "SCIP"
    return cp.HIGHS, "HiGHS"


def _run(problem: cp.Problem, spec: PortfolioSpec) -> tuple[str, float]:
    solver, name = _diagnostic_solver(spec)
    if solver not in cp.installed_solvers():
        raise DiagnosisError(
            f"{name} is required to diagnose this constraint system but is not installed."
        )
    start = time.perf_counter()
    try:
        problem.solve(solver=solver)
    except cp.SolverError as exc:
        raise DiagnosisError(f"{name} failed during infeasibility diagnosis: {exc}") from exc
    return name, 1000.0 * (time.perf_counter() - start)


def _positive_slack(raw: float, scale: float) -> bool:
    return raw > max(_SLACK_ABS_TOL, _SLACK_REL_TOL * scale)


def elastic_solve(
    spec: PortfolioSpec,
    data: DiagnosisData,
    *,
    relax_ids: Collection[str] | None = None,
    active_ids: Collection[str] | None = None,
    require_violation: bool = True,
) -> ElasticResult:
    """Run the normalized elastic pass, dropping the user's objective.

    ``slack / natural_slack_scale`` is minimized so heterogeneous units do
    not decide which constraint appears cheapest to violate.
    """

    nodes = _constraint_map(spec)
    if relax_ids is None:
        relax_ids = [cid for cid in _feasibility_constraint_ids(spec) if nodes[cid].is_elastic]
    built = _build_diagnostic_problem(spec, data, active_ids=active_ids, relax_ids=relax_ids)
    solver, elapsed = _run(built.problem, built.spec)
    status = built.problem.status

    if status in _PROVEN_INFEASIBLE:
        if not built.cardinality_domain_valid:
            raise DiagnosisError(
                "The pruned Cardinality system lacks its long-only/unit-budget "
                "big-M proof, so an infeasible solver status is inconclusive."
            )
        return ElasticResult(
            kind="hard_infeasible",
            status=status,
            solver=solver,
            solve_time_ms=elapsed,
        )
    if status not in _SOLVED:
        raise DiagnosisError(
            f"Elastic solve returned inconclusive status {status!r}; no conflict claim was made."
        )

    slacks: list[ConstraintSlack] = []
    candidates: list[str] = []
    total = 0.0
    for cid, variable in built.slack_vars.items():
        if variable.value is None:
            raise DiagnosisError(f"Elastic solver did not populate slack for {cid!r}.")
        raw = max(0.0, float(np.asarray(variable.value).item()))
        node = nodes[cid]
        scale = node.natural_slack_scale()
        relative = raw / scale
        total += relative
        slacks.append(
            ConstraintSlack(
                constraint_id=cid,
                human_name=human_name_for(node),
                raw_slack=raw,
                slack_scale=scale,
                relative_slack=relative,
            )
        )
        if _positive_slack(raw, scale):
            candidates.append(cid)

    if require_violation and not candidates:
        raise DiagnosisError(
            "Elastic diagnosis found zero violation; the constraint system is feasible."
        )

    weights_array = np.asarray(built.compiled.recovered_weights(), dtype=float)
    weights = dict(zip(spec.universe, weights_array.tolist(), strict=True))
    return ElasticResult(
        kind="soft_repair",
        status=status,
        solver=solver,
        solve_time_ms=elapsed,
        total_relative_slack=total,
        slacks=tuple(slacks),
        candidate_constraint_ids=tuple(candidates),
        repaired_weights=weights,
    )


def _feasibility_status(
    spec: PortfolioSpec, data: DiagnosisData, active_ids: Collection[str]
) -> Literal["feasible", "infeasible", "unknown"]:
    built = _build_diagnostic_problem(spec, data, active_ids=active_ids)
    _run(built.problem, built.spec)
    if built.problem.status in _PROVEN_FEASIBLE:
        return "feasible"
    if built.problem.status in _PROVEN_INFEASIBLE:
        if not built.cardinality_domain_valid:
            return "unknown"
        return "infeasible"
    return "unknown"


def find_iis(
    spec: PortfolioSpec,
    data: DiagnosisData,
    candidates: Collection[str] = (),
) -> IISResult:
    """Deletion-filter all feasibility nodes into one verified node-level IIS.

    Elastic candidates only influence deletion order.  Restricting membership
    to positive slacks would omit hard Budget rows and zero-slack-but-essential
    soft constraints, so every feasibility-bearing IR node is eligible.
    """

    active = _feasibility_constraint_ids(spec)
    if _feasibility_status(spec, data, active) != "infeasible":
        raise DiagnosisError("IIS search requires a proven-infeasible constraint system.")

    if spec.problem_class == "mip" and len(active) > _MIP_IIS_MAX_CONSTRAINTS:
        fallback = tuple(dict.fromkeys([*candidates, *active]))
        return IISResult(
            constraint_ids=fallback,
            verified=False,
            checks=1,
            fallback_reason=(
                "The mixed-integer conflict set was too large for bounded deletion filtering."
            ),
        )

    priority = set(candidates)
    order = [cid for cid in active if cid not in priority] + [
        cid for cid in active if cid in priority
    ]
    current = list(active)
    checks = 1
    for cid in order:
        if cid not in current:
            continue
        trial = [candidate for candidate in current if candidate != cid]
        status = _feasibility_status(spec, data, trial)
        checks += 1
        if status == "infeasible":
            current = trial
        elif status == "unknown":
            return IISResult(
                constraint_ids=tuple(current),
                verified=False,
                checks=checks,
                fallback_reason=f"Deletion check for {cid!r} was inconclusive.",
            )

    # Defensive proof check: the survivor must be infeasible and removing any
    # one survivor must be feasible before the report may use the word IIS.
    if _feasibility_status(spec, data, current) != "infeasible":
        return IISResult(
            constraint_ids=tuple(current),
            verified=False,
            checks=checks + 1,
            fallback_reason="Survivor was not infeasible.",
        )
    checks += 1
    for cid in current:
        trial = [candidate for candidate in current if candidate != cid]
        status = _feasibility_status(spec, data, trial)
        checks += 1
        if status != "feasible":
            return IISResult(
                constraint_ids=tuple(current),
                verified=False,
                checks=checks,
                fallback_reason=f"Minimality check for {cid!r} was not proven feasible.",
            )
    return IISResult(constraint_ids=tuple(current), verified=True, checks=checks)


def _snap_to_domain_endpoint(
    value: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    """Remove solver dust just outside a validated field's closed domain."""

    if lower is not None and lower - _DOMAIN_SNAP_TOL <= value < lower:
        return lower
    if upper is not None and upper < value <= upper + _DOMAIN_SNAP_TOL:
        return upper
    return value


def _round_outward(value: float, quantum: float, direction: Literal["raise", "lower"]) -> float:
    """Round a bound in the direction that preserves repaired feasibility."""

    decimal_value = Decimal(str(value))
    step = Decimal(str(quantum))
    rounding = ROUND_CEILING if direction == "raise" else ROUND_FLOOR
    units = (decimal_value / step).to_integral_value(rounding=rounding)
    return float(units * step)


def _percent(value: float) -> str:
    return f"{100.0 * value:g}%"


def _target_weights(node: Box, spec: PortfolioSpec, weights: dict[str, float]) -> list[float]:
    tickers = spec.universe if node.tickers is None else node.tickers
    return [weights[ticker] for ticker in tickers]


def _changes_for_constraint(
    node: Constraint,
    raw_slack: float,
    spec: PortfolioSpec,
    data: DiagnosisData,
    weights: dict[str, float],
) -> tuple[Constraint, tuple[RepairChange, ...], str] | None:
    """Turn one elastic optimum into a safe-rounded replacement node."""

    scale = node.natural_slack_scale()
    changes: list[RepairChange] = []
    updates: dict[str, float | int] = {}

    def add_change(
        field: str,
        old: float,
        required: float,
        direction: Literal["raise", "lower"],
        unit: Literal["raw", "fraction", "count"],
        quantum: float,
        domain_lower: float | None = None,
        domain_upper: float | None = None,
    ) -> None:
        required = _snap_to_domain_endpoint(
            required,
            lower=domain_lower,
            upper=domain_upper,
        )
        applied = _round_outward(required, quantum, direction)
        applied = _snap_to_domain_endpoint(
            applied,
            lower=domain_lower,
            upper=domain_upper,
        )
        delta = abs(required - old)
        updates[field] = int(applied) if unit == "count" else applied
        changes.append(
            RepairChange(
                constraint_id=node.id,
                field=field,
                direction=direction,
                old_value=float(old),
                solver_required_value=float(required),
                applied_value=float(applied),
                required_change=float(delta),
                normalized_change=float(delta / scale),
                unit=unit,
            )
        )

    if isinstance(node, Box):
        target = _target_weights(node, spec, weights)
        required_upper = max(target)
        required_lower = min(target)
        if required_upper > node.upper + _SLACK_ABS_TOL:
            add_change("upper", node.upper, required_upper, "raise", "fraction", 0.01)
        if required_lower < node.lower - _SLACK_ABS_TOL:
            add_change("lower", node.lower, required_lower, "lower", "fraction", 0.01)
        scope = "per-name position" if node.tickers is None else "/".join(node.tickers)
        description_subject = f"{scope} bounds"
    elif isinstance(node, GroupCap):
        if data.sectors is None:
            return None
        group_weight = sum(
            weights[ticker] for ticker in spec.universe if data.sectors.get(ticker) == node.group
        )
        if group_weight > node.max_weight + _SLACK_ABS_TOL:
            add_change(
                "max_weight",
                node.max_weight,
                group_weight,
                "raise",
                "fraction",
                0.01,
                domain_lower=0.0,
                domain_upper=1.0,
            )
        if node.min_weight is not None and group_weight < node.min_weight - _SLACK_ABS_TOL:
            add_change(
                "min_weight",
                node.min_weight,
                group_weight,
                "lower",
                "fraction",
                0.01,
                domain_lower=0.0,
                domain_upper=1.0,
            )
        description_subject = f"{node.group} group bound"
    elif isinstance(node, TurnoverCap):
        turnover = float(
            np.sum(np.abs(np.array([weights[t] for t in spec.universe], dtype=float) - data.w_prev))
        )
        add_change("max_turnover", node.max_turnover, turnover, "raise", "fraction", 0.01)
        description_subject = "turnover cap"
    elif isinstance(node, CVaRLimit):
        required = node.max_cvar + raw_slack
        add_change("max_cvar", node.max_cvar, required, "raise", "fraction", 0.001)
        description_subject = "CVaR limit"
    elif isinstance(node, TrackingErrorCap):
        required = node.max_te + raw_slack
        add_change("max_te", node.max_te, required, "raise", "fraction", 0.001)
        description_subject = "tracking-error cap"
    elif isinstance(node, FactorExposure):
        if data.factor_loadings is None or node.factor not in data.factor_loadings:
            return None
        exposure = float(
            data.factor_loadings[node.factor]
            @ np.array([weights[t] for t in spec.universe], dtype=float)
        )
        if node.max_exposure is not None and exposure > node.max_exposure + _SLACK_ABS_TOL:
            add_change("max_exposure", node.max_exposure, exposure, "raise", "raw", 0.01)
        if node.min_exposure is not None and exposure < node.min_exposure - _SLACK_ABS_TOL:
            add_change("min_exposure", node.min_exposure, exposure, "lower", "raw", 0.01)
        description_subject = f"{node.factor} factor bound"
    elif isinstance(node, Cardinality):
        required = node.max_names + raw_slack
        add_change("max_names", node.max_names, required, "raise", "count", 1.0)
        description_subject = "maximum holding count"
    else:
        return None

    if not changes:
        return None
    # ``model_copy(update=...)`` deliberately skips Pydantic validation.  A
    # solver-derived suggestion must pass the same node validation as user
    # input before it can become a trusted repair (for example, GroupCap may
    # not be raised above 100%, and a cardinality floor may not exceed its
    # newly suggested cap).
    try:
        replacement = type(node).model_validate({**node.model_dump(), **updates})
    except ValueError:
        return None
    rendered: list[str] = []
    for change in changes:
        if change.unit == "fraction":
            old_text = _percent(change.old_value)
            new_text = _percent(change.applied_value)
        else:
            old_text = f"{change.old_value:g}"
            new_text = f"{change.applied_value:g}"
        verb = "Raise" if change.direction == "raise" else "Lower"
        rendered.append(f"{verb} {change.field} from {old_text} to {new_text}")
    description = f"{description_subject}: " + "; ".join(rendered) + "."
    return replacement, tuple(changes), description


def _compile_original(spec: PortfolioSpec, data: DiagnosisData):
    return compile_spec(
        spec,
        mu=data.mu,
        sigma=data.sigma,
        scenarios=data.scenarios,
        w_prev=data.w_prev,
        sectors=data.sectors,
        benchmark_weights=data.benchmark_weights,
        factor_loadings=data.factor_loadings,
    )


def _verified_feasible(spec: PortfolioSpec, data: DiagnosisData) -> bool:
    """Run the actual production objective; a suggested patch must solve."""

    try:
        compiled = _compile_original(spec, data)
        choice = select_solver(spec)
        compiled.problem.solve(solver=choice.cp_solver)
    except (cp.SolverError, TruffleError, ValueError):
        return False
    # ``optimal_inaccurate`` is useful as a diagnostic hint, but without an
    # explicit primal-residual audit it is not proof that a user-facing repair
    # is feasible.
    return compiled.problem.status in _PROVEN_FEASIBLE


def _make_repair(
    spec: PortfolioSpec,
    data: DiagnosisData,
    node: Constraint,
    elastic: ElasticResult,
    *,
    kind: Literal["single_lever", "joint"] = "single_lever",
) -> Repair | None:
    if elastic.repaired_weights is None:
        return None
    slack_by_id = {slack.constraint_id: slack for slack in elastic.slacks}
    slack = slack_by_id.get(node.id)
    if slack is None or not _positive_slack(slack.raw_slack, slack.slack_scale):
        return None
    changed = _changes_for_constraint(node, slack.raw_slack, spec, data, elastic.repaired_weights)
    if changed is None:
        return None
    replacement, changes, description = changed
    patch = SpecPatch(
        remove_constraint_ids=[node.id],
        add_constraints=[replacement],
    )
    repaired = apply_patch(spec, patch)
    if not _verified_feasible(repaired, data):
        return None
    relative = sum(change.normalized_change for change in changes)
    return Repair(
        repair_id="pending",
        description=description,
        patch=patch,
        changes=changes,
        required_change=slack.raw_slack if kind == "single_lever" else None,
        relative_change=relative,
        kind=kind,
        rank=1,
    )


def single_lever_repairs(
    spec: PortfolioSpec,
    data: DiagnosisData,
    iis: IISResult,
) -> list[Repair]:
    """Find verified changes where relaxing one IIS member is sufficient."""

    nodes = _constraint_map(spec)
    repairs: list[Repair] = []
    for cid in iis.constraint_ids:
        node = nodes[cid]
        if not node.is_elastic:
            continue
        elastic = elastic_solve(spec, data, relax_ids=[cid], require_violation=True)
        if elastic.kind != "soft_repair":
            continue
        repair = _make_repair(spec, data, node, elastic)
        if repair is not None:
            repairs.append(repair)
    repairs.sort(key=lambda repair: (repair.relative_change, repair.description))
    return repairs


def _joint_repair(
    spec: PortfolioSpec, data: DiagnosisData, elastic: ElasticResult
) -> Repair | None:
    nodes = _constraint_map(spec)
    replacements: list[Constraint] = []
    changes: list[RepairChange] = []
    descriptions: list[str] = []
    for cid in elastic.candidate_constraint_ids:
        node = nodes[cid]
        if elastic.repaired_weights is None:
            continue
        slack = next(s for s in elastic.slacks if s.constraint_id == cid)
        changed = _changes_for_constraint(
            node, slack.raw_slack, spec, data, elastic.repaired_weights
        )
        if changed is None:
            continue
        replacement, node_changes, description = changed
        replacements.append(replacement)
        changes.extend(node_changes)
        descriptions.append(description.rstrip("."))
    if len(replacements) < 2:
        return None
    patch = SpecPatch(
        remove_constraint_ids=[node.id for node in replacements],
        add_constraints=replacements,
    )
    if not _verified_feasible(apply_patch(spec, patch), data):
        return None
    return Repair(
        repair_id="pending",
        description="Joint repair: " + "; ".join(descriptions) + ".",
        patch=patch,
        changes=tuple(changes),
        required_change=None,
        relative_change=sum(change.normalized_change for change in changes),
        kind="joint",
        rank=1,
    )


def _constraint_parameters(node: Constraint, spec: PortfolioSpec) -> tuple[GroundValue, ...]:
    values: list[GroundValue] = []
    if isinstance(node, Budget):
        values.append(
            GroundValue(key="budget_total", value=node.total, unit="fraction", source="spec")
        )
    elif isinstance(node, Box):
        values.extend(
            [
                GroundValue(key="lower", value=node.lower, unit="fraction", source="spec"),
                GroundValue(key="upper", value=node.upper, unit="fraction", source="spec"),
                GroundValue(
                    key="covered_assets",
                    value=float(len(spec.universe if node.tickers is None else node.tickers)),
                    unit="count",
                    source="spec",
                ),
            ]
        )
    elif isinstance(node, GroupCap):
        values.append(
            GroundValue(key="max_weight", value=node.max_weight, unit="fraction", source="spec")
        )
        if node.min_weight is not None:
            values.append(
                GroundValue(key="min_weight", value=node.min_weight, unit="fraction", source="spec")
            )
    elif isinstance(node, TurnoverCap):
        values.append(
            GroundValue(key="max_turnover", value=node.max_turnover, unit="fraction", source="spec")
        )
    elif isinstance(node, CVaRLimit):
        values.extend(
            [
                GroundValue(key="alpha", value=node.alpha, unit="fraction", source="spec"),
                GroundValue(key="max_cvar", value=node.max_cvar, unit="fraction", source="spec"),
            ]
        )
    elif isinstance(node, TrackingErrorCap):
        values.append(GroundValue(key="max_te", value=node.max_te, unit="fraction", source="spec"))
    elif isinstance(node, FactorExposure):
        if node.min_exposure is not None:
            values.append(GroundValue(key="min_exposure", value=node.min_exposure, source="spec"))
        if node.max_exposure is not None:
            values.append(GroundValue(key="max_exposure", value=node.max_exposure, source="spec"))
    elif isinstance(node, Cardinality):
        values.append(
            GroundValue(key="max_names", value=float(node.max_names), unit="count", source="spec")
        )
        if node.min_names is not None:
            values.append(
                GroundValue(
                    key="min_names", value=float(node.min_names), unit="count", source="spec"
                )
            )
        if node.min_position is not None:
            values.append(
                GroundValue(
                    key="min_position", value=node.min_position, unit="fraction", source="spec"
                )
            )
    return tuple(values)


def _evidence(spec: PortfolioSpec, conflict_ids: Collection[str]) -> tuple[ConflictEvidence, ...]:
    nodes = _constraint_map(spec)
    conflict = [nodes[cid] for cid in conflict_ids]
    values = [
        GroundValue(
            key="universe_size",
            value=float(len(spec.universe)),
            unit="count",
            source="spec",
        )
    ]
    evidence: list[ConflictEvidence] = [
        ConflictEvidence(
            text="The confirmed constraints have no common feasible allocation.",
            values=tuple(values),
        )
    ]
    budget = next((node for node in conflict if isinstance(node, Budget)), None)
    box = next(
        (node for node in conflict if isinstance(node, Box) and node.tickers is None),
        None,
    )
    card = next((node for node in conflict if isinstance(node, Cardinality)), None)
    if budget is not None and box is not None:
        usable_names = (
            len(spec.universe) if card is None else min(len(spec.universe), card.max_names)
        )
        maximum = float(usable_names * box.upper)
        minimum = float(len(spec.universe) * box.lower)
        if maximum < budget.total - _SLACK_ABS_TOL:
            text = (
                f"At most {usable_names} names can carry {_percent(box.upper)} each, "
                f"so their maximum total is {_percent(maximum)} versus a "
                f"{_percent(budget.total)} budget."
            )
            fact_values = [
                GroundValue(
                    key="usable_names", value=float(usable_names), unit="count", source="derived"
                ),
                GroundValue(key="position_cap", value=box.upper, unit="fraction", source="spec"),
                GroundValue(key="maximum_total", value=maximum, unit="fraction", source="derived"),
                GroundValue(key="budget_total", value=budget.total, unit="fraction", source="spec"),
            ]
            evidence.append(ConflictEvidence(text=text, values=tuple(fact_values)))
        elif minimum > budget.total + _SLACK_ABS_TOL:
            text = (
                f"All {len(spec.universe)} names must carry at least "
                f"{_percent(box.lower)} each, so their minimum total is "
                f"{_percent(minimum)} versus a {_percent(budget.total)} budget."
            )
            fact_values = [
                GroundValue(
                    key="covered_names",
                    value=float(len(spec.universe)),
                    unit="count",
                    source="derived",
                ),
                GroundValue(key="position_floor", value=box.lower, unit="fraction", source="spec"),
                GroundValue(key="minimum_total", value=minimum, unit="fraction", source="derived"),
                GroundValue(key="budget_total", value=budget.total, unit="fraction", source="spec"),
            ]
            evidence.append(ConflictEvidence(text=text, values=tuple(fact_values)))
    return tuple(evidence)


def diagnose(
    spec: PortfolioSpec,
    data: DiagnosisData,
) -> ConflictReport:
    """Diagnose one already-infeasible spec and return grounded repair data."""

    elastic = elastic_solve(spec, data)
    iis = find_iis(spec, data, elastic.candidate_constraint_ids)
    nodes = _constraint_map(spec)
    slack_by_id = {slack.constraint_id: slack for slack in elastic.slacks}
    conflict_ids = iis.constraint_ids
    members: list[ConflictMember] = []
    for cid in conflict_ids:
        node = nodes[cid]
        slack = slack_by_id.get(cid)
        if node.is_elastic:
            relaxability: Literal["relaxable", "structural", "user_locked"] = "relaxable"
        elif node.elastic is False and node.elasticity_supported:
            relaxability = "user_locked"
        else:
            relaxability = "structural"
        members.append(
            ConflictMember(
                constraint_id=cid,
                constraint_kind=node.kind,
                human_name=human_name_for(node),
                relaxability=relaxability,
                required_slack=None if slack is None else slack.raw_slack,
                slack_scale=None if slack is None else slack.slack_scale,
                relative_slack=None if slack is None else slack.relative_slack,
                parameters=_constraint_parameters(node, spec),
            )
        )

    relaxable_count = sum(member.relaxability == "relaxable" for member in members)
    if relaxable_count == 0:
        scope: Literal["soft_only", "mixed", "hard_only"] = "hard_only"
    elif relaxable_count == len(members):
        scope = "soft_only"
    else:
        scope = "mixed"

    repairs: list[Repair] = []
    if iis.verified:
        repairs = single_lever_repairs(spec, data, iis)
        if not repairs:
            joint = _joint_repair(spec, data, elastic)
            if joint is not None:
                repairs.append(joint)
    repairs.sort(key=lambda repair: (repair.kind == "joint", repair.relative_change))
    repairs = [
        repair.model_copy(update={"repair_id": f"repair_{index}", "rank": index})
        for index, repair in enumerate(repairs[:4], start=1)
    ]

    return ConflictReport(
        solver_status="infeasible",
        n_assets=len(spec.universe),
        minimality_status="verified_iis" if iis.verified else "unverified_candidate",
        conflict_scope=scope,
        candidate_constraint_ids=elastic.candidate_constraint_ids,
        conflict_set=tuple(members),
        elastic=elastic,
        evidence=_evidence(spec, conflict_ids),
        repairs=tuple(repairs),
    )


def diagnose_infeasibility(
    spec: PortfolioSpec,
    prices: pd.DataFrame,
    *,
    sectors: dict[str, str] | None = None,
    benchmarks: dict[str, dict[str, float]] | None = None,
    factors: dict[str, dict[str, float]] | None = None,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> ConflictReport:
    """Public convenience wrapper: prepare numeric inputs, then diagnose."""

    data = prepare_diagnosis_data(
        spec,
        prices,
        sectors=sectors,
        benchmarks=benchmarks,
        factors=factors,
        periods_per_year=periods_per_year,
    )
    return diagnose(spec, data)
