"""Typed, row-aware sensitivity harvesting.

CVXPY attaches a dual to a *solver row*.  An IR constraint can compile to one
row (Budget, TurnoverCap), one row per asset (LongOnly), or a stacked vector of
lower and upper rows (Box, GroupCap, FactorExposure).  Collapsing those arrays
to one magnitude loses the affected asset, the side of the bound, the sign of
the derivative, and its units.  This module preserves all of that information.

``raw_solver_dual`` is the multiplier returned by CVXPY.  The reported
``objective_derivative_per_bound_unit`` is instead the signed derivative with
respect to the user-facing bound:

* equality: ``-dual``;
* lower bound: ``+dual * parameter_scale``;
* upper bound: ``-dual * parameter_scale``.

The scale is one for ordinary portfolio variables.  Max-Sharpe's homogenized
weight bounds are expressed in the transformed variable ``y`` and therefore
use ``kappa = sum(y)`` to convert a derivative back to one unit of portfolio
weight.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import cvxpy as cp
import numpy as np

from core.compile_context import CompiledProblem
from core.exceptions import DualsUnavailableError
from core.ir import (
    Box,
    Budget,
    Cardinality,
    CVaRLimit,
    FactorExposure,
    GroupCap,
    LongOnly,
    PortfolioSpec,
    TrackingErrorCap,
    TransactionCost,
    TurnoverCap,
)

SensitivitySide = Literal["lower", "upper", "equality"]
SensitivityAvailability = Literal["available", "conditional", "unavailable"]

_BINDING_REL_TOL = 1e-6
_ACTIVE_ROW_ABS_TOL = 1e-5
_DUAL_ACTIVITY_TOL = 1e-8
_DEPENDENCY_REL_TOL = 1e-9

_DEPENDENT_ROW_REASON = (
    "Active solver rows are linearly dependent, so the individual dual is "
    "non-unique, the bound derivative is non-identifiable, and no authoritative "
    "derivative is reported."
)
_UNVERIFIED_ROW_REASON = (
    "The active solver row's local gradient could not be validated, so no "
    "authoritative bound derivative is reported."
)


@dataclass(frozen=True, slots=True)
class SensitivityRecord:
    """One user-facing constraint row and its signed local derivative."""

    constraint_id: str
    kind: str
    row_index: int
    row_label: str
    side: SensitivitySide
    bound_value: float
    bound_unit: str
    raw_solver_dual: float
    parameter_scale: float
    objective_derivative_per_bound_unit: float
    objective_unit: str
    primal_value: float
    slack: float
    is_binding: bool
    conditional: bool

    @property
    def shadow_price(self) -> float:
        """Magnitude of the signed objective derivative for ranking only."""

        return abs(self.objective_derivative_per_bound_unit)


@dataclass(frozen=True, slots=True)
class SensitivityCoverage:
    """Whether one IR constraint has an interpretable sensitivity result."""

    availability: SensitivityAvailability
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class _Row:
    label: str
    side: SensitivitySide
    bound: float
    bound_unit: str
    primal: float
    parameter_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class _ActiveJacobianRow:
    """One active scalar solver row in the full CVXPY variable space."""

    constraint_id: str | None
    gradient: np.ndarray


def _check_solved(compiled: CompiledProblem) -> None:
    status = compiled.problem.status
    if status is None:
        raise DualsUnavailableError(
            "Problem has not been solved yet — call problem.solve(...) before "
            "harvest_sensitivities."
        )
    if status != "optimal":
        raise DualsUnavailableError(
            "Sensitivity rows require solver status 'optimal' from an optimal "
            f"continuous solve; solver status was {status!r}."
        )


def _dependent_row_mask(matrix: np.ndarray) -> np.ndarray:
    """Rows participating in any dependence, from one left-nullspace SVD."""

    if matrix.size == 0:
        return np.zeros(matrix.shape[0], dtype=bool)
    try:
        left_vectors, singular_values, _ = np.linalg.svd(
            matrix,
            full_matrices=True,
        )
    except np.linalg.LinAlgError:
        # Rank could not be certified. Suppress all active named rows rather
        # than treating an arbitrary multiplier as authoritative.
        return np.ones(matrix.shape[0], dtype=bool)
    if (
        singular_values.size == 0
        or not np.all(np.isfinite(singular_values))
        or not np.all(np.isfinite(left_vectors))
    ):
        return np.ones(matrix.shape[0], dtype=bool)
    tolerance = (
        _DEPENDENCY_REL_TOL
        * max(matrix.shape)
        * max(1.0, float(singular_values[0]))
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank == matrix.shape[0]:
        return np.zeros(matrix.shape[0], dtype=bool)

    # A row participates in a linear dependence iff at least one vector in
    # null(A.T) has a nonzero coefficient at that row. This identifies every
    # member of every dependent circuit with one SVD, instead of an SVD per row.
    left_nullspace = left_vectors[:, rank:]
    participation = np.linalg.norm(left_nullspace, axis=1)
    coefficient_tolerance = _DEPENDENCY_REL_TOL * max(matrix.shape)
    return participation > coefficient_tolerance


def _flat_dual(constraint: cp.Constraint, size: int) -> np.ndarray | None:
    """Best-effort scalar-row duals used only to recognize active rows."""

    value = constraint.dual_value
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    if array.size != size or not np.all(np.isfinite(array)):
        return None
    return array


def _active_mask(constraint: cp.Constraint, values: np.ndarray) -> np.ndarray | None:
    """Return active scalar elements for ordinary equality/inequality rows."""

    if isinstance(constraint, cp.constraints.Equality):
        return np.ones(values.size, dtype=bool)
    if not isinstance(
        constraint,
        (cp.constraints.Inequality, cp.constraints.NonNeg),
    ):
        return None
    dual = _flat_dual(constraint, values.size)
    # Use a slightly wider fail-closed threshold than the presentation-layer
    # binding flag. A nearly active duplicate can still destabilize the dual
    # split even when its residual lands just outside the display tolerance.
    active = np.abs(values) <= _ACTIVE_ROW_ABS_TOL
    if dual is not None:
        active |= np.abs(dual) > _DUAL_ACTIVITY_TOL
    return active


def _expression_row_gradient(
    expression: cp.Expression,
    row_index: int,
    *,
    variable_offsets: Mapping[int, tuple[int, int]],
    total_variable_size: int,
) -> np.ndarray | None:
    """Lift one expression-element gradient into the full variable space."""

    try:
        gradients = expression.grad
    except Exception:
        return None
    if gradients is None:
        return None

    result = np.zeros(total_variable_size, dtype=float)
    for variable, jacobian in gradients.items():
        if jacobian is None or variable.id not in variable_offsets:
            return None
        start, size = variable_offsets[variable.id]
        try:
            dense = np.asarray(jacobian.toarray(), dtype=float)
        except AttributeError:
            try:
                dense = np.asarray(jacobian, dtype=float)
            except (TypeError, ValueError, OverflowError):
                return None
        except (TypeError, ValueError, OverflowError):
            return None
        if dense.shape != (size, expression.size) or not np.all(np.isfinite(dense)):
            return None
        result[start : start + size] = dense[:, row_index]
    if not np.all(np.isfinite(result)):
        return None
    return result


def sensitivity_dependency_reasons(
    compiled: CompiledProblem,
) -> dict[str, str]:
    """Identify named constraints whose active-row duals are not authoritative.

    A solver may return one arbitrary multiplier split when active scalar rows
    are linearly dependent (for example ``LongOnly`` plus ``Box(lower=0)``).
    Such a multiplier is not the unique directional derivative of that bound.
    We inspect the active-row Jacobian in the *complete* CVXPY variable space,
    including auxiliary and synthetic conditional rows, and fail closed for
    every named constraint participating in a dependence.

    Lightweight mocked ``CompiledProblem`` objects used by semantic unit tests
    do not expose CVXPY problem introspection; those retain their supplied dual
    fixtures and return no dependency reasons.
    """

    problem = compiled.problem
    constraints = getattr(problem, "constraints", None)
    variables_method = getattr(problem, "variables", None)
    if constraints is None or not callable(variables_method):
        return {}
    try:
        variables = list(variables_method())
    except Exception:
        return {}
    if not variables:
        return {}

    variable_offsets: dict[int, tuple[int, int]] = {}
    total_variable_size = 0
    for variable in variables:
        size = int(variable.size)
        variable_offsets[variable.id] = (total_variable_size, size)
        total_variable_size += size

    named_ids = {id(value): key for key, value in compiled.constraint_objs.items()}
    active_rows: list[_ActiveJacobianRow] = []
    unavailable: dict[str, str] = {}

    for solver_constraint in constraints:
        expression = getattr(solver_constraint, "expr", None)
        if expression is None or expression.value is None:
            continue
        try:
            values = np.asarray(expression.value, dtype=float).reshape(-1)
        except (TypeError, ValueError, OverflowError):
            continue
        if values.size != expression.size or not np.all(np.isfinite(values)):
            continue
        active = _active_mask(solver_constraint, values)
        if active is None:
            continue
        constraint_id = named_ids.get(id(solver_constraint))
        for row_index in np.flatnonzero(active):
            gradient = _expression_row_gradient(
                expression,
                int(row_index),
                variable_offsets=variable_offsets,
                total_variable_size=total_variable_size,
            )
            if gradient is None:
                if constraint_id is not None:
                    unavailable[constraint_id] = _UNVERIFIED_ROW_REASON
                continue
            norm = float(np.linalg.norm(gradient))
            if not math.isfinite(norm) or norm <= _DEPENDENCY_REL_TOL:
                if constraint_id is not None:
                    unavailable[constraint_id] = _UNVERIFIED_ROW_REASON
                continue
            active_rows.append(
                _ActiveJacobianRow(
                    constraint_id=constraint_id,
                    gradient=gradient / norm,
                )
            )

    if not active_rows:
        return unavailable
    matrix = np.vstack([row.gradient for row in active_rows])
    for row, is_dependent in zip(
        active_rows,
        _dependent_row_mask(matrix),
        strict=True,
    ):
        if is_dependent and row.constraint_id is not None:
            unavailable[row.constraint_id] = _DEPENDENT_ROW_REASON
    return unavailable


def _finite_vector(value: object, n: int, *, label: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DualsUnavailableError(f"{label} must be a numeric vector.") from exc
    if vector.shape != (n,):
        raise DualsUnavailableError(
            f"{label} shape {vector.shape} does not match universe shape ({n},)."
        )
    if not np.all(np.isfinite(vector)):
        raise DualsUnavailableError(f"{label} must contain only finite values.")
    return vector


def _finite_matrix(
    value: object | None,
    shape: tuple[int, int] | None,
    *,
    label: str,
) -> np.ndarray:
    if value is None:
        raise DualsUnavailableError(f"{label} is required to compute sensitivity slack.")
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DualsUnavailableError(f"{label} must be a numeric matrix.") from exc
    if matrix.ndim != 2 or (shape is not None and matrix.shape != shape):
        expected = "a two-dimensional matrix" if shape is None else f"shape {shape}"
        raise DualsUnavailableError(f"{label} must have {expected}; got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise DualsUnavailableError(f"{label} must contain only finite values.")
    return matrix


def _aligned_vector(
    values: Mapping[str, np.ndarray] | None,
    name: str,
    n: int,
    *,
    label: str,
) -> np.ndarray:
    if values is None or name not in values:
        raise DualsUnavailableError(
            f"Aligned {label} data for {name!r} is required to compute sensitivity slack."
        )
    return _finite_vector(values[name], n, label=f"Aligned {label} {name!r}")


def _objective_unit(spec: PortfolioSpec) -> str:
    if any(isinstance(c, TransactionCost) for c in spec.constraints):
        return "composite_objective_score"
    return {
        "min_variance": "annualized_variance",
        "mean_variance": "mean_variance_score",
        "min_cvar": "scenario_loss_fraction",
        "max_sharpe": "inverse_sharpe_squared",
        "risk_parity": "risk_parity_surrogate",
        "min_tracking_error": "annualized_tracking_variance",
    }[spec.objective.kind]


def _dual_vector(value: object, expected: int, *, constraint_id: str) -> np.ndarray:
    if value is None:
        raise DualsUnavailableError(
            f"Constraint {constraint_id!r} has no dual value after the continuous solve."
        )
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DualsUnavailableError(
            f"Constraint {constraint_id!r} has a non-numeric dual value."
        ) from exc
    valid_shape = array.shape in {(), (1,)} if expected == 1 else array.shape == (expected,)
    if not valid_shape:
        expected_shape = "scalar" if expected == 1 else f"({expected},)"
        raise DualsUnavailableError(
            f"Constraint {constraint_id!r} dual shape {array.shape} does not match "
            f"the expected {expected_shape} solver rows."
        )
    array = array.reshape(expected)
    if not np.all(np.isfinite(array)):
        raise DualsUnavailableError(
            f"Constraint {constraint_id!r} dual contains a non-finite value."
        )
    return array


def _max_sharpe_scale(compiled: CompiledProblem) -> float:
    raw = compiled.weights.value
    if raw is None:
        raise DualsUnavailableError(
            "Max-Sharpe solve did not populate its transformed weight vector."
        )
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DualsUnavailableError("Max-Sharpe transformed weights must be numeric.") from exc
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise DualsUnavailableError("Max-Sharpe transformed weights must be a finite vector.")
    scale = math.fsum(float(value) for value in values)
    if not math.isfinite(scale) or scale <= 0.0:
        raise DualsUnavailableError(
            "Max-Sharpe sensitivity scaling requires positive finite kappa=sum(y)."
        )
    return scale


def _empirical_cvar(losses: np.ndarray, alpha: float) -> float:
    """Exact equally-weighted empirical expected shortfall for fixed weights."""

    descending = np.sort(losses)[::-1]
    tail_count = (1.0 - alpha) * descending.size
    whole = int(math.floor(tail_count))
    fraction = tail_count - whole
    total = math.fsum(float(value) for value in descending[:whole])
    if fraction > 0.0:
        # When tail_count < 1 this correctly returns the single worst loss.
        total += fraction * float(descending[whole])
    value = total / tail_count
    if not math.isfinite(value):
        raise DualsUnavailableError("Computed CVaR sensitivity primal is non-finite.")
    return value


def _rows_for_constraint(
    constraint: object,
    *,
    spec: PortfolioSpec,
    weights: np.ndarray,
    w_prev: np.ndarray,
    sigma: np.ndarray,
    scenarios: np.ndarray | None,
    sectors: Mapping[str, str] | None,
    benchmarks: Mapping[str, np.ndarray] | None,
    factors: Mapping[str, np.ndarray] | None,
    max_sharpe_scale: float,
) -> list[_Row]:
    universe = spec.universe
    n = len(universe)
    is_max_sharpe = spec.objective.kind == "max_sharpe"

    if isinstance(constraint, Budget):
        return [
            _Row(
                label="portfolio",
                side="equality",
                bound=constraint.total,
                bound_unit="portfolio_weight_fraction",
                primal=math.fsum(float(value) for value in weights),
            )
        ]

    if isinstance(constraint, LongOnly):
        scale = max_sharpe_scale if is_max_sharpe else 1.0
        return [
            _Row(
                label=ticker,
                side="lower",
                bound=0.0,
                bound_unit="portfolio_weight_fraction",
                primal=float(weights[index]),
                parameter_scale=scale,
            )
            for index, ticker in enumerate(universe)
        ]

    if isinstance(constraint, Box):
        tickers = universe if constraint.tickers is None else constraint.tickers
        ticker_index = {ticker: index for index, ticker in enumerate(universe)}
        scale = max_sharpe_scale if is_max_sharpe else 1.0
        lower = [
            _Row(
                label=ticker,
                side="lower",
                bound=constraint.lower,
                bound_unit="portfolio_weight_fraction",
                primal=float(weights[ticker_index[ticker]]),
                parameter_scale=scale,
            )
            for ticker in tickers
        ]
        upper = [
            _Row(
                label=ticker,
                side="upper",
                bound=constraint.upper,
                bound_unit="portfolio_weight_fraction",
                primal=float(weights[ticker_index[ticker]]),
                parameter_scale=scale,
            )
            for ticker in tickers
        ]
        # The transformed compiler stacks upper then lower; ordinary Box stacks
        # lower then upper.  Preserve solver ordering exactly.
        return upper + lower if is_max_sharpe else lower + upper

    if isinstance(constraint, GroupCap):
        if sectors is None:
            raise DualsUnavailableError(
                f"Sector mapping is required to compute GroupCap {constraint.id!r} slack."
            )
        indices = [
            index
            for index, ticker in enumerate(universe)
            if sectors.get(ticker) == constraint.group
        ]
        if not indices:
            raise DualsUnavailableError(
                f"GroupCap {constraint.id!r} group {constraint.group!r} matches no ticker."
            )
        primal = math.fsum(float(weights[index]) for index in indices)
        rows = [
            _Row(
                label=constraint.group,
                side="upper",
                bound=constraint.max_weight,
                bound_unit="portfolio_weight_fraction",
                primal=primal,
            )
        ]
        if constraint.min_weight is not None:
            rows.append(
                _Row(
                    label=constraint.group,
                    side="lower",
                    bound=constraint.min_weight,
                    bound_unit="portfolio_weight_fraction",
                    primal=primal,
                )
            )
        return rows

    if isinstance(constraint, FactorExposure):
        loadings = _aligned_vector(factors, constraint.factor, n, label="factor")
        primal = float(loadings @ weights)
        if not math.isfinite(primal):
            raise DualsUnavailableError(
                f"Computed factor exposure for {constraint.factor!r} is non-finite."
            )
        rows: list[_Row] = []
        if constraint.max_exposure is not None:
            rows.append(
                _Row(
                    label=constraint.factor,
                    side="upper",
                    bound=constraint.max_exposure,
                    bound_unit="factor_exposure",
                    primal=primal,
                )
            )
        if constraint.min_exposure is not None:
            rows.append(
                _Row(
                    label=constraint.factor,
                    side="lower",
                    bound=constraint.min_exposure,
                    bound_unit="factor_exposure",
                    primal=primal,
                )
            )
        return rows

    if isinstance(constraint, TurnoverCap):
        primal = math.fsum(float(value) for value in np.abs(weights - w_prev))
        return [
            _Row(
                label="portfolio_turnover",
                side="upper",
                bound=constraint.max_turnover,
                bound_unit="l1_weight_fraction",
                primal=primal,
            )
        ]

    if isinstance(constraint, CVaRLimit):
        scenario_matrix = _finite_matrix(
            scenarios,
            None,
            label=f"Scenario matrix for CVaRLimit {constraint.id!r}",
        )
        if scenario_matrix.shape[0] < 1 or scenario_matrix.shape[1] != n:
            raise DualsUnavailableError(
                f"Scenario matrix for CVaRLimit {constraint.id!r} must have shape (S, {n}) "
                f"with S >= 1; got {scenario_matrix.shape}."
            )
        losses = -(scenario_matrix @ weights)
        if not np.all(np.isfinite(losses)):
            raise DualsUnavailableError("Computed CVaR scenario losses are non-finite.")
        primal = _empirical_cvar(losses, constraint.alpha)
        return [
            _Row(
                label=f"cvar_{constraint.alpha:g}",
                side="upper",
                bound=constraint.max_cvar,
                bound_unit="scenario_loss_fraction",
                primal=primal,
            )
        ]

    if isinstance(constraint, TrackingErrorCap):
        benchmark = _aligned_vector(benchmarks, constraint.benchmark, n, label="benchmark")
        active = weights - benchmark
        variance = float(active @ sigma @ active)
        scale = max(1.0, float(np.max(np.abs(sigma))))
        if not math.isfinite(variance) or variance < -1e-12 * scale:
            raise DualsUnavailableError(
                f"Computed tracking-error variance for {constraint.benchmark!r} is invalid."
            )
        primal = math.sqrt(max(variance, 0.0))
        return [
            _Row(
                label=constraint.benchmark,
                side="upper",
                bound=constraint.max_te,
                bound_unit="annualized_volatility_fraction",
                primal=primal,
            )
        ]

    raise DualsUnavailableError(
        f"Constraint kind {getattr(constraint, 'kind', type(constraint).__name__)!r} "
        "does not have sensitivity row semantics."
    )


def _slack(row: _Row) -> float:
    if row.side == "lower":
        return row.primal - row.bound
    if row.side == "upper":
        return row.bound - row.primal
    return row.primal - row.bound


def _derivative(dual: float, row: _Row) -> float:
    if row.side == "equality":
        value = -dual
    elif row.side == "lower":
        value = dual * row.parameter_scale
    else:
        value = -dual * row.parameter_scale
    if not math.isfinite(value):
        raise DualsUnavailableError(
            "Sensitivity derivative overflowed while applying the parameter scale."
        )
    return value


def harvest_sensitivities(
    compiled: CompiledProblem,
    recovered_weights: np.ndarray,
    w_prev: np.ndarray,
    sigma: np.ndarray,
    scenarios: np.ndarray | None = None,
    sectors: Mapping[str, str] | None = None,
    benchmarks: Mapping[str, np.ndarray] | None = None,
    factors: Mapping[str, np.ndarray] | None = None,
    *,
    conditional: bool = False,
) -> tuple[SensitivityRecord, ...]:
    """Harvest every interpretable named solver row without scalar collapse.

    ``benchmarks`` and ``factors`` must already be aligned to ``spec.universe``.
    Cardinality has no MIP dual and TransactionCost is an objective penalty;
    both are represented by :func:`sensitivity_coverage`, not fake records.
    """

    _check_solved(compiled)
    spec = compiled.spec
    if spec is None:
        raise DualsUnavailableError("Compiled problem does not retain its originating spec.")
    n = len(spec.universe)
    weights = _finite_vector(recovered_weights, n, label="Recovered weights")
    previous = _finite_vector(w_prev, n, label="Pre-trade weights")
    covariance = _finite_matrix(sigma, (n, n), label="Covariance matrix")
    max_sharpe_scale = _max_sharpe_scale(compiled) if spec.objective.kind == "max_sharpe" else 1.0
    objective_unit = _objective_unit(spec)
    unavailable_ids = set(sensitivity_dependency_reasons(compiled))

    records: list[SensitivityRecord] = []
    for constraint in spec.constraints:
        if isinstance(constraint, TransactionCost | Cardinality):
            continue
        if constraint.id in unavailable_ids:
            # Suppress the entire IR constraint when any active scalar row has
            # a non-authoritative multiplier. Coverage is constraint-level, so
            # retaining its inactive sibling rows would falsely imply complete
            # derivative coverage.
            continue
        solver_constraint = compiled.constraint_objs.get(constraint.id)
        if solver_constraint is None:
            # These rows are intentionally implicit in transformed objectives.
            if spec.objective.kind == "max_sharpe" and isinstance(constraint, Budget):
                continue
            if spec.objective.kind == "risk_parity" and isinstance(constraint, Budget | LongOnly):
                continue
            raise DualsUnavailableError(
                f"Constraint {constraint.id!r} is absent from the compiler's named-row map."
            )

        rows = _rows_for_constraint(
            constraint,
            spec=spec,
            weights=weights,
            w_prev=previous,
            sigma=covariance,
            scenarios=scenarios,
            sectors=sectors,
            benchmarks=benchmarks,
            factors=factors,
            max_sharpe_scale=max_sharpe_scale,
        )
        duals = _dual_vector(
            solver_constraint.dual_value,
            len(rows),
            constraint_id=constraint.id,
        )
        for row_index, (row, raw_dual) in enumerate(zip(rows, duals, strict=True)):
            slack = _slack(row)
            if not math.isfinite(slack):
                raise DualsUnavailableError(
                    f"Constraint {constraint.id!r} row {row_index} has non-finite slack."
                )
            tolerance = _BINDING_REL_TOL * max(1.0, abs(float(row.bound)), abs(float(row.primal)))
            records.append(
                SensitivityRecord(
                    constraint_id=constraint.id,
                    kind=constraint.kind,
                    row_index=row_index,
                    row_label=row.label,
                    side=row.side,
                    bound_value=float(row.bound),
                    bound_unit=row.bound_unit,
                    raw_solver_dual=float(raw_dual),
                    parameter_scale=float(row.parameter_scale),
                    objective_derivative_per_bound_unit=_derivative(float(raw_dual), row),
                    objective_unit=objective_unit,
                    primal_value=float(row.primal),
                    slack=float(slack),
                    is_binding=abs(slack) <= tolerance,
                    conditional=bool(conditional),
                )
            )
    return tuple(records)


def sensitivity_coverage(
    spec: PortfolioSpec,
    records: Sequence[SensitivityRecord],
    conditional: bool,
) -> dict[str, SensitivityCoverage]:
    """Describe sensitivity availability for every explicit IR constraint."""

    ids_with_records = {record.constraint_id for record in records}
    availability: SensitivityAvailability = "conditional" if conditional else "available"
    coverage: dict[str, SensitivityCoverage] = {}
    for constraint in spec.constraints:
        if isinstance(constraint, TransactionCost):
            coverage[constraint.id] = SensitivityCoverage(
                availability="unavailable",
                reason="TransactionCost is an objective penalty, not a hard constraint row.",
            )
        elif isinstance(constraint, Cardinality):
            coverage[constraint.id] = SensitivityCoverage(
                availability="unavailable",
                reason=(
                    "Cardinality is mixed-integer and has no meaningful native dual; "
                    "other rows require a validated fixed-selection continuous solve."
                ),
            )
        elif spec.objective.kind == "max_sharpe" and isinstance(constraint, Budget):
            coverage[constraint.id] = SensitivityCoverage(
                availability="unavailable",
                reason=(
                    "The unit budget is implicit in Max-Sharpe weight recovery and has no "
                    "named transformed solver row."
                ),
            )
        elif spec.objective.kind == "risk_parity" and isinstance(constraint, Budget | LongOnly):
            coverage[constraint.id] = SensitivityCoverage(
                availability="unavailable",
                reason=(
                    "Risk-parity budget and positivity semantics are implicit in normalization "
                    "and the log-domain surrogate, not named solver rows."
                ),
            )
        elif constraint.id in ids_with_records:
            reason = (
                "Conditional on the mixed-integer solver's fixed selected name set."
                if conditional
                else None
            )
            coverage[constraint.id] = SensitivityCoverage(
                availability=availability,
                reason=reason,
            )
        else:
            coverage[constraint.id] = SensitivityCoverage(
                availability="unavailable",
                reason="No validated solver-row sensitivity record was produced.",
            )
    has_budget = any(isinstance(constraint, Budget) for constraint in spec.constraints)
    has_long_only = any(isinstance(constraint, LongOnly) for constraint in spec.constraints)
    if spec.objective.kind in {"max_sharpe", "risk_parity"} and not has_budget:
        coverage["__implicit_unit_budget__"] = SensitivityCoverage(
            availability="unavailable",
            reason=(
                "The unit budget is implicit in transformed-weight normalization and "
                "has no named solver row."
            ),
        )
    if spec.objective.kind == "risk_parity" and not has_long_only:
        coverage["__implicit_positive_domain__"] = SensitivityCoverage(
            availability="unavailable",
            reason=(
                "Strict positivity is implicit in the risk-parity log domain and has "
                "no named solver row."
            ),
        )
    return coverage
