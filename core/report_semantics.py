"""Typed, unit-aware objective and portfolio reporting semantics.

This module is deliberately pure: it reads a solved :class:`CompiledProblem`
and the aligned numeric inputs used to build it, but never mutates solver or IR
state.  It keeps two ideas separate:

* objective terms reconstruct the scalar score the solver minimized; and
* portfolio metrics describe financial quantities in their natural units.

That distinction matters whenever unlike financial quantities are combined in
one optimization score (for example annualized variance plus a proportional
transaction-cost penalty), and for transformed objectives whose raw solver
variable is not the recovered portfolio weight vector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np

from core.compile_context import CompiledProblem
from core.constraints.transaction_cost import TransactionCost
from core.ir import (
    MaxSharpe,
    MeanVariance,
    MinCVaR,
    MinTrackingError,
    MinVariance,
    PortfolioSpec,
    RiskParity,
)

ObjectiveRole = Literal["base", "reward", "penalty", "transform"]


@dataclass(frozen=True, slots=True)
class ObjectiveTerm:
    """One auditable contribution to the solver's scalar objective score."""

    key: str
    label: str
    role: ObjectiveRole
    natural_value: float
    natural_unit: str
    coefficient: float
    objective_contribution: float
    source_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectiveDecomposition:
    """Reconstruction of the minimized solver score from typed terms."""

    sense: Literal["minimize"] = "minimize"
    solver_value: float
    solver_unit: str = "objective_score"
    terms: tuple[ObjectiveTerm, ...]
    reconstruction_error: float
    reconstruction_error_unit: str = "objective_score"


@dataclass(frozen=True, slots=True)
class PortfolioMetric:
    """A portfolio statistic with an explicit definition, unit, and horizon."""

    key: str
    label: str
    value: float
    unit: str
    definition: str
    annualization_periods: int | None = None
    confidence_level: float | None = None
    benchmark: str | None = None


ReportSemantics = tuple[ObjectiveDecomposition, tuple[PortfolioMetric, ...]]


def _finite_scalar(value: object, *, label: str) -> float:
    """Return ``value`` as a finite scalar or fail closed."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite real scalar.") from exc
    if array.shape != () or np.iscomplexobj(array):
        raise ValueError(f"{label} must be a finite real scalar.")
    try:
        result = float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite real scalar.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _finite_vector(value: object, n: int, *, label: str) -> np.ndarray:
    """Return a finite real vector of length ``n``."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite real vector of shape ({n},).") from exc
    if np.iscomplexobj(raw):
        raise ValueError(f"{label} must be a finite real vector of shape ({n},).")
    try:
        vector = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite real vector of shape ({n},).") from exc
    if vector.shape != (n,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be a finite real vector of shape ({n},).")
    return vector


def _finite_matrix(value: object, shape: tuple[int, int], *, label: str) -> np.ndarray:
    """Return a finite real matrix with exactly ``shape``."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite real matrix of shape {shape}.") from exc
    if np.iscomplexobj(raw):
        raise ValueError(f"{label} must be a finite real matrix of shape {shape}.")
    try:
        matrix = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite real matrix of shape {shape}.") from exc
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite real matrix of shape {shape}.")
    return matrix


def _quadratic(vector: np.ndarray, matrix: np.ndarray, *, label: str) -> float:
    """Evaluate a portfolio quadratic and reject overflow or material negativity."""

    with np.errstate(over="ignore", invalid="ignore"):
        value = _finite_scalar(vector @ matrix @ vector, label=label)
    # Inputs reaching this layer have already passed the compiler's PSD check.
    # Permit only eigensolver/solver-scale negative roundoff before sqrt.
    scale = max(1.0, float(np.max(np.abs(matrix))) * float(vector @ vector))
    tolerance = 1e-12 * scale
    if value < -tolerance:
        raise ValueError(f"{label} is negative ({value:g}) despite a PSD covariance matrix.")
    return max(value, 0.0)


def _scenario_matrix(value: object, n: int) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Scenario matrix must be finite with shape (S, n) and S >= 1.") from exc
    if np.iscomplexobj(raw):
        raise ValueError("Scenario matrix must be finite with shape (S, n) and S >= 1.")
    try:
        scenarios = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Scenario matrix must be finite with shape (S, n) and S >= 1.") from exc
    if (
        scenarios.ndim != 2
        or scenarios.shape[0] < 1
        or scenarios.shape[1] != n
        or not np.all(np.isfinite(scenarios))
    ):
        raise ValueError("Scenario matrix must be finite with shape (S, n) and S >= 1.")
    return scenarios


def _metric(
    key: str,
    label: str,
    value: float,
    unit: str,
    definition: str,
    *,
    annualization_periods: int | None = None,
    confidence_level: float | None = None,
    benchmark: str | None = None,
) -> PortfolioMetric:
    return PortfolioMetric(
        key=key,
        label=label,
        value=_finite_scalar(value, label=f"Metric {key!r}"),
        unit=unit,
        definition=definition,
        annualization_periods=annualization_periods,
        confidence_level=confidence_level,
        benchmark=benchmark,
    )


def build_report_semantics(
    spec: PortfolioSpec,
    compiled: CompiledProblem,
    weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    scenarios: np.ndarray | None = None,
    w_prev: np.ndarray | None = None,
    benchmark_weights: dict[str, np.ndarray] | None = None,
    periods_per_year: int = 252,
) -> ReportSemantics:
    """Build a typed objective decomposition and portfolio metrics.

    ``mu`` and ``sigma`` are the annualized estimates supplied to the compiler;
    scenarios retain their source-period horizon.  ``weights`` must be the
    recovered portfolio weights, while transformed objective terms are read
    from ``compiled.weights.value`` so their raw solver semantics are preserved.
    """

    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, Integral)
        or periods_per_year < 1
    ):
        raise ValueError("periods_per_year must be a positive integer.")
    annualization = int(periods_per_year)
    n = len(spec.universe)
    portfolio = _finite_vector(weights, n, label="Recovered portfolio weights")
    expected_returns = _finite_vector(mu, n, label="Expected-return vector")
    covariance = _finite_matrix(sigma, (n, n), label="Covariance matrix")
    solver_value = _finite_scalar(compiled.problem.value, label="Solver objective value")

    expected_return = _finite_scalar(
        expected_returns @ portfolio,
        label="Portfolio expected return",
    )
    variance = _quadratic(portfolio, covariance, label="Portfolio variance")
    volatility = math.sqrt(variance)

    metrics: list[PortfolioMetric] = [
        _metric(
            "expected_return",
            "Expected return",
            expected_return,
            "fraction_per_year",
            "mu @ weights using the annualized mean log-return estimate.",
            annualization_periods=annualization,
        ),
        _metric(
            "variance",
            "Portfolio variance",
            variance,
            "fraction_squared_per_year",
            "weights.T @ annualized_covariance @ weights.",
            annualization_periods=annualization,
        ),
        _metric(
            "volatility",
            "Portfolio volatility",
            volatility,
            "fraction_per_sqrt_year",
            "Square root of annualized portfolio variance.",
            annualization_periods=annualization,
        ),
    ]
    terms: list[ObjectiveTerm] = []

    objective = spec.objective
    if isinstance(objective, MinVariance):
        terms.append(
            ObjectiveTerm(
                key="variance",
                label="Annualized portfolio variance",
                role="base",
                natural_value=variance,
                natural_unit="fraction_squared_per_year",
                coefficient=1.0,
                objective_contribution=variance,
            )
        )
    elif isinstance(objective, MeanVariance):
        reward = -objective.risk_aversion * expected_return
        terms.extend(
            [
                ObjectiveTerm(
                    key="variance",
                    label="Annualized portfolio variance",
                    role="base",
                    natural_value=variance,
                    natural_unit="fraction_squared_per_year",
                    coefficient=1.0,
                    objective_contribution=variance,
                ),
                ObjectiveTerm(
                    key="expected_return_reward",
                    label="Expected-return reward",
                    role="reward",
                    natural_value=expected_return,
                    natural_unit="fraction_per_year",
                    coefficient=-float(objective.risk_aversion),
                    objective_contribution=_finite_scalar(
                        reward,
                        label="Mean-variance expected-return contribution",
                    ),
                ),
            ]
        )
        metrics.append(
            _metric(
                "mean_variance_score",
                "Mean-variance score",
                variance + reward,
                "objective_score",
                "Annualized variance minus risk_aversion times annualized expected return; "
                "excludes transaction-cost penalties.",
                annualization_periods=annualization,
            )
        )
    elif isinstance(objective, MinCVaR):
        if scenarios is None:
            raise ValueError("MinCVaR report semantics require the scenario matrix used to solve.")
        scenario_returns = _scenario_matrix(scenarios, n)
        t_var = compiled.extra_vars.get("t")
        if t_var is None or t_var.value is None:
            raise ValueError("MinCVaR solve did not populate its VaR variable t.")
        value_at_risk = _finite_scalar(t_var.value, label="MinCVaR VaR variable t")
        with np.errstate(over="ignore", invalid="ignore"):
            losses = -(scenario_returns @ portfolio)
            excess = np.maximum(losses - value_at_risk, 0.0)
        if not np.all(np.isfinite(losses)) or not np.all(np.isfinite(excess)):
            raise ValueError("MinCVaR loss or tail-excess calculation was non-finite.")
        mean_tail_excess = _finite_scalar(
            math.fsum(float(value) for value in excess) / len(excess),
            label="Mean CVaR tail excess",
        )
        tail_coefficient = 1.0 / (1.0 - objective.cvar_alpha)
        tail_contribution = _finite_scalar(
            tail_coefficient * mean_tail_excess,
            label="CVaR tail contribution",
        )
        cvar = _finite_scalar(value_at_risk + tail_contribution, label="CVaR")
        terms.extend(
            [
                ObjectiveTerm(
                    key="cvar_var_threshold",
                    label="Rockafellar-Uryasev VaR threshold",
                    role="base",
                    natural_value=value_at_risk,
                    natural_unit="fraction_per_scenario_period",
                    coefficient=1.0,
                    objective_contribution=value_at_risk,
                ),
                ObjectiveTerm(
                    key="cvar_tail_excess",
                    label="Mean loss above the VaR threshold",
                    role="base",
                    natural_value=mean_tail_excess,
                    natural_unit="fraction_per_scenario_period",
                    coefficient=tail_coefficient,
                    objective_contribution=tail_contribution,
                ),
            ]
        )
        metrics.extend(
            [
                _metric(
                    "value_at_risk",
                    "Value at risk",
                    value_at_risk,
                    "fraction_per_scenario_period",
                    "The optimized Rockafellar-Uryasev loss threshold t.",
                    confidence_level=float(objective.cvar_alpha),
                ),
                _metric(
                    "conditional_value_at_risk",
                    "Conditional value at risk",
                    cvar,
                    "fraction_per_scenario_period",
                    "t + mean(max(loss - t, 0)) / (1 - confidence_level).",
                    confidence_level=float(objective.cvar_alpha),
                ),
            ]
        )
    elif isinstance(objective, MaxSharpe):
        raw = _finite_vector(compiled.weights.value, n, label="MaxSharpe transformed weights")
        transformed_variance = _quadratic(
            raw,
            covariance,
            label="MaxSharpe transformed inverse-Sharpe-squared term",
        )
        terms.append(
            ObjectiveTerm(
                key="inverse_sharpe_squared_transform",
                label="Charnes-Cooper inverse-Sharpe-squared term",
                role="transform",
                natural_value=transformed_variance,
                natural_unit="inverse_sharpe_squared",
                coefficient=1.0,
                objective_contribution=transformed_variance,
            )
        )
        excess_return = _finite_scalar(
            expected_return - objective.risk_free_rate,
            label="Portfolio excess expected return",
        )
        if volatility <= 0.0:
            raise ValueError("Sharpe ratio is undefined because portfolio volatility is zero.")
        sharpe = _finite_scalar(excess_return / volatility, label="Sharpe ratio")
        metrics.extend(
            [
                _metric(
                    "excess_expected_return",
                    "Excess expected return",
                    excess_return,
                    "fraction_per_year",
                    "Annualized expected return minus the objective's annualized risk-free rate.",
                    annualization_periods=annualization,
                ),
                _metric(
                    "sharpe_ratio",
                    "Sharpe ratio",
                    sharpe,
                    "dimensionless",
                    "Annualized excess expected return divided by annualized volatility.",
                    annualization_periods=annualization,
                ),
            ]
        )
    elif isinstance(objective, RiskParity):
        raw = _finite_vector(compiled.weights.value, n, label="RiskParity transformed weights")
        if np.any(raw <= 0.0):
            raise ValueError("RiskParity transformed weights must be strictly positive for log.")
        raw_quadratic = _quadratic(
            raw,
            covariance,
            label="RiskParity transformed quadratic",
        )
        log_sum = _finite_scalar(
            math.fsum(math.log(float(value)) for value in raw),
            label="RiskParity log-barrier sum",
        )
        risk_contribution = 0.5 * raw_quadratic
        log_contribution = -(1.0 / n) * log_sum
        terms.extend(
            [
                ObjectiveTerm(
                    key="risk_parity_quadratic_transform",
                    label="RiskParity transformed quadratic risk",
                    role="transform",
                    natural_value=raw_quadratic,
                    natural_unit="transformed_fraction_squared_per_year",
                    coefficient=0.5,
                    objective_contribution=risk_contribution,
                ),
                ObjectiveTerm(
                    key="risk_parity_log_barrier",
                    label="RiskParity log-barrier",
                    role="transform",
                    natural_value=log_sum,
                    natural_unit="log_transformed_weight",
                    coefficient=-1.0 / n,
                    objective_contribution=log_contribution,
                ),
            ]
        )
        if variance <= 0.0:
            raise ValueError(
                "Risk-contribution shares are undefined because portfolio variance is zero."
            )
        component_risk = portfolio * (covariance @ portfolio)
        shares = component_risk / variance
        if not np.all(np.isfinite(shares)):
            raise ValueError("Risk-contribution share calculation was non-finite.")
        max_share_deviation = _finite_scalar(
            np.max(np.abs(shares - 1.0 / n)),
            label="Maximum risk-contribution share deviation",
        )
        metrics.append(
            _metric(
                "risk_contribution_max_share_deviation",
                "Maximum risk-contribution share deviation",
                max_share_deviation,
                "fraction_of_total_variance",
                "max_i(abs((w_i * (covariance @ w)_i) / portfolio_variance - 1 / n)).",
                annualization_periods=annualization,
            )
        )
    elif isinstance(objective, MinTrackingError):
        if benchmark_weights is None or objective.benchmark not in benchmark_weights:
            raise ValueError(
                f"MinTrackingError report semantics require aligned benchmark "
                f"{objective.benchmark!r}."
            )
        benchmark = _finite_vector(
            benchmark_weights[objective.benchmark],
            n,
            label=f"Benchmark {objective.benchmark!r}",
        )
        active = portfolio - benchmark
        tracking_variance = _quadratic(
            active,
            covariance,
            label="Tracking-error variance",
        )
        tracking_error = math.sqrt(tracking_variance)
        terms.append(
            ObjectiveTerm(
                key="tracking_error_variance",
                label=f"Tracking-error variance vs {objective.benchmark}",
                role="base",
                natural_value=tracking_variance,
                natural_unit="fraction_squared_per_year",
                coefficient=1.0,
                objective_contribution=tracking_variance,
            )
        )
        metrics.extend(
            [
                _metric(
                    "tracking_error_variance",
                    "Tracking-error variance",
                    tracking_variance,
                    "fraction_squared_per_year",
                    "(weights - benchmark).T @ annualized_covariance @ "
                    "(weights - benchmark).",
                    annualization_periods=annualization,
                    benchmark=objective.benchmark,
                ),
                _metric(
                    "tracking_error",
                    "Tracking error",
                    tracking_error,
                    "fraction_per_sqrt_year",
                    "Square root of annualized tracking-error variance.",
                    annualization_periods=annualization,
                    benchmark=objective.benchmark,
                ),
            ]
        )
    else:  # pragma: no cover - the discriminated IR union makes this defensive.
        raise TypeError(f"Unsupported objective type: {type(objective).__name__}.")

    transaction_costs = [
        constraint for constraint in spec.constraints if isinstance(constraint, TransactionCost)
    ]
    if transaction_costs:
        previous = (
            np.zeros(n, dtype=float)
            if w_prev is None
            else _finite_vector(w_prev, n, label="Pre-trade weight vector")
        )
        with np.errstate(over="ignore", invalid="ignore"):
            deviations = np.abs(portfolio - previous)
        if not np.all(np.isfinite(deviations)):
            raise ValueError("L1 turnover calculation was non-finite.")
        turnover = _finite_scalar(
            math.fsum(float(value) for value in deviations),
            label="L1 turnover",
        )
        modeled_costs: list[float] = []
        for constraint in transaction_costs:
            coefficient = float(constraint.bps) / 1e4
            contribution = _finite_scalar(
                coefficient * turnover,
                label=f"Transaction-cost contribution {constraint.id!r}",
            )
            modeled_costs.append(contribution)
            terms.append(
                ObjectiveTerm(
                    key=f"transaction_cost:{constraint.id}",
                    label="Transaction-cost penalty",
                    role="penalty",
                    natural_value=turnover,
                    natural_unit="portfolio_weight_fraction",
                    coefficient=coefficient,
                    objective_contribution=contribution,
                    source_id=constraint.id,
                )
            )
        aggregate_cost = _finite_scalar(
            math.fsum(modeled_costs),
            label="Aggregate modeled transaction cost",
        )
        metrics.extend(
            [
                _metric(
                    "l1_turnover",
                    "L1 turnover",
                    turnover,
                    "portfolio_weight_fraction",
                    "sum(abs(target_weight - pre_trade_weight)) across all assets.",
                ),
                _metric(
                    "modeled_transaction_cost",
                    "Modeled transaction cost",
                    aggregate_cost,
                    "portfolio_value_fraction",
                    "Sum of each configured proportional transaction-cost rate "
                    "times L1 turnover.",
                ),
            ]
        )

    try:
        reconstructed = math.fsum(term.objective_contribution for term in terms)
    except OverflowError as exc:
        raise ValueError("Objective-term reconstruction overflowed.") from exc
    reconstruction_error = _finite_scalar(
        solver_value - reconstructed,
        label="Objective reconstruction error",
    )
    decomposition = ObjectiveDecomposition(
        solver_value=solver_value,
        terms=tuple(terms),
        reconstruction_error=reconstruction_error,
    )
    return decomposition, tuple(metrics)


__all__ = [
    "ObjectiveDecomposition",
    "ObjectiveTerm",
    "PortfolioMetric",
    "ReportSemantics",
    "build_report_semantics",
]
