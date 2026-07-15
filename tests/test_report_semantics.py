"""Hand-check the typed objective and portfolio reporting semantics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from core.compile_context import CompiledProblem
from core.ir import (
    Budget,
    LongOnly,
    MaxSharpe,
    MeanVariance,
    MinCVaR,
    MinTrackingError,
    MinVariance,
    Objective,
    PortfolioSpec,
    RiskParity,
    TransactionCost,
)
from core.report_semantics import ObjectiveTerm, build_report_semantics
from core.solve import solve_spec

_UNIVERSE = ["A", "B"]
_MU = np.array([0.12, 0.08])
_SIGMA = np.diag([0.04, 0.09]).astype(float)


def _compiled(
    solver_value: float,
    *,
    raw_weights: np.ndarray | None = None,
    extra_vars: dict[str, object] | None = None,
) -> CompiledProblem:
    """Small solved-problem stand-in; semantics are intentionally solver-free."""

    return cast(
        CompiledProblem,
        SimpleNamespace(
            problem=SimpleNamespace(value=solver_value),
            weights=SimpleNamespace(value=raw_weights),
            extra_vars=extra_vars or {},
        ),
    )


def _term_map(decomposition: object) -> dict[str, ObjectiveTerm]:
    return {term.key: term for term in decomposition.terms}  # type: ignore[attr-defined]


def _metric_map(metrics: tuple[object, ...]) -> dict[str, object]:
    return {metric.key: metric for metric in metrics}  # type: ignore[attr-defined]


def test_min_variance_and_transaction_cost_terms_are_hand_checkable() -> None:
    weights = np.array([0.25, 0.75])
    previous = np.array([0.50, 0.50])
    variance = float(weights @ _SIGMA @ weights)
    turnover = 0.50
    # Two configured penalty nodes really are additive in the compiler.
    modeled_cost = turnover * (10.0 + 5.0) / 1e4
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MinVariance(),
        constraints=[
            TransactionCost(id="cost_a", bps=10.0),
            TransactionCost(id="cost_b", bps=5.0),
        ],
    )

    decomposition, metrics = build_report_semantics(
        spec,
        _compiled(variance + modeled_cost),
        weights,
        _MU,
        _SIGMA,
        w_prev=previous,
        periods_per_year=12,
    )

    terms = _term_map(decomposition)
    assert terms["variance"].natural_unit == "fraction_squared_per_year"
    assert terms["variance"].objective_contribution == pytest.approx(variance)
    assert terms["transaction_cost:cost_a"].source_id == "cost_a"
    assert terms["transaction_cost:cost_a"].coefficient == pytest.approx(0.001)
    assert terms["transaction_cost:cost_b"].objective_contribution == pytest.approx(0.00025)
    assert decomposition.reconstruction_error == pytest.approx(0.0)

    by_key = _metric_map(metrics)
    assert by_key["expected_return"].value == pytest.approx(float(_MU @ weights))
    assert by_key["variance"].value == pytest.approx(variance)
    assert by_key["volatility"].value == pytest.approx(np.sqrt(variance))
    assert by_key["expected_return"].annualization_periods == 12
    assert by_key["l1_turnover"].value == pytest.approx(turnover)
    assert by_key["modeled_transaction_cost"].value == pytest.approx(modeled_cost)


def test_mean_variance_separates_variance_from_expected_return_reward() -> None:
    weights = np.array([0.60, 0.40])
    risk_aversion = 2.5
    variance = float(weights @ _SIGMA @ weights)
    expected_return = float(_MU @ weights)
    score = variance - risk_aversion * expected_return
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MeanVariance(risk_aversion=risk_aversion),
    )

    decomposition, metrics = build_report_semantics(
        spec,
        _compiled(score),
        weights,
        _MU,
        _SIGMA,
    )

    terms = _term_map(decomposition)
    reward = terms["expected_return_reward"]
    assert reward.role == "reward"
    assert reward.natural_value == pytest.approx(expected_return)
    assert reward.coefficient == pytest.approx(-risk_aversion)
    assert reward.objective_contribution == pytest.approx(-risk_aversion * expected_return)
    assert decomposition.reconstruction_error == pytest.approx(0.0)
    assert _metric_map(metrics)["mean_variance_score"].value == pytest.approx(score)


def test_min_cvar_reports_ru_var_tail_and_scenario_period_cvar() -> None:
    weights = np.array([1.0, 0.0])
    scenarios = np.array([[0.10, 0.0], [-0.20, 0.0], [0.00, 0.0], [-0.10, 0.0]])
    alpha = 0.50
    value_at_risk = 0.05
    # Losses [-.10, .20, 0, .10] produce mean positive excess .05.
    mean_tail_excess = 0.05
    cvar = value_at_risk + mean_tail_excess / (1.0 - alpha)
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MinCVaR(cvar_alpha=alpha),
    )
    t_var = SimpleNamespace(value=value_at_risk)

    decomposition, metrics = build_report_semantics(
        spec,
        _compiled(cvar, extra_vars={"t": t_var}),
        weights,
        _MU,
        _SIGMA,
        scenarios=scenarios,
    )

    terms = _term_map(decomposition)
    assert terms["cvar_var_threshold"].natural_value == pytest.approx(value_at_risk)
    assert terms["cvar_tail_excess"].natural_value == pytest.approx(mean_tail_excess)
    assert terms["cvar_tail_excess"].coefficient == pytest.approx(2.0)
    assert decomposition.reconstruction_error == pytest.approx(0.0)
    by_key = _metric_map(metrics)
    assert by_key["value_at_risk"].value == pytest.approx(value_at_risk)
    assert by_key["conditional_value_at_risk"].value == pytest.approx(cvar)
    assert by_key["conditional_value_at_risk"].confidence_level == alpha
    assert by_key["conditional_value_at_risk"].unit == "fraction_per_scenario_period"


def test_max_sharpe_uses_raw_transform_and_reports_recovered_sharpe() -> None:
    weights = np.array([0.60, 0.40])
    risk_free_rate = 0.02
    expected_return = float(_MU @ weights)
    excess_return = expected_return - risk_free_rate
    variance = float(weights @ _SIGMA @ weights)
    raw_y = weights / excess_return
    transformed_variance = float(raw_y @ _SIGMA @ raw_y)
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MaxSharpe(risk_free_rate=risk_free_rate),
    )

    decomposition, metrics = build_report_semantics(
        spec,
        _compiled(transformed_variance, raw_weights=raw_y),
        weights,
        _MU,
        _SIGMA,
    )

    term = _term_map(decomposition)["inverse_sharpe_squared_transform"]
    assert term.role == "transform"
    assert term.natural_value == pytest.approx(transformed_variance)
    assert term.natural_value == pytest.approx(variance / excess_return**2)
    assert decomposition.reconstruction_error == pytest.approx(0.0)
    by_key = _metric_map(metrics)
    assert by_key["excess_expected_return"].value == pytest.approx(excess_return)
    assert by_key["sharpe_ratio"].value == pytest.approx(excess_return / np.sqrt(variance))


def test_max_sharpe_fails_closed_when_volatility_is_zero() -> None:
    spec = PortfolioSpec(universe=_UNIVERSE, objective=MaxSharpe())
    with pytest.raises(ValueError, match="Sharpe ratio is undefined"):
        build_report_semantics(
            spec,
            _compiled(0.0, raw_weights=np.ones(2)),
            np.array([0.5, 0.5]),
            _MU,
            np.zeros((2, 2)),
        )


def test_risk_parity_reconstructs_surrogate_and_measures_share_deviation() -> None:
    raw = np.array([1.0, 2.0])
    weights = raw / raw.sum()
    raw_quadratic = float(raw @ _SIGMA @ raw)
    log_sum = float(np.log(raw).sum())
    solver_value = 0.5 * raw_quadratic - 0.5 * log_sum
    spec = PortfolioSpec(universe=_UNIVERSE, objective=RiskParity())

    decomposition, metrics = build_report_semantics(
        spec,
        _compiled(solver_value, raw_weights=raw),
        weights,
        _MU,
        _SIGMA,
    )

    terms = _term_map(decomposition)
    assert terms["risk_parity_quadratic_transform"].coefficient == pytest.approx(0.5)
    assert terms["risk_parity_quadratic_transform"].natural_value == pytest.approx(raw_quadratic)
    assert terms["risk_parity_log_barrier"].coefficient == pytest.approx(-0.5)
    assert terms["risk_parity_log_barrier"].natural_value == pytest.approx(log_sum)
    assert decomposition.reconstruction_error == pytest.approx(0.0)
    # Component-risk shares are [0.1, 0.9], versus the equal target [0.5, 0.5].
    deviation = _metric_map(metrics)["risk_contribution_max_share_deviation"]
    assert deviation.value == pytest.approx(0.4)
    assert deviation.unit == "fraction_of_total_variance"


def test_min_tracking_error_reports_variance_and_root_against_named_benchmark() -> None:
    weights = np.array([0.60, 0.40])
    benchmark = np.array([0.50, 0.50])
    active = weights - benchmark
    tracking_variance = float(active @ _SIGMA @ active)
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MinTrackingError(benchmark="bench"),
    )

    decomposition, metrics = build_report_semantics(
        spec,
        _compiled(tracking_variance),
        weights,
        _MU,
        _SIGMA,
        benchmark_weights={"bench": benchmark},
    )

    term = _term_map(decomposition)["tracking_error_variance"]
    assert term.objective_contribution == pytest.approx(tracking_variance)
    assert decomposition.reconstruction_error == pytest.approx(0.0)
    by_key = _metric_map(metrics)
    assert by_key["tracking_error_variance"].value == pytest.approx(tracking_variance)
    assert by_key["tracking_error"].value == pytest.approx(np.sqrt(tracking_variance))
    assert by_key["tracking_error"].benchmark == "bench"


def test_reconstruction_error_is_solver_value_minus_typed_contributions() -> None:
    weights = np.array([0.25, 0.75])
    variance = float(weights @ _SIGMA @ weights)
    decomposition, _ = build_report_semantics(
        PortfolioSpec(universe=_UNIVERSE, objective=MinVariance()),
        _compiled(variance + 0.125),
        weights,
        _MU,
        _SIGMA,
    )
    assert decomposition.reconstruction_error == pytest.approx(0.125)


def test_semantic_records_are_frozen_and_nonfinite_solver_values_fail_closed() -> None:
    term = ObjectiveTerm(
        key="x",
        label="X",
        role="base",
        natural_value=1.0,
        natural_unit="unit",
        coefficient=1.0,
        objective_contribution=1.0,
    )
    with pytest.raises(FrozenInstanceError):
        term.label = "changed"  # type: ignore[misc]

    with pytest.raises(ValueError, match="Solver objective value must be finite"):
        build_report_semantics(
            PortfolioSpec(universe=_UNIVERSE, objective=MinVariance()),
            _compiled(float("nan")),
            np.array([0.5, 0.5]),
            _MU,
            _SIGMA,
        )


@pytest.mark.parametrize(
    ("objective", "metric_key"),
    [
        (MinVariance(), "variance"),
        (MeanVariance(risk_aversion=1.5), "mean_variance_score"),
        (MinCVaR(cvar_alpha=0.90), "conditional_value_at_risk"),
        (MaxSharpe(risk_free_rate=0.0), "sharpe_ratio"),
        (RiskParity(), "risk_contribution_max_share_deviation"),
        (MinTrackingError(benchmark="bench"), "tracking_error"),
    ],
)
def test_solve_spec_wires_typed_semantics_for_every_objective(
    objective: Objective,
    metric_key: str,
) -> None:
    prices = pd.read_csv(
        Path(__file__).parent.parent / "examples" / "prices_sample.csv",
        parse_dates=[0],
        index_col=0,
    )
    universe = list(prices.columns)
    spec = PortfolioSpec(
        universe=universe,
        objective=objective,
        constraints=[Budget(), LongOnly()],
    )
    benchmarks = (
        {"bench": {ticker: 1.0 / len(universe) for ticker in universe}}
        if isinstance(objective, MinTrackingError)
        else None
    )

    _, report = solve_spec(spec, prices, benchmarks=benchmarks)

    assert report.objective_decomposition is not None
    assert report.objective_decomposition.solver_unit == "objective_score"
    assert report.metric(metric_key) is not None
    assert report.objective_decomposition.reconstruction_error == pytest.approx(0.0, abs=1e-6)


def test_solve_report_does_not_call_penalty_bearing_score_variance() -> None:
    prices = pd.read_csv(
        Path(__file__).parent.parent / "examples" / "prices_sample.csv",
        parse_dates=[0],
        index_col=0,
    )
    universe = list(prices.columns)
    spec = PortfolioSpec(
        universe=universe,
        objective=MinVariance(),
        constraints=[
            Budget(),
            LongOnly(),
            TransactionCost(id="cost", bps=10.0),
        ],
    )

    _, report = solve_spec(spec, prices)

    assert report.objective_decomposition is not None
    terms = _term_map(report.objective_decomposition)
    variance = report.metric("variance")
    cost = report.metric("modeled_transaction_cost")
    assert variance is not None and cost is not None
    assert terms["transaction_cost:cost"].source_id == "cost"
    assert report.objective_value == pytest.approx(variance.value + cost.value, abs=1e-6)
    assert report.objective_value != pytest.approx(variance.value, abs=1e-6)
