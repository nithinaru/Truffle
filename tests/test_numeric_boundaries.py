"""Sprint 1 regression tests for Truffle's numerical trust boundary."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import cvxpy as cp
import numpy as np
import pytest
from pydantic import ValidationError

from core.compile_context import CompiledProblem, validate_inputs
from core.compiler import compile_spec
from core.exceptions import CompilationError
from core.ir import (
    Box,
    Budget,
    Cardinality,
    CVaRLimit,
    FactorExposure,
    GroupCap,
    LongOnly,
    MaxSharpe,
    MeanVariance,
    MinCVaR,
    MinTrackingError,
    MinVariance,
    PortfolioSpec,
    RiskParity,
    TrackingErrorCap,
    TransactionCost,
    TurnoverCap,
)

_BAD_FLOATS = [
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
]


def _current_weights(value: float) -> PortfolioSpec:
    return PortfolioSpec(
        universe=["A"],
        objective=MinVariance(),
        current_weights={"A": value},
    )


_FLOAT_FIELD_FACTORIES: list[tuple[str, Callable[[float], object]]] = [
    ("mean-variance-risk-aversion", lambda value: MeanVariance(risk_aversion=value)),
    ("min-cvar-alpha", lambda value: MinCVaR(cvar_alpha=value)),
    ("max-sharpe-risk-free-rate", lambda value: MaxSharpe(risk_free_rate=value)),
    ("budget-total", lambda value: Budget(total=value)),
    ("box-lower", lambda value: Box(lower=value, upper=1.0)),
    ("box-upper", lambda value: Box(lower=0.0, upper=value)),
    ("group-cap-maximum", lambda value: GroupCap(group="g", max_weight=value)),
    (
        "group-cap-minimum",
        lambda value: GroupCap(group="g", max_weight=1.0, min_weight=value),
    ),
    ("turnover-cap", lambda value: TurnoverCap(max_turnover=value)),
    ("transaction-cost", lambda value: TransactionCost(bps=value)),
    ("cvar-limit-alpha", lambda value: CVaRLimit(alpha=value, max_cvar=0.1)),
    ("cvar-limit-maximum", lambda value: CVaRLimit(alpha=0.95, max_cvar=value)),
    (
        "tracking-error-cap",
        lambda value: TrackingErrorCap(benchmark="bench", max_te=value),
    ),
    (
        "factor-exposure-minimum",
        lambda value: FactorExposure(factor="factor", min_exposure=value),
    ),
    (
        "factor-exposure-maximum",
        lambda value: FactorExposure(factor="factor", max_exposure=value),
    ),
    ("cardinality-minimum-position", lambda value: Cardinality(max_names=1, min_position=value)),
    ("current-weights", _current_weights),
]


@pytest.mark.parametrize(
    "factory",
    [pytest.param(factory, id=name) for name, factory in _FLOAT_FIELD_FACTORIES],
)
@pytest.mark.parametrize("bad_value", _BAD_FLOATS)
def test_typed_ir_rejects_every_nonfinite_float_field(
    factory: Callable[[float], object], bad_value: float
) -> None:
    """Every public float-bearing IR field inherits the same finite-number rule."""
    with pytest.raises(ValidationError) as exc_info:
        factory(bad_value)
    assert "finite_number" in {error["type"] for error in exc_info.value.errors()}


def test_typed_ir_rejects_string_coerced_nonfinite_number() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Budget.model_validate({"total": "NaN"})
    assert "finite_number" in {error["type"] for error in exc_info.value.errors()}


def _base_spec() -> PortfolioSpec:
    return PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[Budget(), LongOnly()],
    )


def _assert_clarabel_data_finite(compiled: CompiledProblem) -> None:
    data = compiled.problem.get_problem_data(cp.CLARABEL)[0]
    for key in ("P", "A", "b", "c"):
        value = data.get(key)
        if value is None:
            continue
        array = value.data if hasattr(value, "toarray") else np.asarray(value)
        assert np.all(np.isfinite(array)), f"non-finite Clarabel data in {key}"


@pytest.mark.parametrize("bad_value", _BAD_FLOATS)
def test_compile_rejects_nonfinite_expected_returns(bad_value: float) -> None:
    mu = np.array([0.0, bad_value])
    with pytest.raises(CompilationError, match="Expected-return vector.*finite"):
        compile_spec(_base_spec(), mu=mu, sigma=np.eye(2))


@pytest.mark.parametrize("bad_value", _BAD_FLOATS)
def test_compile_rejects_nonfinite_covariance(bad_value: float) -> None:
    sigma = np.eye(2)
    sigma[0, 0] = bad_value
    with pytest.raises(CompilationError, match="Covariance matrix.*finite"):
        compile_spec(_base_spec(), mu=np.zeros(2), sigma=sigma)


@pytest.mark.parametrize("bad_value", _BAD_FLOATS)
def test_compile_rejects_nonfinite_supplied_scenarios_even_when_unused(
    bad_value: float,
) -> None:
    scenarios = np.array([[0.01, bad_value]])
    with pytest.raises(CompilationError, match="Scenario matrix.*finite"):
        compile_spec(_base_spec(), mu=np.zeros(2), sigma=np.eye(2), scenarios=scenarios)


@pytest.mark.parametrize("bad_value", _BAD_FLOATS)
def test_compile_rejects_nonfinite_pretrade_weights(bad_value: float) -> None:
    with pytest.raises(CompilationError, match="w_prev vector.*finite"):
        compile_spec(
            _base_spec(),
            mu=np.zeros(2),
            sigma=np.eye(2),
            w_prev=np.array([0.0, bad_value]),
        )


@pytest.mark.parametrize("bad_value", _BAD_FLOATS)
def test_compile_rejects_every_nonfinite_supplied_benchmark(bad_value: float) -> None:
    with pytest.raises(CompilationError, match="Benchmark 'unused' vector.*finite"):
        compile_spec(
            _base_spec(),
            mu=np.zeros(2),
            sigma=np.eye(2),
            benchmark_weights={"unused": np.array([0.0, bad_value])},
        )


@pytest.mark.parametrize("bad_value", _BAD_FLOATS)
def test_compile_rejects_every_nonfinite_supplied_factor(bad_value: float) -> None:
    with pytest.raises(CompilationError, match="Factor 'unused' vector.*finite"):
        compile_spec(
            _base_spec(),
            mu=np.zeros(2),
            sigma=np.eye(2),
            factor_loadings={"unused": np.array([0.0, bad_value])},
        )


@pytest.mark.parametrize(
    ("keyword", "label"),
    [
        ("benchmark_weights", "Benchmark 'bad' vector shape"),
        ("factor_loadings", "Factor 'bad' vector shape"),
    ],
)
def test_compile_rejects_misaligned_named_vectors(keyword: str, label: str) -> None:
    with pytest.raises(CompilationError, match=label):
        compile_spec(
            _base_spec(),
            mu=np.zeros(2),
            sigma=np.eye(2),
            **{keyword: {"bad": np.zeros(3)}},
        )


def test_compile_coerces_numeric_array_likes() -> None:
    compiled = compile_spec(_base_spec(), mu=[0.0, 0.0], sigma=[[1.0, 0.0], [0.0, 1.0]])
    assert compiled.problem.is_dcp()


def test_compile_normalizes_array_conversion_errors() -> None:
    with pytest.raises(CompilationError, match="Expected-return vector must be a numeric array"):
        compile_spec(_base_spec(), mu=["not-a-number", 0.0], sigma=np.eye(2))


@pytest.mark.parametrize(
    ("mu", "sigma", "label"),
    [
        (np.array([0.0 + 1.0j, 0.0]), np.eye(2), "Expected-return vector"),
        (np.zeros(2), np.eye(2, dtype=complex), "Covariance matrix"),
    ],
)
def test_compile_rejects_complex_arrays_without_discarding_imaginary_parts(
    mu: np.ndarray, sigma: np.ndarray, label: str
) -> None:
    with pytest.raises(CompilationError, match=f"{label}.*real values"):
        compile_spec(_base_spec(), mu=mu, sigma=sigma)


def test_compile_rejects_covariance_that_overflows_quadratic_coefficients() -> None:
    with pytest.raises(CompilationError, match="Quadratic covariance coefficients.*finite"):
        compile_spec(_base_spec(), mu=np.zeros(2), sigma=np.eye(2) * 1e308)


def test_compile_rejects_overflowed_mean_variance_coefficients() -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MeanVariance(risk_aversion=1e200),
        constraints=[Budget(), LongOnly()],
    )
    with pytest.raises(
        CompilationError, match="Mean-variance scaled expected-return vector.*finite"
    ):
        compile_spec(spec, mu=np.array([1e200, 1e200]), sigma=np.eye(2))


def test_compile_rejects_overflowed_max_sharpe_excess_returns() -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MaxSharpe(risk_free_rate=-1e308),
        constraints=[LongOnly()],
    )
    with pytest.raises(CompilationError, match="max_sharpe excess-return vector.*finite"):
        compile_spec(spec, mu=np.array([1e308, 1e308]), sigma=np.eye(2))


def test_risk_parity_rejects_numerically_singular_covariance() -> None:
    spec = PortfolioSpec(universe=["A", "B"], objective=RiskParity(), constraints=[])
    with pytest.raises(CompilationError, match="requires a positive-definite covariance"):
        compile_spec(spec, mu=np.zeros(2), sigma=np.diag([1.0, 0.0]))


def test_centered_tracking_error_accepts_finite_solver_data_without_expansion() -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinTrackingError(benchmark="huge"),
        constraints=[Budget(total=2e100)],
    )
    compiled = compile_spec(
        spec,
        mu=np.zeros(2),
        sigma=np.eye(2) * 1e200,
        benchmark_weights={"huge": np.array([1e100, 1e100])},
    )
    _assert_clarabel_data_finite(compiled)


@pytest.mark.parametrize("objective", [MinCVaR(), RiskParity()])
def test_non_full_quadratic_objectives_accept_representable_extreme_covariance(
    objective: object,
) -> None:
    spec = PortfolioSpec(universe=["A", "B"], objective=objective, constraints=[])
    kwargs = {"scenarios": np.zeros((2, 2))} if isinstance(objective, MinCVaR) else {}
    # CVXPY 1.9 infers reduction shapes using uninitialized temporary arrays;
    # near-float-max test data can provoke a spurious warning even when the
    # canonical solver matrices verified below are finite.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="overflow encountered in reduce")
        compiled = compile_spec(
            spec,
            mu=np.zeros(2),
            sigma=np.eye(2) * 1e308,
            **kwargs,
        )
        _assert_clarabel_data_finite(compiled)


def test_compiler_owns_numeric_arrays_after_compilation() -> None:
    loadings = np.array([1.0, 2.0])
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[Budget(), FactorExposure(factor="f", max_exposure=2.0)],
    )
    compiled = compile_spec(
        spec,
        mu=np.zeros(2),
        sigma=np.eye(2),
        factor_loadings={"f": loadings},
    )
    loadings[0] = float("nan")
    for constant in compiled.problem.constants():
        assert np.all(np.isfinite(np.asarray(constant.value, dtype=float)))
    _assert_clarabel_data_finite(compiled)


def test_tracking_error_cap_rejects_overflowed_transformed_benchmark() -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[Budget(), TrackingErrorCap(benchmark="huge", max_te=1.0)],
    )
    with pytest.raises(CompilationError, match="Tracking-error transformed benchmark.*finite"):
        compile_spec(
            spec,
            mu=np.zeros(2),
            sigma=np.eye(2) * 1e220,
            benchmark_weights={"huge": np.array([1e220, 1e220])},
        )


def test_compiler_revalidates_in_place_mutated_ir_containers() -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[Budget()],
        current_weights={"A": 0.0},
    )
    assert spec.current_weights is not None
    spec.current_weights["A"] = float("nan")
    with pytest.raises(CompilationError, match="PortfolioSpec is invalid at compilation"):
        compile_spec(spec, mu=np.zeros(2), sigma=np.eye(2))


def test_compiler_revalidates_cross_field_invariants_after_mutation() -> None:
    box = Box(lower=0.0, upper=1.0)
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[Budget(), box],
    )
    box.lower = 2.0
    with pytest.raises(CompilationError, match="PortfolioSpec is invalid at compilation"):
        compile_spec(spec, mu=np.zeros(2), sigma=np.eye(2))


@pytest.mark.parametrize(
    ("lower", "upper", "message"),
    [
        (np.array([0.0, float("nan")]), np.ones(2), "lower bounds.*finite"),
        (np.zeros(2), np.array([1.0, float("inf")]), "upper bounds.*finite"),
        (np.zeros(1), np.ones(2), "lower bounds shape"),
        (np.array([0.0 + 1.0j, 0.0]), np.ones(2), "lower bounds.*real"),
        (np.ones(2), np.zeros(2), "lower bounds must not exceed upper bounds"),
    ],
)
def test_cardinality_diagnostic_arrays_are_validated_before_cvxpy(
    lower: np.ndarray, upper: np.ndarray, message: str
) -> None:
    with pytest.raises(CompilationError, match=message):
        compile_spec(
            _base_spec(),
            mu=np.zeros(2),
            sigma=np.eye(2),
            _cardinality_weight_bounds=(lower, upper),
        )


def _covariance_with_eigenvalues(scale: float, negative_ratio: float) -> np.ndarray:
    theta = 0.37
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return rotation @ np.diag([scale, negative_ratio * scale]) @ rotation.T


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_covariance_roundoff_within_relative_psd_tolerance_is_projected(scale: float) -> None:
    sigma = _covariance_with_eigenvalues(scale, negative_ratio=-0.5e-10)
    _, sanitized = validate_inputs(_base_spec(), np.zeros(2), sigma)
    min_eigenvalue = float(np.linalg.eigvalsh(sanitized)[0])
    assert min_eigenvalue >= -1e-14 * scale
    np.testing.assert_array_equal(sanitized, sanitized.T)


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_covariance_beyond_relative_psd_tolerance_is_rejected(scale: float) -> None:
    sigma = _covariance_with_eigenvalues(scale, negative_ratio=-2.0e-10)
    with pytest.raises(CompilationError, match="not positive semidefinite"):
        validate_inputs(_base_spec(), np.zeros(2), sigma)


def test_exact_singular_psd_covariance_is_accepted() -> None:
    _, sanitized = validate_inputs(_base_spec(), np.zeros(2), np.diag([1.0, 0.0]))
    np.testing.assert_array_equal(sanitized, np.diag([1.0, 0.0]))


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_covariance_symmetry_tolerance_is_relative_to_scale(scale: float) -> None:
    accepted = np.array([[scale, 0.5e-8 * scale], [0.0, scale]])
    _, sanitized = validate_inputs(_base_spec(), np.zeros(2), accepted)
    np.testing.assert_array_equal(sanitized, sanitized.T)

    rejected = np.array([[scale, 2.0e-8 * scale], [0.0, scale]])
    with pytest.raises(CompilationError, match="not symmetric"):
        validate_inputs(_base_spec(), np.zeros(2), rejected)


def test_invalid_covariance_fails_before_cvxpy_variable_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_variable(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("CVXPY variable construction must not be reached")

    monkeypatch.setattr("core.compiler.cp.Variable", forbidden_variable)
    with pytest.raises(CompilationError, match="not positive semidefinite"):
        compile_spec(
            _base_spec(),
            mu=np.zeros(2),
            sigma=np.array([[1.0, 2.0], [2.0, 1.0]]),
        )
