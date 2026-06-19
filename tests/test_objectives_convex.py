"""Slice 3 tests: the three new convex objectives.

Each asserts the optimization math plus the convexity / Clarabel guardrail.
For the change-of-variable objectives we check the *recovered* weights (post
Charnes–Cooper / post log-barrier normalization), not the raw transformed
variable.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from core.compiler import compile_spec
from core.exceptions import CompilationError
from core.ir import (
    Box,
    Budget,
    GroupCap,
    LongOnly,
    MaxSharpe,
    MinTrackingError,
    PortfolioSpec,
    RiskParity,
)

_UNIVERSE = ["A", "B", "C"]
_SIGMA = np.diag([0.04, 0.09, 0.16]).astype(float)
_MU = np.array([0.12, 0.10, 0.08])


def _solve(spec: PortfolioSpec, **kw):
    compiled = compile_spec(spec, mu=_MU, sigma=_SIGMA, **kw)
    compiled.problem.solve(solver=cp.CLARABEL)
    assert compiled.problem.status in {"optimal", "optimal_inaccurate"}
    assert spec.problem_class == "convex"
    return compiled


# ---------------------------------------------------------------------------
# MaxSharpe
# ---------------------------------------------------------------------------


def test_max_sharpe_recovers_fully_invested_weights() -> None:
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MaxSharpe(),
        constraints=[Budget(), LongOnly()],
    )
    compiled = _solve(spec)
    w = compiled.recovered_weights()
    np.testing.assert_allclose(np.sum(w), 1.0, atol=1e-6)
    assert np.all(w >= -1e-7)


def test_max_sharpe_beats_minvariance_sharpe() -> None:
    """The max-Sharpe portfolio must have Sharpe >= the min-variance portfolio's."""
    from core.ir import MinVariance

    sharpe_spec = PortfolioSpec(
        universe=_UNIVERSE, objective=MaxSharpe(), constraints=[Budget(), LongOnly()]
    )
    mv_spec = PortfolioSpec(
        universe=_UNIVERSE, objective=MinVariance(), constraints=[Budget(), LongOnly()]
    )

    def sharpe(w: np.ndarray) -> float:
        return float(_MU @ w) / float(np.sqrt(w @ _SIGMA @ w))

    w_sharpe = _solve(sharpe_spec).recovered_weights()
    mv = compile_spec(mv_spec, mu=_MU, sigma=_SIGMA)
    mv.problem.solve(solver=cp.CLARABEL)
    w_mv = np.asarray(mv.weights.value)
    assert sharpe(w_sharpe) >= sharpe(w_mv) - 1e-6


def test_max_sharpe_respects_box_after_transform() -> None:
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MaxSharpe(),
        constraints=[Budget(), LongOnly(), Box(id="cap", lower=0.0, upper=0.5)],
    )
    w = _solve(spec).recovered_weights()
    assert np.all(w <= 0.5 + 1e-6)


def test_max_sharpe_rejects_unsupported_constraint() -> None:
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MaxSharpe(),
        constraints=[Budget(), LongOnly(), GroupCap(group="X", max_weight=0.5)],
    )
    with pytest.raises(CompilationError, match="unsupported with max_sharpe"):
        compile_spec(spec, mu=_MU, sigma=_SIGMA, sectors={"A": "X", "B": "X", "C": "Y"})


def test_max_sharpe_requires_long_only() -> None:
    spec = PortfolioSpec(universe=_UNIVERSE, objective=MaxSharpe(), constraints=[Budget()])
    with pytest.raises(CompilationError, match="requires a long_only"):
        compile_spec(spec, mu=_MU, sigma=_SIGMA)


def test_max_sharpe_precondition_no_excess_return() -> None:
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MaxSharpe(risk_free_rate=0.50),  # higher than every mu
        constraints=[Budget(), LongOnly()],
    )
    with pytest.raises(CompilationError, match="no portfolio beats"):
        compile_spec(spec, mu=_MU, sigma=_SIGMA)


# ---------------------------------------------------------------------------
# RiskParity
# ---------------------------------------------------------------------------


def test_risk_parity_equalizes_risk_contributions() -> None:
    # Use a non-diagonal but well-conditioned covariance.
    sigma = np.array(
        [[0.04, 0.006, 0.0], [0.006, 0.09, 0.01], [0.0, 0.01, 0.16]]
    )
    spec = PortfolioSpec(
        universe=_UNIVERSE, objective=RiskParity(), constraints=[Budget(), LongOnly()]
    )
    compiled = compile_spec(spec, mu=_MU, sigma=sigma)
    compiled.problem.solve(solver=cp.CLARABEL)
    assert compiled.problem.status in {"optimal", "optimal_inaccurate"}
    w = compiled.recovered_weights()
    np.testing.assert_allclose(np.sum(w), 1.0, atol=1e-6)
    # Risk contribution RC_i = w_i * (Σw)_i; equal across assets at the optimum.
    rc = w * (sigma @ w)
    np.testing.assert_allclose(rc, rc.mean(), rtol=5e-2)


def test_risk_parity_rejects_unsupported_constraint() -> None:
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=RiskParity(),
        constraints=[Box(id="b", lower=0.0, upper=0.5)],
    )
    with pytest.raises(CompilationError, match="unsupported with risk_parity"):
        compile_spec(spec, mu=_MU, sigma=_SIGMA)


# ---------------------------------------------------------------------------
# MinTrackingError
# ---------------------------------------------------------------------------


def test_min_tracking_error_hugs_benchmark() -> None:
    benchmark = {"bench": np.array([0.2, 0.3, 0.5])}
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MinTrackingError(benchmark="bench"),
        constraints=[Budget(), LongOnly()],
    )
    compiled = _solve(spec, benchmark_weights=benchmark)
    w = compiled.recovered_weights()
    # With only budget + long-only, the TE minimizer is the benchmark itself.
    np.testing.assert_allclose(w, benchmark["bench"], atol=1e-4)


def test_min_tracking_error_missing_benchmark_raises() -> None:
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MinTrackingError(benchmark="spx"),
        constraints=[Budget(), LongOnly()],
    )
    with pytest.raises(CompilationError, match="were not supplied"):
        compile_spec(spec, mu=_MU, sigma=_SIGMA)
