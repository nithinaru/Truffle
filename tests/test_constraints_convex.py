"""Slice 2 tests: the six new convex constraint nodes.

Each test asserts (a) the math actually binds/relaxes as intended on sample
data, and (b) the guardrail invariant — the problem stays convex and Clarabel
solves it. Convexity is checked via ``spec.problem_class == "convex"`` and a
clean Clarabel ``optimal`` status.
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
    CVaRLimit,
    FactorExposure,
    GroupCap,
    LongOnly,
    MinVariance,
    PortfolioSpec,
    TrackingErrorCap,
    TransactionCost,
    TurnoverCap,
)

# A small, well-conditioned 4-asset risk model reused across tests.
_UNIVERSE = ["A", "B", "C", "D"]
_SIGMA = np.diag([0.04, 0.09, 0.16, 0.25]).astype(float)
_MU = np.array([0.08, 0.10, 0.12, 0.06])
_SECTORS = {"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"}


def _solve(spec: PortfolioSpec, **kw) -> tuple[np.ndarray, object]:
    compiled = compile_spec(spec, mu=_MU, sigma=_SIGMA, **kw)
    compiled.problem.solve(solver=cp.CLARABEL)
    assert compiled.problem.status == "optimal"
    # Guardrail: every Sprint-3 node stays convex / Clarabel-solvable.
    assert spec.problem_class == "convex"
    return np.asarray(compiled.weights.value), compiled


def test_group_cap_binds_and_relaxes() -> None:
    base = PortfolioSpec(
        universe=_UNIVERSE, objective=MinVariance(), constraints=[Budget(), LongOnly()]
    )
    w_base, _ = _solve(base)
    tech_base = w_base[0] + w_base[1]

    capped = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MinVariance(),
        constraints=[
            Budget(),
            LongOnly(),
            GroupCap(id="tech", group="Tech", max_weight=0.30),
        ],
    )
    w_cap, _ = _solve(capped, sectors=_SECTORS)
    assert tech_base > 0.30  # min-variance naturally piles into low-vol Tech names
    assert w_cap[0] + w_cap[1] <= 0.30 + 1e-6


def test_group_cap_missing_mapping_raises() -> None:
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MinVariance(),
        constraints=[Budget(), LongOnly(), GroupCap(group="Tech", max_weight=0.3)],
    )
    with pytest.raises(CompilationError, match="ticker->group mapping"):
        compile_spec(spec, mu=_MU, sigma=_SIGMA)  # no sectors


def test_group_cap_unknown_group_raises() -> None:
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MinVariance(),
        constraints=[Budget(), LongOnly(), GroupCap(group="Healthcare", max_weight=0.3)],
    )
    with pytest.raises(CompilationError, match="matches no ticker"):
        compile_spec(spec, mu=_MU, sigma=_SIGMA, sectors=_SECTORS)


def test_turnover_cap_limits_distance_from_w_prev() -> None:
    w_prev = np.array([0.25, 0.25, 0.25, 0.25])
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MinVariance(),
        constraints=[
            Budget(),
            LongOnly(),
            TurnoverCap(max_turnover=0.10),
        ],
    )
    w, _ = _solve(spec, w_prev=w_prev)
    turnover = float(np.sum(np.abs(w - w_prev)))
    assert turnover <= 0.10 + 1e-6


def test_transaction_cost_pulls_solution_toward_w_prev() -> None:
    w_prev = np.array([0.25, 0.25, 0.25, 0.25])
    no_cost = PortfolioSpec(
        universe=_UNIVERSE, objective=MinVariance(), constraints=[Budget(), LongOnly()]
    )
    w_free, _ = _solve(no_cost, w_prev=w_prev)

    with_cost = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MinVariance(),
        constraints=[Budget(), LongOnly(), TransactionCost(bps=500.0)],
    )
    compiled = compile_spec(with_cost, mu=_MU, sigma=_SIGMA, w_prev=w_prev)
    compiled.problem.solve(solver=cp.CLARABEL)
    w_taxed = np.asarray(compiled.weights.value)

    # A costly trade penalty keeps us closer to the starting weights.
    assert np.sum(np.abs(w_taxed - w_prev)) < np.sum(np.abs(w_free - w_prev)) - 1e-4
    # Penalty-only node: no hard constraint, hence no dual entry.
    assert all(not k.startswith("txcost") for k in compiled.constraint_objs)


def test_cvar_limit_caps_tail_risk() -> None:
    rng = np.random.default_rng(1)
    scenarios = rng.normal(0.0, 0.03, size=(400, 4))
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MinVariance(),
        constraints=[
            Budget(),
            LongOnly(),
            CVaRLimit(id="cv", alpha=0.95, max_cvar=0.04),
        ],
    )
    compiled = compile_spec(spec, mu=_MU, sigma=_SIGMA, scenarios=scenarios)
    compiled.problem.solve(solver=cp.CLARABEL)
    assert compiled.problem.status == "optimal"
    assert spec.problem_class == "convex"
    # Realized CVaR of the solution respects the cap (the named constraint holds).
    w = np.asarray(compiled.weights.value)
    losses = -scenarios @ w
    var = float(compiled.extra_vars["t_cv"].value)
    tail = losses[losses >= var - 1e-9]
    realized_cvar = var + np.mean(np.maximum(losses - var, 0.0)) / (1 - 0.95)
    assert realized_cvar <= 0.04 + 1e-3
    assert tail.size > 0


def test_cvar_limit_without_scenarios_raises() -> None:
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MinVariance(),
        constraints=[Budget(), LongOnly(), CVaRLimit(alpha=0.95, max_cvar=0.04)],
    )
    with pytest.raises(CompilationError, match="scenario matrix"):
        compile_spec(spec, mu=_MU, sigma=_SIGMA)


def test_tracking_error_cap_pulls_toward_benchmark() -> None:
    benchmark = {"bench": np.array([0.25, 0.25, 0.25, 0.25])}
    tight = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MinVariance(),
        constraints=[
            Budget(),
            LongOnly(),
            TrackingErrorCap(benchmark="bench", max_te=0.02),
        ],
    )
    w_tight, _ = _solve(tight, benchmark_weights=benchmark)
    loose = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MinVariance(),
        constraints=[
            Budget(),
            LongOnly(),
            TrackingErrorCap(benchmark="bench", max_te=0.20),
        ],
    )
    w_loose, _ = _solve(loose, benchmark_weights=benchmark)
    b = benchmark["bench"]
    # A tighter TE cap forces the portfolio closer to the benchmark.
    assert np.linalg.norm(w_tight - b) < np.linalg.norm(w_loose - b) + 1e-9


def test_tracking_error_missing_benchmark_raises() -> None:
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MinVariance(),
        constraints=[Budget(), LongOnly(), TrackingErrorCap(benchmark="spx", max_te=0.05)],
    )
    with pytest.raises(CompilationError, match="were not supplied"):
        compile_spec(spec, mu=_MU, sigma=_SIGMA)


def test_factor_exposure_bounds_loadings() -> None:
    loadings = {"value": np.array([1.0, 0.5, -0.5, -1.0])}
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MinVariance(),
        constraints=[
            Budget(),
            LongOnly(),
            FactorExposure(factor="value", max_exposure=0.10),
        ],
    )
    w, _ = _solve(spec, factor_loadings=loadings)
    assert float(loadings["value"] @ w) <= 0.10 + 1e-6


def test_factor_exposure_requires_a_bound() -> None:
    with pytest.raises(ValueError, match="at least one of"):
        FactorExposure(factor="value")


def test_factor_exposure_missing_data_raises() -> None:
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MinVariance(),
        constraints=[Budget(), LongOnly(), FactorExposure(factor="value", max_exposure=0.1)],
    )
    with pytest.raises(CompilationError, match="were not supplied"):
        compile_spec(spec, mu=_MU, sigma=_SIGMA)


def test_box_still_works_after_refactor() -> None:
    """Regression: Sprint-1 Box keeps its inline builder and dual id."""
    spec = PortfolioSpec(
        universe=_UNIVERSE,
        objective=MinVariance(),
        constraints=[Budget(), Box(id="cap", lower=0.0, upper=0.4)],
    )
    _w, compiled = _solve(spec)
    assert "cap" in compiled.constraint_objs
