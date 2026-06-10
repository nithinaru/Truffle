"""Tests for core/duals.py.

The headline test puts up a deliberately binding Box upper bound and checks
the harvested shadow price is positive while a non-binding constraint
returns ~zero. We also assert the not-solved / infeasible paths raise the
typed exceptions the explanation layer expects.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from core.compiler import compile_spec
from core.duals import harvest_duals
from core.exceptions import DualsUnavailableError, InfeasibleError
from core.ir import Box, Budget, LongOnly, MinVariance, PortfolioSpec


def test_binding_box_has_positive_shadow_price() -> None:
    """Min-variance on Σ = diag(0.04, 0.09, 0.16) wants unconstrained
    inverse-variance weights w* ∝ (25, 11.11, 6.25)/Σ ≈ (0.59, 0.26, 0.15).

    Force the cap on asset A to 0.40 (well below its preferred 0.59) so the
    Box becomes binding. The dual of the (binding) upper-bound entry must be
    strictly positive; the long-only dual (non-binding here) must be ~0.
    """
    universe = ["A", "B", "C"]
    sigma = np.diag([0.04, 0.09, 0.16])
    mu = np.zeros(3)

    spec = PortfolioSpec(
        universe=universe,
        objective=MinVariance(),
        constraints=[
            Budget(id="budget"),
            LongOnly(id="long_only"),
            Box(id="cap_A", lower=0.0, upper=0.40, tickers=["A"]),
        ],
    )
    compiled = compile_spec(spec, mu, sigma)
    compiled.problem.solve(solver=cp.CLARABEL)
    assert compiled.problem.status == "optimal"

    duals = harvest_duals(compiled)

    # All IR ids accounted for.
    assert set(duals.keys()) == {"budget", "long_only", "cap_A"}

    # Cap on A is binding — solver picks w_A = 0.40 exactly.
    np.testing.assert_allclose(compiled.weights.value[0], 0.40, atol=1e-6)
    assert duals["cap_A"] > 1e-6, f"Expected positive shadow price, got {duals['cap_A']}"

    # Long-only is slack at this solution (all w_i > 0), so its dual is ~0.
    assert duals["long_only"] < 1e-6


def test_harvest_duals_raises_when_not_solved() -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[Budget()],
    )
    compiled = compile_spec(spec, mu=np.zeros(2), sigma=np.eye(2))
    with pytest.raises(DualsUnavailableError, match="not been solved"):
        harvest_duals(compiled)


def test_harvest_duals_raises_on_infeasible_problem() -> None:
    """Box upper = 0.10 on every asset with Budget = 1 cannot satisfy
    Σ w = 1 across only 3 assets (max achievable Σw = 0.30). Infeasible."""
    spec = PortfolioSpec(
        universe=["A", "B", "C"],
        objective=MinVariance(),
        constraints=[
            Budget(total=1.0),
            LongOnly(),
            Box(lower=0.0, upper=0.10),
        ],
    )
    compiled = compile_spec(spec, mu=np.zeros(3), sigma=np.eye(3))
    compiled.problem.solve(solver=cp.CLARABEL)
    assert "infeasible" in compiled.problem.status
    with pytest.raises(InfeasibleError, match="infeasible"):
        harvest_duals(compiled)
