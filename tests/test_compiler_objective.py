"""Slice 0 structural tests: penalty accumulation + reusable CVaR block.

These guard the two refactors the rest of Sprint 3 depends on:

1. ``_assemble_objective(base, penalties)`` really computes ``base + Σ penalties``
   (and reduces to ``base`` when there are no penalties).
2. ``build_cvar_block`` is a faithful, reusable extraction — a hand-built
   problem using the helper reproduces the ``MinCVaR`` objective compile path.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from core.compiler import (
    _assemble_objective,
    build_cvar_block,
    compile_spec,
)
from core.ir import Budget, LongOnly, MinCVaR, PortfolioSpec


def test_assemble_objective_no_penalties_is_base() -> None:
    w = cp.Variable(2, name="w")
    base = cp.sum_squares(w)
    obj = _assemble_objective(base, [])
    # No penalties => the assembled objective minimizes exactly the base.
    prob = cp.Problem(obj, [cp.sum(w) == 1.0])
    prob.solve(solver=cp.CLARABEL)
    np.testing.assert_allclose(w.value, [0.5, 0.5], atol=1e-6)


def test_assemble_objective_adds_penalty_terms() -> None:
    w = cp.Variable(2, name="w")
    base = cp.sum_squares(w)
    # A linear penalty that pushes weight away from asset 0.
    penalty = 10.0 * w[0]
    obj = _assemble_objective(base, [penalty])
    prob = cp.Problem(obj, [cp.sum(w) == 1.0, w >= 0.0])
    prob.solve(solver=cp.CLARABEL)
    # The penalty must bend the optimum toward asset 1 vs the unpenalised 50/50.
    assert w.value[0] < 0.5 - 1e-3
    assert prob.value > base.value - 1e-9  # objective includes the extra term


def test_build_cvar_block_matches_min_cvar_objective() -> None:
    """A problem assembled from the helper reproduces the MinCVaR compile path."""
    universe = ["A", "B"]
    rng = np.random.default_rng(3)
    scenarios = rng.normal(0.0, 0.02, size=(150, 2))
    alpha = 0.9

    # Reference: the real compile path.
    spec = PortfolioSpec(
        universe=universe,
        objective=MinCVaR(cvar_alpha=alpha),
        constraints=[Budget(total=1.0), LongOnly()],
    )
    compiled = compile_spec(spec, mu=np.zeros(2), sigma=np.eye(2), scenarios=scenarios)
    compiled.problem.solve(solver=cp.CLARABEL)

    # Hand-built using only the public helper.
    w = cp.Variable(2, name="w")
    cvar_expr, t_var, _z, aux = build_cvar_block(scenarios, w, alpha, name_suffix="_ref")
    prob = cp.Problem(cp.Minimize(cvar_expr), [cp.sum(w) == 1.0, w >= 0.0, *aux])
    prob.solve(solver=cp.CLARABEL)

    np.testing.assert_allclose(prob.value, compiled.problem.value, atol=1e-6)
    np.testing.assert_allclose(
        float(t_var.value), float(compiled.extra_vars["t"].value), atol=1e-5
    )


def test_build_cvar_block_uses_name_suffix() -> None:
    w = cp.Variable(1, name="w")
    _expr, t_var, z_var, _aux = build_cvar_block(
        np.array([[0.01], [-0.02]]), w, 0.9, name_suffix="_lim"
    )
    assert t_var.name() == "t_lim"
    assert z_var.name() == "z_lim"
