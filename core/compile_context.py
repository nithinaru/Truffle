"""Compilation context and shared CVXPY building blocks.

A leaf module (imports only cvxpy / numpy / :mod:`core.exceptions`) so that the
per-constraint modules under ``core/constraints`` and the main
:mod:`core.compiler` can both depend on it without an import cycle.

:class:`BuildContext` is the bundle every constraint builder receives: the
weight variable, universe metadata, the pre-trade vector, the risk inputs, and
the optional data inputs (group map, benchmark weights, factor loadings). It
also carries the mutable accumulators a builder may append to:

* ``penalties`` — objective penalty terms (TransactionCost).
* ``aux_constraints`` — extra (unnamed) constraints the node needs but that do
  not carry a dual of their own (e.g. CVaR linking inequalities, L1 epigraph
  rows). These are *not* registered in the ``{id -> Constraint}`` map.
* ``extra_vars`` — auxiliary variables a caller may want to read post-solve.

A builder returns the single *named* hard constraint to register under the
node's id (so :mod:`core.duals` can attach a shadow price), or ``None`` when the
node only contributes a penalty (TransactionCost).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np

from core.exceptions import CompilationError

if TYPE_CHECKING:
    from core.ir import PortfolioSpec


@dataclass(slots=True)
class CompiledProblem:
    """Container for everything the solver layer needs after compilation.

    Attributes:
        problem: The CVXPY ``Problem``. Solve it externally so the compiler
            stays a pure builder (easier to test, easier to reason about).
        weights: The ``cp.Variable`` whose value the solver populates. For most
            objectives this *is* the portfolio weight vector. For change-of-
            variable objectives (``max_sharpe``, ``risk_parity``) it is the
            transformed variable, and ``weight_recovery`` maps its post-solve
            value back to portfolio weights — call :meth:`recovered_weights`.
        constraint_objs: ``{ir_constraint_id -> cvxpy.Constraint}``. Used by
            :mod:`core.duals` to recover shadow prices and name them back to
            the user. Only *hard* constraints appear here — penalty-only nodes
            (e.g. ``TransactionCost``) contribute to the objective and are
            intentionally absent (they have no dual).
        spec: The originating ``PortfolioSpec`` (kept for downstream reporting).
        extra_vars: Objective-specific auxiliary variables. For ``min_cvar``
            this exposes ``{"t": <scalar Variable>, "z": <S-vector Variable>}``
            so the caller can read VaR (``= t.value``) after the solve.
        weight_recovery: Optional post-solve callable returning the final
            portfolio weights as an array. ``None`` means weights are read
            directly from ``weights.value``.
    """

    problem: cp.Problem
    weights: cp.Variable
    constraint_objs: dict[str, cp.Constraint] = field(default_factory=dict)
    spec: PortfolioSpec | None = None
    extra_vars: dict[str, cp.Variable] = field(default_factory=dict)
    weight_recovery: Callable[[], np.ndarray] | None = None

    def recovered_weights(self) -> np.ndarray:
        """Return final portfolio weights, applying ``weight_recovery`` if set."""
        if self.weight_recovery is not None:
            return np.asarray(self.weight_recovery(), dtype=float)
        return np.asarray(self.weights.value, dtype=float)


def validate_inputs(spec: PortfolioSpec, mu: np.ndarray, sigma: np.ndarray) -> None:
    """Shape/symmetry checks shared by every compile path."""
    n = len(spec.universe)
    if sigma.shape != (n, n):
        raise CompilationError(f"Covariance shape {sigma.shape} does not match universe size {n}.")
    if mu.shape != (n,):
        raise CompilationError(
            f"Expected-return vector shape {mu.shape} does not match universe size ({n},)."
        )
    # Symmetrize sigma defensively — CVXPY's `quad_form` insists on PSD, and
    # off-by-eps asymmetry from floating point is a common compile-time surprise.
    asym = float(np.max(np.abs(sigma - sigma.T))) if sigma.size else 0.0
    if asym > 1e-8:
        raise CompilationError(f"Covariance matrix is not symmetric (max |Σ − Σᵀ| = {asym:.2e}).")


def resolve_w_prev(w_prev: np.ndarray | None, n: int) -> np.ndarray:
    """Resolve the pre-trade weight vector used by turnover / transaction cost.

    Convention (documented on :attr:`core.ir.PortfolioSpec.current_weights`):
    ``None`` means a zero vector of length ``n`` — the portfolio is being built
    fresh from cash, so every position change equals the target weight. A
    supplied vector must already be aligned to the universe and length ``n``.
    """
    if w_prev is None:
        return np.zeros(n, dtype=float)
    w_prev = np.asarray(w_prev, dtype=float)
    if w_prev.shape != (n,):
        raise CompilationError(
            f"w_prev vector shape {w_prev.shape} does not match universe size ({n},)."
        )
    return w_prev


def validate_scenarios(scenarios: np.ndarray | None, n: int) -> np.ndarray:
    """Coerce and shape-check a scenario matrix for the Rockafellar–Uryasev LP.

    Shared by the ``MinCVaR`` objective and the ``CVaRLimit`` constraint so the
    error messages (and the contract) are identical at both call sites.
    """
    if scenarios is None:
        raise CompilationError(
            "min_cvar objective requires a scenario matrix; got scenarios=None. "
            "Pass scenarios from data.scenarios.historical_scenarios(prices) (or another generator)."
        )
    scenarios = np.asarray(scenarios, dtype=float)
    if scenarios.ndim != 2 or scenarios.shape[1] != n:
        raise CompilationError(f"Scenario matrix must have shape (S, {n}); got {scenarios.shape}.")
    if scenarios.shape[0] < 1:
        raise CompilationError("Scenario matrix must have at least one scenario row.")
    return scenarios


def build_cvar_block(
    scenarios: np.ndarray,
    w: cp.Variable,
    alpha: float,
    *,
    name_suffix: str = "",
) -> tuple[cp.Expression, cp.Variable, cp.Variable, list[cp.Constraint]]:
    """Build the Rockafellar–Uryasev CVaR block for weights ``w``.

    For ``S`` equally-likely return scenarios (rows of ``scenarios``), portfolio
    losses are ``L_s = -r_s · w`` and

        CVaR_α(w) = min_t  t + (1/((1-α)S)) Σ_s max(L_s - t, 0).

    Linearizing the ``max`` with non-negative slacks ``z_s`` gives the LP whose
    auxiliary pieces this helper returns. The optimal ``t`` is the VaR.

    Args:
        scenarios: ``(S, n)`` return matrix (already validated/coerced).
        w: weight variable of length ``n``.
        alpha: CVaR confidence level in ``(0, 1)``.
        name_suffix: appended to the ``t``/``z`` variable names so multiple
            blocks (e.g. a ``MinCVaR`` objective plus one or more ``CVaRLimit``
            constraints) coexist with distinct names.

    Returns:
        ``(cvar_expr, t_var, z_var, aux_constraints)`` — the scalar CVaR
        expression, the VaR variable ``t``, the slack vector ``z``, and the
        list of linking constraints (``z >= loss - t``) the caller must add to
        the problem. ``z >= 0`` is encoded via ``nonneg=True``.
    """
    s = scenarios.shape[0]
    t_var = cp.Variable(name=f"t{name_suffix}")
    z_var = cp.Variable(s, name=f"z{name_suffix}", nonneg=True)
    loss = -scenarios @ w
    aux: list[cp.Constraint] = [z_var >= loss - t_var]
    cvar_expr = t_var + cp.sum(z_var) / ((1.0 - alpha) * s)
    return cvar_expr, t_var, z_var, aux


def abs_deviation(
    w: cp.Variable, w_prev: np.ndarray, *, name_suffix: str
) -> tuple[cp.Variable, list[cp.Constraint]]:
    """Epigraph linearization of ``|w - w_prev|`` elementwise.

    Returns a non-negative variable ``u`` and the two epigraph constraints
    ``u >= w - w_prev`` and ``u >= w_prev - w``. ``sum(u)`` is then the L1 norm
    ``‖w - w_prev‖₁`` used by both turnover (a hard cap) and transaction cost (a
    penalty). Shared so both nodes linearize the L1 the same way.
    """
    n = w.shape[0]
    u = cp.Variable(n, name=f"u{name_suffix}", nonneg=True)
    delta = w - w_prev
    cons = [u >= delta, u >= -delta]
    return u, cons


@dataclass
class BuildContext:
    """Everything a constraint builder needs, plus mutable accumulators."""

    w: cp.Variable
    n: int
    ticker_index: dict[str, int]
    w_prev: np.ndarray
    sigma: np.ndarray
    scenarios: np.ndarray | None = None
    group_map: dict[str, str] | None = None
    benchmark_weights: dict[str, np.ndarray] | None = None
    factor_loadings: dict[str, np.ndarray] | None = None
    # Per-asset weight upper bounds (from Box nodes), used as the cardinality
    # big-M. ``None`` means "no tighter cap than 1.0 per name".
    weight_upper: np.ndarray | None = None
    # Per-asset lower bounds for the Cardinality selection link. Production
    # long-only specs use zero; diagnostic counterfactuals may supply rigorously
    # derived signed bounds after deleting a domain constraint.
    weight_lower: np.ndarray | None = None

    penalties: list[cp.Expression] = field(default_factory=list)
    aux_constraints: list[cp.Constraint] = field(default_factory=list)
    extra_vars: dict[str, cp.Variable] = field(default_factory=dict)

    def aligned_benchmark(self, name: str) -> np.ndarray:
        """Return benchmark weights aligned to the universe, or raise."""
        if self.benchmark_weights is None or name not in self.benchmark_weights:
            raise CompilationError(
                f"Benchmark {name!r} weights were not supplied. Pass --benchmark "
                "(CLI) or benchmark_weights to compile_spec."
            )
        b = np.asarray(self.benchmark_weights[name], dtype=float)
        if b.shape != (self.n,):
            raise CompilationError(
                f"Benchmark {name!r} vector shape {b.shape} does not match "
                f"universe size ({self.n},)."
            )
        return b

    def aligned_factor(self, name: str) -> np.ndarray:
        """Return factor loadings aligned to the universe, or raise."""
        if self.factor_loadings is None or name not in self.factor_loadings:
            raise CompilationError(
                f"Factor {name!r} loadings were not supplied. Pass --factors "
                "(CLI) or factor_loadings to compile_spec."
            )
        loadings = np.asarray(self.factor_loadings[name], dtype=float)
        if loadings.shape != (self.n,):
            raise CompilationError(
                f"Factor {name!r} loadings shape {loadings.shape} does not match "
                f"universe size ({self.n},)."
            )
        return loadings
