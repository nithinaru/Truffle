"""IR → CVXPY compiler.

This module owns *all* of the math. The LLM layer will never construct CVXPY
expressions directly; it only emits IR, and :func:`compile_spec` deterministically
translates that IR into a CVXPY ``Problem``. No string-built expressions, no
``eval``, no LLM involvement.

The compiler also returns the dictionary ``constraint_objs`` that maps every
IR constraint's ``id`` to its CVXPY ``Constraint`` object. :mod:`core.duals`
walks that dict after the solve to lift shadow prices back into the IR's
naming, which is the foundation of the explanation layer (BLUEPRINT §5/§6).

Objective assembly (Sprint 3, Slice 0)
--------------------------------------
The final objective is **base objective + Σ(penalty terms)**. The base term
comes from the :class:`~core.ir.Objective` node; penalty terms are contributed
by *constraints* that modify the objective rather than adding a hard constraint
(the canonical case is ``TransactionCost``). Penalty accumulation is explicit —
constraints append to a local list that :func:`_assemble_objective` folds into
the base — with no hidden global state.

Reusable Rockafellar–Uryasev block (Slice 0)
---------------------------------------------
:func:`build_cvar_block` constructs the CVaR auxiliary variables ``t`` (VaR) and
``z`` (per-scenario tail slacks) plus the linking inequality. It is used both by
the ``MinCVaR`` *objective* and the ``CVaRLimit`` *constraint* so the LP
reformulation lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from core.exceptions import CompilationError
from core.ir import Box, Budget, LongOnly, MeanVariance, MinCVaR, MinVariance, PortfolioSpec


@dataclass(slots=True)
class CompiledProblem:
    """Container for everything the solver layer needs after compilation.

    Attributes:
        problem: The CVXPY ``Problem``. Solve it externally so the compiler
            stays a pure builder (easier to test, easier to reason about).
        weights: The ``cp.Variable`` representing the asset weight vector.
        constraint_objs: ``{ir_constraint_id -> cvxpy.Constraint}``. Used by
            :mod:`core.duals` to recover shadow prices and name them back to
            the user. Only *hard* constraints appear here — penalty-only nodes
            (e.g. ``TransactionCost``) contribute to the objective and are
            intentionally absent (they have no dual).
        spec: The originating ``PortfolioSpec`` (kept for downstream reporting).
        extra_vars: Objective-specific auxiliary variables. For ``min_cvar``
            this exposes ``{"t": <scalar Variable>, "z": <S-vector Variable>}``
            so the caller can read VaR (``= t.value``) after the solve.
    """

    problem: cp.Problem
    weights: cp.Variable
    constraint_objs: dict[str, cp.Constraint] = field(default_factory=dict)
    spec: PortfolioSpec | None = None
    extra_vars: dict[str, cp.Variable] = field(default_factory=dict)


def _validate_inputs(spec: PortfolioSpec, mu: np.ndarray, sigma: np.ndarray) -> None:
    n = len(spec.universe)
    if sigma.shape != (n, n):
        raise CompilationError(
            f"Covariance shape {sigma.shape} does not match universe size {n}."
        )
    if mu.shape != (n,):
        raise CompilationError(
            f"Expected-return vector shape {mu.shape} does not match universe size ({n},)."
        )
    # Symmetrize sigma defensively — CVXPY's `quad_form` insists on PSD, and
    # off-by-eps asymmetry from floating point is a common compile-time surprise.
    asym = float(np.max(np.abs(sigma - sigma.T))) if sigma.size else 0.0
    if asym > 1e-8:
        raise CompilationError(
            f"Covariance matrix is not symmetric (max |Σ − Σᵀ| = {asym:.2e})."
        )


def _validate_scenarios(scenarios: np.ndarray | None, n: int) -> np.ndarray:
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
        raise CompilationError(
            f"Scenario matrix must have shape (S, {n}); got {scenarios.shape}."
        )
    if scenarios.shape[0] < 1:
        raise CompilationError("Scenario matrix must have at least one scenario row.")
    return scenarios


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


def _build_quad_objective_expr(
    spec: PortfolioSpec, w: cp.Variable, mu: np.ndarray, sigma: np.ndarray
) -> cp.Expression:
    obj = spec.objective
    # `cp.psd_wrap` tells CVXPY to trust the matrix as PSD; the Ledoit–Wolf
    # estimator we use guarantees this, but a sample Σ on a tiny window can
    # be numerically indefinite, so wrapping is the safe contract.
    quad = cp.quad_form(w, cp.psd_wrap(sigma))
    if isinstance(obj, MinVariance):
        return quad
    if isinstance(obj, MeanVariance):
        return quad - obj.risk_aversion * (mu @ w)
    raise CompilationError(
        f"_build_quad_objective_expr called on non-quadratic: {type(obj).__name__}"
    )


def _assemble_objective(
    base_expr: cp.Expression, penalties: list[cp.Expression]
) -> cp.Minimize:
    """Fold penalty terms into the base objective: ``min base + Σ penalties``.

    Kept as a tiny named function so the "objective = base + penalties" law is
    explicit and directly unit-testable, rather than inlined into the compile
    flow where it would be invisible.
    """
    expr = base_expr
    for term in penalties:
        expr = expr + term
    return cp.Minimize(expr)


def _build_constraint(
    c: Budget | LongOnly | Box,
    w: cp.Variable,
    ticker_index: dict[str, int],
    w_prev: np.ndarray,
) -> cp.Constraint:
    # w_prev is accepted uniformly so every builder has the same signature;
    # Budget/LongOnly/Box do not reference it (Slice 2 nodes will).
    if isinstance(c, Budget):
        # Σ w = total. Stays an equality so its dual is a free-sign multiplier
        # — duals on equalities can be negative; the explanation layer
        # interprets the sign per BLUEPRINT §5 "duals everywhere".
        return cp.sum(w) == c.total
    if isinstance(c, LongOnly):
        return w >= 0.0
    if isinstance(c, Box):
        if c.tickers is None:
            target = w
        else:
            idx = np.array([ticker_index[t] for t in c.tickers], dtype=int)
            target = w[idx]
        # Stack lower-side and upper-side slacks into a single non-negativity
        # constraint so this Box maps to *one* CVXPY Constraint (one id, one
        # dual vector of length 2k). First k entries are duals on the lower
        # bound, last k on the upper bound — :mod:`core.duals` documents this.
        slacks = cp.hstack([target - c.lower, c.upper - target])
        return slacks >= 0
    raise CompilationError(f"Unsupported constraint kind: {type(c).__name__}")


def compile_spec(
    spec: PortfolioSpec,
    mu: np.ndarray,
    sigma: np.ndarray,
    scenarios: np.ndarray | None = None,
    w_prev: np.ndarray | None = None,
) -> CompiledProblem:
    """Deterministically build a CVXPY problem from an IR spec.

    Args:
        spec: Validated ``PortfolioSpec``.
        mu: Expected-return vector, length ``len(spec.universe)``. Ignored by
            min-variance and min-CVaR objectives but the signature is uniform
            across kinds so callers don't branch.
        sigma: Annualized covariance matrix, shape ``(n, n)``, symmetric PSD.
            Ignored by min-CVaR but accepted for signature uniformity.
        scenarios: Per-period return matrix of shape ``(S, n)`` used by the
            CVaR objective only. ``None`` is allowed when the objective does
            not require scenarios; ``None`` with a ``min_cvar`` objective
            raises :class:`CompilationError`.
        w_prev: Pre-trade weight vector aligned to ``spec.universe``, used by
            turnover / transaction-cost terms. ``None`` resolves to the zero
            vector ("fresh from cash"); see :func:`resolve_w_prev`.

    Returns:
        ``CompiledProblem`` wrapping the unsolved CVXPY problem, the weight
        variable, the IR-id → cvxpy.Constraint map, and any objective-specific
        auxiliary variables (``t`` and ``z`` for CVaR).

    Raises:
        CompilationError: if shapes mismatch, Σ is not symmetric, scenarios
            are missing/malformed for CVaR, or the IR contains an
            objective/constraint kind the compiler does not understand.
    """
    _validate_inputs(spec, mu, sigma)

    n = len(spec.universe)
    w = cp.Variable(n, name="w")
    ticker_index = {t: i for i, t in enumerate(spec.universe)}
    # Resolved once so every constraint builder sees the same pre-trade vector.
    # Threaded into _build_constraint now; consumed by Slice 2's turnover /
    # transaction-cost nodes.
    w_prev_vec = resolve_w_prev(w_prev, n)

    constraint_objs: dict[str, cp.Constraint] = {}
    hard_constraints: list[cp.Constraint] = []
    for c in spec.constraints:
        cons = _build_constraint(c, w, ticker_index, w_prev_vec)
        constraint_objs[c.id] = cons
        hard_constraints.append(cons)

    # Penalty terms contributed by constraints that modify the objective rather
    # than adding a hard constraint. Empty until Slice 2's TransactionCost; the
    # accumulation path is wired now so that feature is a pure addition.
    penalties: list[cp.Expression] = []

    extra_vars: dict[str, cp.Variable] = {}

    obj = spec.objective
    if isinstance(obj, MinCVaR):
        scenarios = _validate_scenarios(scenarios, n)
        base_expr, t_var, z_var, aux = build_cvar_block(scenarios, w, obj.cvar_alpha)
        hard_constraints = hard_constraints + aux
        extra_vars = {"t": t_var, "z": z_var}
    else:
        base_expr = _build_quad_objective_expr(spec, w, mu, sigma)

    objective = _assemble_objective(base_expr, penalties)
    problem = cp.Problem(objective, hard_constraints)

    return CompiledProblem(
        problem=problem,
        weights=w,
        constraint_objs=constraint_objs,
        spec=spec,
        extra_vars=extra_vars,
    )
