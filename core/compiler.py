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

from collections.abc import Callable

import cvxpy as cp
import numpy as np

from core.compile_context import (
    BuildContext,
    CompiledProblem,
    build_cvar_block,
    resolve_w_prev,
    validate_inputs,
    validate_scenarios,
)
from core.constraints import (
    cvar_limit,
    factor_exposure,
    group_cap,
    tracking_error_cap,
    transaction_cost,
    turnover_cap,
)
from core.exceptions import CompilationError
from core.ir import (
    Box,
    Budget,
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
from core.objectives import max_sharpe, min_tracking_error, risk_parity

# build_cvar_block and resolve_w_prev are re-exported (imported above) so callers
# and tests can keep importing them from core.compiler. They now live in the leaf
# core.compile_context module to avoid an import cycle with the constraint nodes.
__all__ = ["CompiledProblem", "build_cvar_block", "compile_spec", "resolve_w_prev"]

# Dispatch table for Sprint 3 convex constraints. Budget/LongOnly/Box keep their
# inline builders in _build_constraint (Sprint 1). Each entry maps the IR node
# type to a module-level build(node, ctx) function.
_CONSTRAINT_BUILDERS: dict[type, Callable[..., cp.Constraint | None]] = {
    GroupCap: group_cap.build,
    TurnoverCap: turnover_cap.build,
    TransactionCost: transaction_cost.build,
    CVaRLimit: cvar_limit.build,
    TrackingErrorCap: tracking_error_cap.build,
    FactorExposure: factor_exposure.build,
}


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
    sectors: dict[str, str] | None = None,
    benchmark_weights: dict[str, np.ndarray] | None = None,
    factor_loadings: dict[str, np.ndarray] | None = None,
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
        sectors: ``{ticker -> group}`` mapping required by ``GroupCap``.
        benchmark_weights: ``{name -> weight vector}`` aligned to the universe,
            required by ``TrackingErrorCap`` (and ``MinTrackingError``).
        factor_loadings: ``{name -> loading vector}`` aligned to the universe,
            required by ``FactorExposure``.

    Returns:
        ``CompiledProblem`` wrapping the unsolved CVXPY problem, the weight
        variable, the IR-id → cvxpy.Constraint map, and any objective-specific
        auxiliary variables (``t`` and ``z`` for CVaR).

    Raises:
        CompilationError: if shapes mismatch, Σ is not symmetric, scenarios
            are missing/malformed for CVaR, or the IR contains an
            objective/constraint kind the compiler does not understand.
    """
    validate_inputs(spec, mu, sigma)

    # Change-of-variable objectives build their own transformed problem (they do
    # not optimize over w directly) and return early with a weight-recovery hook.
    if isinstance(spec.objective, MaxSharpe):
        return max_sharpe.build(spec.objective, spec, mu, sigma)
    if isinstance(spec.objective, RiskParity):
        return risk_parity.build(spec.objective, spec, mu, sigma)

    n = len(spec.universe)
    w = cp.Variable(n, name="w")
    ticker_index = {t: i for i, t in enumerate(spec.universe)}
    # Resolved once so every constraint builder sees the same pre-trade vector.
    w_prev_vec = resolve_w_prev(w_prev, n)

    ctx = BuildContext(
        w=w,
        n=n,
        ticker_index=ticker_index,
        w_prev=w_prev_vec,
        sigma=sigma,
        scenarios=scenarios,
        group_map=sectors,
        benchmark_weights=benchmark_weights,
        factor_loadings=factor_loadings,
    )

    constraint_objs: dict[str, cp.Constraint] = {}
    hard_constraints: list[cp.Constraint] = []
    for c in spec.constraints:
        if isinstance(c, Budget | LongOnly | Box):
            cons = _build_constraint(c, w, ticker_index, w_prev_vec)
        else:
            builder = _CONSTRAINT_BUILDERS.get(type(c))
            if builder is None:
                raise CompilationError(f"No compiler builder for constraint {type(c).__name__}.")
            cons = builder(c, ctx)
        # Penalty-only nodes (TransactionCost) return None: no hard constraint,
        # no dual, so they are intentionally absent from constraint_objs.
        if cons is not None:
            constraint_objs[c.id] = cons
            hard_constraints.append(cons)

    # Penalty terms (TransactionCost) and unnamed auxiliary constraints (L1
    # epigraphs, CVaR linking rows) the builders accumulated on the context.
    penalties: list[cp.Expression] = list(ctx.penalties)
    hard_constraints.extend(ctx.aux_constraints)
    extra_vars: dict[str, cp.Variable] = dict(ctx.extra_vars)

    obj = spec.objective
    if isinstance(obj, MinCVaR):
        scenarios = validate_scenarios(scenarios, n)
        base_expr, t_var, z_var, aux = build_cvar_block(scenarios, w, obj.cvar_alpha)
        hard_constraints = hard_constraints + aux
        extra_vars["t"] = t_var
        extra_vars["z"] = z_var
    elif isinstance(obj, MinTrackingError):
        base_expr = min_tracking_error.build(obj, ctx)
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
