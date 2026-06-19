"""Dual-value harvesting.

After a successful convex solve, every named constraint has a *dual variable*
— the Lagrange multiplier whose magnitude is the shadow price of that
constraint. This module lifts those numbers back into IR id space so the
explanation layer (BLUEPRINT §6) can say *"your 30% tech cap is costing
~11 bps of expected return"* and stay grounded in real solver output.

We summarize each constraint's (possibly vector) dual to a single scalar
suitable for ranking by impact:

* Equalities (Budget): the dual is signed; the *absolute value* is the
  shadow price magnitude. Sign is preserved separately if a future caller
  needs it.
* Inequalities (LongOnly, Box): KKT requires duals ≥ 0; we report the
  maximum component, which is the largest binding bps cost across the
  vector entries (e.g. across assets for a universe-wide Box).

Sign convention (carried over from Sprint 1, unchanged): for Truffle's
inequality encoding ``g(w) ≥ 0`` (long-only, the stacked-slack Box/GroupCap),
a **positive** shadow price means the constraint is binding and the objective
would *improve* (decrease, since everything is a minimization) by that many
units per unit of relaxation of the binding side. ~0 means slack/non-binding.

Mixed-integer problems and conditional duals
--------------------------------------------
An integer program has **no meaningful dual variables**: the feasible set is
not convex, so there is no Lagrangian whose multipliers price the constraints,
and cvxpy returns ``dual_value is None`` on a MIP. Truffle therefore does *not*
harvest duals from the MIP itself. Instead :func:`core.solve._fix_and_resolve`
fixes the binaries at the optimal selection ``y*`` (equivalently restricts the
universe to the selected names and drops integrality), re-solves the resulting
**continuous** problem, and harvests duals from *that*. :func:`harvest_duals`
runs only on such continuous problems and is unchanged.

Those numbers are **conditional shadow prices**: they price each constraint
*given the selected name set held fixed*, not globally — relaxing a cap might
also change which names should be selected, which a conditional dual cannot see.
The sign convention above carries over verbatim; the report flags them with
``duals_conditional=True`` so the narration states the conditionality.
"""

from __future__ import annotations

import numpy as np

from core.compiler import CompiledProblem
from core.exceptions import DualsUnavailableError, InfeasibleError, SolverError, UnboundedError

_OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}


def _check_solved(compiled: CompiledProblem) -> None:
    status = compiled.problem.status
    if status is None:
        raise DualsUnavailableError(
            "Problem has not been solved yet — call problem.solve(...) before harvest_duals."
        )
    if status in {"infeasible", "infeasible_inaccurate"}:
        raise InfeasibleError(
            f"Problem is infeasible (solver status: {status!r}). "
            "Sprint 3 will run an elastic relaxation to identify the conflicting set."
        )
    if status in {"unbounded", "unbounded_inaccurate"}:
        raise UnboundedError(
            f"Problem is unbounded (solver status: {status!r}). "
            "Check that you have a Budget constraint and finite expected returns."
        )
    if status not in _OPTIMAL_STATUSES:
        raise SolverError(f"Solver returned non-optimal status: {status!r}.")


def harvest_duals(compiled: CompiledProblem) -> dict[str, float]:
    """Return ``{ir_constraint_id -> shadow_price}`` after a successful solve.

    Args:
        compiled: A ``CompiledProblem`` whose ``problem.solve(...)`` has been
            called (status ``"optimal"`` or ``"optimal_inaccurate"``).

    Returns:
        Mapping from IR constraint id to a scalar shadow-price magnitude.
        Magnitudes ≈ 0 indicate non-binding constraints; large magnitudes
        indicate constraints that are actively shaping the optimum.

    Raises:
        DualsUnavailableError: if the problem has not been solved.
        InfeasibleError / UnboundedError / SolverError: with structured
            messages suitable for surfacing to the agent and the user.
    """
    _check_solved(compiled)

    out: dict[str, float] = {}
    for cid, c in compiled.constraint_objs.items():
        dual = c.dual_value
        if dual is None:
            # harvest_duals only ever runs on continuous solves (the convex path,
            # or the fix-and-resolve restriction). A None dual here means a
            # continuous solve genuinely failed to attach one — a real error, not
            # the expected "MIP has no duals" case, which is handled upstream by
            # routing MIPs through core.solve._fix_and_resolve instead of here.
            raise DualsUnavailableError(
                f"Constraint {cid!r} has no dual value attached after a continuous "
                "solve. MIPs are handled by fix-and-resolve (see core.solve); this "
                "indicates a solver path that did not produce duals."
            )
        arr = np.atleast_1d(np.asarray(dual, dtype=float))
        # Use the max-magnitude entry as a one-number summary. For a Box,
        # this is the most binding of {lower-side, upper-side} across assets.
        out[cid] = float(np.max(np.abs(arr)))
    return out
