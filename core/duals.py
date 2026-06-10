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
            # MIPs and a few solver paths don't return duals; in Sprint 1 the
            # only objective forms are convex so this should be unreachable
            # — but if it ever fires, the caller deserves a real exception
            # rather than a silent zero.
            raise DualsUnavailableError(
                f"Constraint {cid!r} has no dual value attached. "
                "This typically means the problem was solved by a method that does not "
                "produce duals (e.g. a MIP path)."
            )
        arr = np.atleast_1d(np.asarray(dual, dtype=float))
        # Use the max-magnitude entry as a one-number summary. For a Box,
        # this is the most binding of {lower-side, upper-side} across assets.
        out[cid] = float(np.max(np.abs(arr)))
    return out
