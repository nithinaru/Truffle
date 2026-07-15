"""Deprecated scalar dual adapter.

New solve reports use :mod:`core.sensitivity`, which preserves every row,
ticker/label, side, sign, transform scale, primal slack, conditionality, and
unit. ``harvest_duals`` remains only for callers on the pre-2.0 API and reduces
each vector to a max absolute solver multiplier. That reduction is unsuitable
for narration or financial interpretation and is no longer used by
:func:`core.solve.solve_spec`.
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
    """Return the legacy ``{constraint_id: max_abs_dual}`` mapping.

    Args:
        compiled: A ``CompiledProblem`` whose ``problem.solve(...)`` has been
            called (status ``"optimal"`` or ``"optimal_inaccurate"``).

    Returns:
        Compatibility-only scalar magnitudes. Use
        :func:`core.sensitivity.harvest_sensitivities` for semantics.

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
