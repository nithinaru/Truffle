"""Solver routing — pick the backend from the compiled problem's character.

Sprint 1–3 only ever produced continuous convex programs, so the solver was a
hard-coded ``Clarabel``. Sprint 4 introduces mixed-integer constraints
(``Cardinality``), which Clarabel cannot touch. This module owns the *single*
decision of which solver a spec routes to, so the deterministic ``solve`` CLI
and the chat loop never diverge.

Routing table
-------------
====================  =========================================  =========  ==========  ==================
``problem_class``     objective form                              solver     cvxpy enum  ``problem_form``
====================  =========================================  =========  ==========  ==================
``convex``            any convex objective                        Clarabel   CLARABEL    continuous-convex
``mip``               linear  (``min_cvar`` → Rockafellar LP)      HiGHS      HIGHS       MILP
``mip``               quadratic (``min_variance``/``mean_variance``)  SCIP   SCIP        MIQP
``mip``               any supported objective + TE-cap SOC         SCIP   SCIP        MISOCP
====================  =========================================  =========  ==========  ==================

Why three solvers and not a fallback: a mixed-integer *quadratic* program
(cardinality + variance objective) is not something HiGHS solves, and a
mixed-integer *linear* program (cardinality + CVaR) is HiGHS's home turf but
not Clarabel's. A tracking-error cap adds a second-order cone, so combining it
with cardinality is a MISOCP and must also route to SCIP even when the objective
itself is linear. Silently routing one of these forms to the wrong backend
would either error deep in the solver or, worse, return a wrong answer, so an
unavailable solver raises a clear, actionable error naming the missing solver
and how to install it — never a silent fallback.

The two change-of-variable objectives (``max_sharpe``, ``risk_parity``) and
``min_tracking_error`` cannot currently be combined with a mixed-integer
constraint; routing such a spec raises :class:`~core.exceptions.CompilationError`
rather than guessing a backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp

from core.exceptions import CompilationError, SolverError
from core.ir import (
    MeanVariance,
    MinCVaR,
    MinVariance,
    Objective,
    PortfolioSpec,
    TrackingErrorCap,
)
from core.irbase import ProblemClassImpact

# Objectives whose epigraph is linear once compiled (the CVaR LP). Cardinality
# on top of these yields a MILP.
_LINEAR_MIP_OBJECTIVES: tuple[type, ...] = (MinCVaR,)
# Objectives with a quadratic term. Cardinality on top yields a MIQP.
_QUADRATIC_MIP_OBJECTIVES: tuple[type, ...] = (MinVariance, MeanVariance)

# How to install each backend, surfaced verbatim when a route is unavailable.
_INSTALL_HINTS: dict[str, str] = {
    cp.CLARABEL: (
        "Clarabel is missing — it ships with cvxpy by default. Reinstall with "
        "`pip install -U clarabel cvxpy`."
    ),
    cp.HIGHS: (
        "HiGHS (the MILP backend) is missing. It ships with recent cvxpy; "
        "reinstall with `pip install -U highspy cvxpy`."
    ),
    cp.SCIP: (
        "SCIP (the MIQP/MISOCP backend) is not installed. A cardinality limit "
        "combined with a variance objective or tracking-error cone needs SCIP. "
        "Install it with `pip install pyscipopt` "
        "(Apache-2.0, free; the wheel bundles the SCIP binaries, so no separate "
        "system install is normally required — see the README MIP note)."
    ),
}


@dataclass(frozen=True)
class SolverChoice:
    """The resolved backend for one spec.

    Attributes:
        cp_solver: The cvxpy solver enum string (e.g. ``cp.CLARABEL``) to pass
            as ``problem.solve(solver=...)``.
        name: Human-readable solver name for the report/echo
            (``"Clarabel"`` / ``"HiGHS"`` / ``"SCIP"``).
        problem_form: ``"continuous-convex"`` / ``"MILP"`` / ``"MIQP"`` /
            ``"MISOCP"`` — the label the chat loop uses to warn the user.
    """

    cp_solver: str
    name: str
    problem_form: str

    @property
    def is_mip(self) -> bool:
        """True for the mixed-integer forms (MILP / MIQP / MISOCP)."""
        return self.problem_form in {"MILP", "MIQP", "MISOCP"}


def route_for(
    problem_class: ProblemClassImpact,
    objective: Objective,
    *,
    has_tracking_error_cap: bool = False,
) -> SolverChoice:
    """Pure routing decision from problem class, objective, and cone presence.

    Separated from :func:`select_solver` so the routing table itself is
    unit-testable without constructing a full spec or checking solver
    availability. Does *not* verify the chosen solver is installed — that is
    :func:`select_solver`'s job.
    """
    if problem_class == "convex":
        return SolverChoice(cp.CLARABEL, "Clarabel", "continuous-convex")
    # problem_class == "mip"
    if isinstance(objective, _LINEAR_MIP_OBJECTIVES):
        if has_tracking_error_cap:
            return SolverChoice(cp.SCIP, "SCIP", "MISOCP")
        return SolverChoice(cp.HIGHS, "HiGHS", "MILP")
    if isinstance(objective, _QUADRATIC_MIP_OBJECTIVES):
        if has_tracking_error_cap:
            return SolverChoice(cp.SCIP, "SCIP", "MISOCP")
        return SolverChoice(cp.SCIP, "SCIP", "MIQP")
    raise CompilationError(
        f"Objective {type(objective).__name__!r} cannot be combined with a "
        "mixed-integer constraint (e.g. Cardinality) in this version. "
        "Mixed-integer routing supports min_variance / mean_variance (MIQP) "
        "and min_cvar (MILP). Use one of those objectives, or drop the "
        "cardinality limit to keep the problem continuous."
    )


def select_solver(spec: PortfolioSpec) -> SolverChoice:
    """Select (and verify availability of) the solver for ``spec``.

    This is the single entrypoint both the CLI and chat loop call. It reads the
    spec's aggregated ``problem_class`` and objective kind, consults the routing
    table (:func:`route_for`), then checks the chosen solver is actually
    installed — raising :class:`~core.exceptions.SolverError` with an actionable
    install hint if not, rather than letting cvxpy fail obscurely or falling
    back to a wrong backend.

    Raises:
        CompilationError: if the objective cannot be routed to a MIP backend.
        SolverError: if the routed solver is not installed.
    """
    choice = route_for(
        spec.problem_class,
        spec.objective,
        has_tracking_error_cap=any(
            isinstance(constraint, TrackingErrorCap) for constraint in spec.constraints
        ),
    )
    if choice.cp_solver not in cp.installed_solvers():
        raise SolverError(_INSTALL_HINTS[choice.cp_solver])
    return choice
