"""Structured solver output ready to be narrated.

A ``SolutionReport`` is the *only* surface the explanation layer is allowed
to read from. Every number in the LLM's narration must trace back to a
field on this object (or one of the documented renderings, see
:mod:`agent.grounding`). Building it here in deterministic Python keeps
the trust boundary clean.

Shadow-price sign convention (BLUEPRINT §5 — "duals everywhere"):

* For an *inequality* constraint of the form ``g(w) >= 0`` (Truffle's
  encoding: long-only, the stacked-slack form of Box), the dual ``λ`` is
  non-negative; ``λ > 0`` indicates the constraint is binding and tells
  you how much the objective would improve per unit relaxation of the
  binding side. We report the max-magnitude entry per constraint
  (see :func:`core.duals.harvest_duals`).
* For an *equality* constraint (Budget), the dual is signed; the magnitude
  is the shadow price, the sign indicates which direction relaxes the
  optimum. The narration reports magnitude only (and so MUST NOT claim a
  sign that the report does not contain).

Threshold for "binding" matches the duals module: a magnitude of
``1e-6`` or less is treated as non-binding to avoid surfacing
floating-point noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BINDING_THRESHOLD = 1e-6


@dataclass(frozen=True)
class BindingConstraint:
    """One row of the binding-constraints section of the report."""

    constraint_id: str
    human_name: str
    shadow_price: float


@dataclass(frozen=True)
class SolutionReport:
    """Everything the narration layer is allowed to reference.

    Attributes:
        weights: ``{ticker -> weight}`` in canonical universe order.
        objective_kind: The IR objective ``kind`` discriminator
            (``"min_variance"`` / ``"mean_variance"`` / ``"min_cvar"``).
        objective_value: The solved objective value (variance, MV penalty,
            or CVaR depending on the objective).
        var: Optimal ``t`` for ``min_cvar``; ``None`` otherwise. Reported
            so the explanation can say "VaR(α=0.95) = X" without re-running
            the solver.
        solver: Solver name (e.g. ``"Clarabel"``).
        solve_time_ms: Wall-clock solve time in milliseconds.
        status: CVXPY status string (``"optimal"`` /
            ``"optimal_inaccurate"`` after a successful run).
        binding: Constraints whose absolute shadow price exceeds
            :data:`BINDING_THRESHOLD`, sorted by descending magnitude. The
            grounder will not allow the narration to reference any number
            not in this list (or in the other report fields).
        n_assets: Universe size; useful for sanity-checking "k of n names".
        nonzero_names: Tickers with weight magnitude above a small floor —
            so the narration may say "13 names selected" without that
            number being a hallucination.
        duals_conditional: ``True`` when the shadow prices come from the MIP
            fix-and-resolve restriction (conditional on the selected name set)
            rather than an ordinary convex solve. The narration MUST state the
            conditionality when this is set (see ``explain_system.md``).
        selected_names: For a mixed-integer (cardinality) solve, the names the
            integer program selected. ``None`` on the continuous path.
        optimality_gap: For a mixed-integer solve, the solver's proven
            optimality gap (~0 at ``optimal``; nonzero only if a time limit cut
            the search short). ``None`` on the continuous path.
    """

    weights: dict[str, float]
    objective_kind: str
    objective_value: float
    solver: str
    solve_time_ms: float
    status: str
    n_assets: int
    nonzero_names: int
    var: float | None = None
    binding: list[BindingConstraint] = field(default_factory=list)
    duals_conditional: bool = False
    selected_names: list[str] | None = None
    optimality_gap: float | None = None


def build_report(
    *,
    weights: dict[str, float],
    objective_kind: str,
    objective_value: float,
    solver: str,
    solve_time_ms: float,
    status: str,
    duals: dict[str, float],
    constraint_human_names: dict[str, str],
    var: float | None = None,
    nonzero_floor: float = 1e-4,
    duals_conditional: bool = False,
    selected_names: list[str] | None = None,
    optimality_gap: float | None = None,
) -> SolutionReport:
    """Assemble a ``SolutionReport`` from solver outputs and the IR id map.

    Args:
        weights: Final weights keyed by ticker.
        objective_kind: One of ``"min_variance"`` / ``"mean_variance"`` /
            ``"min_cvar"``.
        objective_value: Solver objective value.
        solver: e.g. ``"Clarabel"``.
        solve_time_ms: Wall-clock solve time in milliseconds.
        status: CVXPY status string.
        duals: Output of :func:`core.duals.harvest_duals`, mapping IR
            constraint id to a scalar shadow-price magnitude.
        constraint_human_names: ``{ir_id -> human-readable phrase}``,
            e.g. ``{"cap_aaa": "the AAA position cap"}``. Used for
            narration. If an id is missing here, the id itself is used.
        var: Optimal ``t`` for ``min_cvar``, else ``None``.
        nonzero_floor: Weights below this magnitude are treated as zero
            for the ``nonzero_names`` count.
    """
    binding = [
        BindingConstraint(
            constraint_id=cid,
            human_name=constraint_human_names.get(cid, cid),
            shadow_price=val,
        )
        for cid, val in duals.items()
        if abs(val) > BINDING_THRESHOLD
    ]
    binding.sort(key=lambda b: -abs(b.shadow_price))
    nonzero_names = sum(1 for w in weights.values() if abs(w) > nonzero_floor)
    return SolutionReport(
        weights=dict(weights),
        objective_kind=objective_kind,
        objective_value=float(objective_value),
        var=None if var is None else float(var),
        solver=solver,
        solve_time_ms=float(solve_time_ms),
        status=status,
        binding=binding,
        n_assets=len(weights),
        nonzero_names=nonzero_names,
        duals_conditional=duals_conditional,
        selected_names=selected_names,
        optimality_gap=optimality_gap,
    )
