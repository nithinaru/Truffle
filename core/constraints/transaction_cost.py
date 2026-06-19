"""TransactionCost: proportional trading cost as an objective *penalty*.

This is the one node that modifies the objective and adds **no hard
constraint**. It contributes ``(bps / 1e4) · ‖w − w_prev‖₁`` to the objective
via the Slice 0 penalty-accumulation path, so trading away from ``w_prev`` is
discouraged in proportion to the stated cost. Because it adds no constraint, it
is intentionally absent from the ``{id -> Constraint}`` dual map — there is no
shadow price for a penalty. ``build`` returns ``None`` to signal this.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import cvxpy as cp
from pydantic import Field

from core.compile_context import BuildContext, abs_deviation
from core.irbase import ProblemClassImpact, _IRModel, _new_id


class TransactionCost(_IRModel):
    """Proportional cost in basis points applied to ``‖w − w_prev‖₁``."""

    kind: Literal["transaction_cost"] = "transaction_cost"
    id: str = Field(default_factory=lambda: _new_id("txcost"))
    bps: float = Field(
        ge=0.0, description="Proportional cost in basis points (1 bp = 0.01%) per unit traded."
    )
    problem_class_impact: ClassVar[ProblemClassImpact] = "convex"


def build(node: TransactionCost, ctx: BuildContext) -> None:
    """Append the cost penalty to the objective; add no hard constraint.

    Returns ``None`` so the compiler does not register a (non-existent) dual.
    """
    u, epigraph = abs_deviation(ctx.w, ctx.w_prev, name_suffix=f"_tx_{node.id}")
    ctx.aux_constraints.extend(epigraph)
    rate = node.bps / 1e4
    ctx.penalties.append(rate * cp.sum(u))
    return None
