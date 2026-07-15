"""TurnoverCap: bound one-way trading volume ``‖w − w_prev‖₁ ≤ max_turnover``.

Depends on the Slice 1 ``w_prev`` plumbing: with the default zero ``w_prev``
("fresh from cash") turnover equals ``‖w‖₁``, i.e. the full deployment. The L1
norm is linearized via the shared :func:`core.compile_context.abs_deviation`
epigraph, keeping the problem convex.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import cvxpy as cp
from pydantic import Field

from core.compile_context import BuildContext, abs_deviation
from core.irbase import ProblemClassImpact, _ConstraintIRModel, _new_id


class TurnoverCap(_ConstraintIRModel):
    """Hard cap on total turnover relative to the pre-trade weights."""

    kind: Literal["turnover_cap"] = "turnover_cap"
    id: str = Field(default_factory=lambda: _new_id("turnover"))
    max_turnover: float = Field(
        gt=0.0, description="Upper bound on ‖w − w_prev‖₁ (sum of absolute weight changes)."
    )
    problem_class_impact: ClassVar[ProblemClassImpact] = "convex"
    elastic_default: ClassVar[bool] = True
    big_m: ClassVar[float | None] = 2.0

    @property
    def slack_scale(self) -> float:
        return self.max_turnover


def build(node: TurnoverCap, ctx: BuildContext) -> cp.Constraint:
    u, epigraph = abs_deviation(ctx.w, ctx.w_prev, name_suffix=f"_turn_{node.id}")
    ctx.aux_constraints.extend(epigraph)
    return cp.sum(u) <= node.max_turnover
