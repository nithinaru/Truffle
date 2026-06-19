"""GroupCap: bound the summed weight of a named group (e.g. a sector).

``Σ_{i ∈ group} w_i ≤ max_weight`` and optionally ``≥ min_weight``. Requires a
ticker→group mapping at compile time (the sector CSV already plumbed into the
chat session). Linear in ``w`` ⇒ convex.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import cvxpy as cp
from pydantic import Field, model_validator

from core.compile_context import BuildContext
from core.exceptions import CompilationError
from core.irbase import ProblemClassImpact, _IRModel, _new_id


class GroupCap(_IRModel):
    """Cap (and optionally floor) the total weight of one group."""

    kind: Literal["group_cap"] = "group_cap"
    id: str = Field(default_factory=lambda: _new_id("groupcap"))
    group: str = Field(min_length=1, description="Group label, matched against the mapping values.")
    max_weight: float = Field(gt=0.0, le=1.0, description="Upper bound on Σ in-group weight.")
    min_weight: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Optional lower bound on Σ in-group weight."
    )
    problem_class_impact: ClassVar[ProblemClassImpact] = "convex"

    @model_validator(mode="after")
    def _check_bounds(self) -> GroupCap:
        if self.min_weight is not None and self.min_weight > self.max_weight:
            raise ValueError(
                f"GroupCap min_weight={self.min_weight} > max_weight={self.max_weight}."
            )
        return self


def build(node: GroupCap, ctx: BuildContext) -> cp.Constraint:
    if ctx.group_map is None:
        raise CompilationError(
            f"GroupCap {node.id} needs a ticker->group mapping; none was supplied "
            "(pass --sectors on the CLI or group_map to compile_spec)."
        )
    idx = [i for t, i in ctx.ticker_index.items() if ctx.group_map.get(t) == node.group]
    if not idx:
        raise CompilationError(
            f"GroupCap {node.id} references group {node.group!r}, which matches no "
            "ticker in the universe under the supplied mapping."
        )
    in_group = cp.sum(ctx.w[idx])
    if node.min_weight is None:
        return in_group <= node.max_weight
    # Stack both sides into one named constraint so the node carries a single id
    # (dual vector length 2: [upper-side, lower-side]).
    return cp.hstack([node.max_weight - in_group, in_group - node.min_weight]) >= 0
