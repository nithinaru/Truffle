"""Cardinality: cap (and optionally floor) the number of held names.

This is the first node that forces a **mixed-integer** problem
(``problem_class_impact = "mip"``). For a long-only weight vector bounded in
``[0, 1]`` it introduces one binary selection variable per asset and a big-M
linking constraint:

    y_i ∈ {0, 1}                       # 1 ⇔ asset i is held
    w_i ≤ M_i · y_i                    # not selected ⇒ weight pinned to 0
    Σ_i y_i ≤ max_names                # at most max_names held
    Σ_i y_i ≥ min_names                # (optional) at least min_names held
    w_i ≥ min_position · y_i           # (optional) no dust positions

**Big-M choice.** With long-only (``w_i ≥ 0``) and weights bounded above by a
position cap, ``M_i`` is exactly that per-asset upper bound. Truffle derives
``M_i`` from the spec's ``Box`` upper bounds (universe-wide and per-ticker),
defaulting to ``1.0`` when no tighter cap exists — valid because a fully
invested long-only book has every ``w_i ≤ 1``. A tighter M is not just cosmetic:
it shrinks the LP relaxation gap and speeds the branch-and-bound search.

The formulation assumes long-only weights; the linking constraint only pins the
*upper* side (``w_i ≤ M_i y_i``), which together with ``w_i ≥ 0`` forces an
unselected weight to zero. A spec that allowed shorting would also need a lower
big-M; that is out of scope here (cardinality pairs with long-only books).

This node contributes no usable dual of its own (integer programs have no
shadow prices — see :mod:`core.duals` and the fix-and-resolve path), so its
builder returns ``None`` and pushes its hard constraints onto the context's
``aux_constraints``, storing the selection vector ``y`` in ``extra_vars`` so the
solve layer can read the optimal selection ``y*``.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import cvxpy as cp
import numpy as np
from pydantic import Field, model_validator

from core.compile_context import BuildContext
from core.irbase import ProblemClassImpact, _IRModel, _new_id


class Cardinality(_IRModel):
    """At most ``max_names`` (optionally at least ``min_names``) held positions.

    Attributes:
        max_names: Upper bound on the count of nonzero positions. Must be ≥ 1
            and ≤ the universe size (the latter is checked at the spec level,
            where the universe is known).
        min_names: Optional lower bound on the count of held positions. Must be
            ≥ 1 and ≤ ``max_names``.
        min_position: Optional floor on a *held* name's weight ("no dust"): a
            selected asset gets at least this weight. Must be in ``(0, 1]`` and
            not exceed the position caps (checked at the spec level).
    """

    kind: Literal["cardinality"] = "cardinality"
    id: str = Field(default_factory=lambda: _new_id("cardinality"))
    max_names: int = Field(ge=1, description="At most this many nonzero positions.")
    min_names: int | None = Field(
        default=None, ge=1, description="Optional floor on the number of held positions."
    )
    min_position: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Optional minimum weight for a held name (no dust positions).",
    )
    problem_class_impact: ClassVar[ProblemClassImpact] = "mip"

    @model_validator(mode="after")
    def _check_counts(self) -> Cardinality:
        if self.min_names is not None and self.min_names > self.max_names:
            raise ValueError(
                f"Cardinality min_names={self.min_names} > max_names={self.max_names}."
            )
        return self


def build(node: Cardinality, ctx: BuildContext) -> None:
    """Add the binary-selection cardinality constraints to ``ctx``.

    Returns ``None`` (no named dual-carrying constraint): a MIP has no duals, so
    the cardinality node never appears in the ``{id -> Constraint}`` map. The
    hard constraints go onto ``ctx.aux_constraints`` and the selection vector is
    exposed as ``ctx.extra_vars["y"]`` for the fix-and-resolve dual path.
    """
    n = ctx.n
    # Per-asset big-M = the position upper bound (≤ 1). Falls back to 1.0.
    if ctx.weight_upper is not None:
        big_m = np.asarray(ctx.weight_upper, dtype=float)
    else:
        big_m = np.ones(n, dtype=float)

    y = cp.Variable(n, boolean=True, name="y")
    ctx.extra_vars["y"] = y

    # Not-selected ⇒ weight pinned to 0 (with long-only w ≥ 0). Selected ⇒
    # weight may reach its cap.
    ctx.aux_constraints.append(ctx.w <= cp.multiply(big_m, y))
    # The defining count cap.
    ctx.aux_constraints.append(cp.sum(y) <= node.max_names)
    if node.min_names is not None:
        ctx.aux_constraints.append(cp.sum(y) >= node.min_names)
    if node.min_position is not None:
        # A held name carries at least min_position; an unheld one stays 0.
        ctx.aux_constraints.append(ctx.w >= node.min_position * y)
    return None
