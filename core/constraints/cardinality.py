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

Production compilation requires long-only, fully invested weights. Diagnostic
deletion trials may temporarily remove one of those domain rows; when that
happens the diagnoser derives finite per-name lower and upper bounds from the
remaining continuous system and supplies a two-sided selection link. It never
uses an unjustified ``M=1`` infeasibility result as proof.

This node contributes no usable MIP dual of its own (integer programs have no
shadow prices — see :mod:`core.duals` and the fix-and-resolve path). Its builder
returns the count cap as the named constraint so diagnosis can elasticize it,
pushes the linking/floor rows onto ``aux_constraints``, and stores the selection
vector ``y`` in ``extra_vars`` so the solve layer can read ``y*``.

Only a max-only Cardinality instance is elastic in Sprint 5.  When
``min_names`` or ``min_position`` is present, the node contains lower rows that
the scalar max-cap slack does not relax.  Such an instance is therefore
structural during diagnosis rather than being mislabeled as repairable.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import cvxpy as cp
import numpy as np
from pydantic import Field, model_validator

from core.compile_context import BuildContext
from core.irbase import ProblemClassImpact, _ConstraintIRModel, _new_id


class Cardinality(_ConstraintIRModel):
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
    elastic_default: ClassVar[bool] = True

    @property
    def elasticity_supported(self) -> bool:
        """True only when relaxing ``max_names`` relaxes the whole node."""

        return self.min_names is None and self.min_position is None

    @property
    def slack_scale(self) -> float:
        return float(self.max_names)

    def diagnostic_big_m(self, universe_size: int | None = None) -> float:
        """A cardinality cap is vacuous at the full universe size."""
        if universe_size is None or universe_size < 1:
            raise ValueError("Cardinality diagnostic_big_m requires a positive universe_size.")
        return float(universe_size)

    @model_validator(mode="after")
    def _check_counts(self) -> Cardinality:
        if self.min_names is not None and self.min_names > self.max_names:
            raise ValueError(
                f"Cardinality min_names={self.min_names} > max_names={self.max_names}."
            )
        if self.elastic is True and not self.elasticity_supported:
            raise ValueError(
                "Cardinality is elastic only in its max_names-only form; "
                "min_names/min_position rows are structural during diagnosis."
            )
        return self


def build(node: Cardinality, ctx: BuildContext) -> cp.Constraint:
    """Add the binary-selection cardinality constraints to ``ctx``.

    The max-name row is returned as the node's named constraint so the
    diagnostic compiler can replace it with ``sum(y) <= K + slack``. It still
    has no MIP dual; ordinary MIP reporting uses fix-and-resolve and drops the
    whole Cardinality node before harvesting continuous duals.
    """
    n = ctx.n
    # Per-asset big-M = the position upper bound (≤ 1 in production).
    if ctx.weight_upper is not None:
        big_m = np.asarray(ctx.weight_upper, dtype=float)
    else:
        big_m = np.ones(n, dtype=float)
    lower_m = (
        np.asarray(ctx.weight_lower, dtype=float)
        if ctx.weight_lower is not None
        else np.zeros(n, dtype=float)
    )

    y = cp.Variable(n, boolean=True, name="y")
    ctx.extra_vars["y"] = y

    # Two-sided links pin an unselected name to zero. Production lower bounds
    # are zero; diagnostic counterfactuals can carry rigorously derived signed
    # bounds after LongOnly or Budget is removed.
    ctx.aux_constraints.append(ctx.w <= cp.multiply(big_m, y))
    ctx.aux_constraints.append(ctx.w >= cp.multiply(lower_m, y))
    # The defining count cap is the one elastic component for this sprint.
    count_cap = cp.sum(y) <= node.max_names
    if node.min_names is not None:
        ctx.aux_constraints.append(cp.sum(y) >= node.min_names)
    if node.min_position is not None:
        # A held name carries at least min_position; an unheld one stays 0.
        ctx.aux_constraints.append(ctx.w >= node.min_position * y)
    return count_cap
