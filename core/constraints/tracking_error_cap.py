"""TrackingErrorCap: bound tracking error vs a benchmark.

Tracking-error variance is ``(w − b)ᵀ Σ (w − b)``; the cap on tracking error
``‖chol(Σ)·(w − b)‖₂ ≤ max_te`` is equivalently expressed as the convex
quadratic ``(w − b)ᵀ Σ (w − b) ≤ max_te²``. Stated this way Clarabel handles it
as a second-order-cone / quadratic constraint. Requires benchmark weights as
data (aligned to the universe).
"""

from __future__ import annotations

from typing import ClassVar, Literal

import cvxpy as cp
from pydantic import Field

from core.compile_context import BuildContext
from core.irbase import ProblemClassImpact, _IRModel, _new_id


class TrackingErrorCap(_IRModel):
    """Upper bound on tracking error relative to a named benchmark."""

    kind: Literal["tracking_error_cap"] = "tracking_error_cap"
    id: str = Field(default_factory=lambda: _new_id("tecap"))
    benchmark: str = Field(min_length=1, description="Benchmark name; weights supplied as data.")
    max_te: float = Field(gt=0.0, description="Upper bound on tracking error (annualized vol units).")
    problem_class_impact: ClassVar[ProblemClassImpact] = "convex"


def build(node: TrackingErrorCap, ctx: BuildContext) -> cp.Constraint:
    b = ctx.aligned_benchmark(node.benchmark)
    active = ctx.w - b
    te_var = cp.quad_form(active, cp.psd_wrap(ctx.sigma))
    return te_var <= node.max_te**2
