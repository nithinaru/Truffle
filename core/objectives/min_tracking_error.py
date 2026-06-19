"""MinTrackingError objective: minimize ``(w − b)ᵀ Σ (w − b)``.

A straight QP in ``w`` against a named benchmark ``b`` (supplied as data). It
behaves like any other base objective — it composes with every Sprint-1/3
constraint and with the penalty-accumulation path — so it is built as a base
expression over the existing weight variable, not as a transformed problem.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import cvxpy as cp
from pydantic import Field

from core.compile_context import BuildContext
from core.irbase import ProblemClassImpact, _IRModel


class MinTrackingError(_IRModel):
    """Minimize tracking-error variance vs a named benchmark."""

    kind: Literal["min_tracking_error"] = "min_tracking_error"
    benchmark: str = Field(min_length=1, description="Benchmark name; weights supplied as data.")
    problem_class_impact: ClassVar[ProblemClassImpact] = "convex"


def build(node: MinTrackingError, ctx: BuildContext) -> cp.Expression:
    """Return the base objective expression ``(w − b)ᵀ Σ (w − b)``."""
    b = ctx.aligned_benchmark(node.benchmark)
    active = ctx.w - b
    return cp.quad_form(active, cp.psd_wrap(ctx.sigma))
