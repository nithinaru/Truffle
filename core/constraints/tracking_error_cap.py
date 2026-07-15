"""TrackingErrorCap: bound tracking error vs a benchmark.

Tracking-error variance is ``(w − b)ᵀ Σ (w − b)``. The cap is compiled in
its natural volatility unit as ``‖Σ¹⁄²(w − b)‖₂ ≤ max_te``. This is a
second-order-cone constraint, and keeping the unsquared form means diagnostic
slack is directly interpretable as tracking error. Requires benchmark weights
as data (aligned to the universe).
"""

from __future__ import annotations

from typing import ClassVar, Literal

import cvxpy as cp
import numpy as np
from pydantic import Field

from core.compile_context import BuildContext, validated_array_result
from core.irbase import ProblemClassImpact, _ConstraintIRModel, _new_id


class TrackingErrorCap(_ConstraintIRModel):
    """Upper bound on tracking error relative to a named benchmark."""

    kind: Literal["tracking_error_cap"] = "tracking_error_cap"
    id: str = Field(default_factory=lambda: _new_id("tecap"))
    benchmark: str = Field(min_length=1, description="Benchmark name; weights supplied as data.")
    max_te: float = Field(
        gt=0.0, description="Upper bound on tracking error (annualized vol units)."
    )
    problem_class_impact: ClassVar[ProblemClassImpact] = "convex"
    elastic_default: ClassVar[bool] = True

    @property
    def slack_scale(self) -> float:
        return self.max_te


def build(node: TrackingErrorCap, ctx: BuildContext) -> cp.Constraint:
    b = ctx.aligned_benchmark(node.benchmark)
    # Express TE in its natural volatility unit, not squared-variance units.
    # This keeps a diagnostic slack directly interpretable as "raise TE by X".
    eigenvalues, eigenvectors = np.linalg.eigh(ctx.sigma)
    sqrt_sigma = validated_array_result(
        lambda: np.diag(np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T,
        label="Tracking-error covariance square root",
    )
    transformed_benchmark = validated_array_result(
        lambda: sqrt_sigma @ b,
        label=f"Tracking-error transformed benchmark {node.benchmark!r}",
    )
    return cp.norm(sqrt_sigma @ ctx.w - transformed_benchmark, 2) <= node.max_te
