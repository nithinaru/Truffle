"""FactorExposure: bound a portfolio's exposure to a named factor.

``min_exposure ≤ loadingsᵀ w ≤ max_exposure`` where the per-asset factor
loadings are supplied as data (aligned to the universe). At least one bound
must be given. Linear ⇒ convex.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import cvxpy as cp
from pydantic import Field, model_validator

from core.compile_context import BuildContext
from core.irbase import ProblemClassImpact, _ConstraintIRModel, _new_id


class FactorExposure(_ConstraintIRModel):
    """Bound ``loadingsᵀ w`` for one factor; supply min, max, or both."""

    kind: Literal["factor_exposure"] = "factor_exposure"
    id: str = Field(default_factory=lambda: _new_id("factor"))
    factor: str = Field(min_length=1, description="Factor name; loadings supplied as data.")
    min_exposure: float | None = Field(default=None, description="Lower bound on loadingsᵀw.")
    max_exposure: float | None = Field(default=None, description="Upper bound on loadingsᵀw.")
    problem_class_impact: ClassVar[ProblemClassImpact] = "convex"
    elastic_default: ClassVar[bool] = True

    @property
    def slack_scale(self) -> float:
        """Bound width, or the magnitude of the sole one-sided bound."""
        if self.min_exposure is not None and self.max_exposure is not None:
            return self.max_exposure - self.min_exposure
        bound = self.max_exposure if self.max_exposure is not None else self.min_exposure
        assert bound is not None  # guaranteed by _check_bounds
        return abs(bound)

    @model_validator(mode="after")
    def _check_bounds(self) -> FactorExposure:
        if self.min_exposure is None and self.max_exposure is None:
            raise ValueError(
                f"FactorExposure {self.factor!r} needs at least one of min_exposure / max_exposure."
            )
        if (
            self.min_exposure is not None
            and self.max_exposure is not None
            and self.min_exposure > self.max_exposure
        ):
            raise ValueError(
                f"FactorExposure min_exposure={self.min_exposure} > "
                f"max_exposure={self.max_exposure}."
            )
        return self


def build(node: FactorExposure, ctx: BuildContext) -> cp.Constraint:
    loadings = ctx.aligned_factor(node.factor)
    exposure = loadings @ ctx.w
    if node.min_exposure is not None and node.max_exposure is not None:
        return cp.hstack([node.max_exposure - exposure, exposure - node.min_exposure]) >= 0
    if node.max_exposure is not None:
        return exposure <= node.max_exposure
    return exposure >= node.min_exposure
