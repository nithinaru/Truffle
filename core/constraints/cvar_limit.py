"""CVaRLimit: bound portfolio CVaR, ``CVaR_α(w) ≤ max_cvar``.

Reuses the Slice 0 Rockafellar–Uryasev block: the CVaR expression is built from
per-scenario tail slacks and bounded above. Requires a scenario matrix, exactly
like the ``MinCVaR`` objective; absent scenarios raise ``CompilationError`` with
the shared message. The reformulation is an LP ⇒ convex.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import cvxpy as cp
from pydantic import Field

from core.compile_context import BuildContext, build_cvar_block, validate_scenarios
from core.irbase import ProblemClassImpact, _IRModel, _new_id


class CVaRLimit(_IRModel):
    """Upper bound on CVaR at confidence ``alpha`` (return-fraction units)."""

    kind: Literal["cvar_limit"] = "cvar_limit"
    id: str = Field(default_factory=lambda: _new_id("cvarlim"))
    alpha: float = Field(gt=0.0, lt=1.0, description="Confidence level α in (0, 1).")
    max_cvar: float = Field(
        description="Upper bound on CVaR_α, in the same units as scenario returns."
    )
    problem_class_impact: ClassVar[ProblemClassImpact] = "convex"


def build(node: CVaRLimit, ctx: BuildContext) -> cp.Constraint:
    scenarios = validate_scenarios(ctx.scenarios, ctx.n)
    cvar_expr, t_var, z_var, aux = build_cvar_block(
        scenarios, ctx.w, node.alpha, name_suffix=f"_lim_{node.id}"
    )
    ctx.aux_constraints.extend(aux)
    # Expose the limit's VaR variable for callers that want to inspect it.
    ctx.extra_vars[f"t_{node.id}"] = t_var
    ctx.extra_vars[f"z_{node.id}"] = z_var
    return cvar_expr <= node.max_cvar
