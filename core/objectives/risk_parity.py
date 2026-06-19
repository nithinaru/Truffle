"""RiskParity objective via the convex log-barrier surrogate.

Direct equal-risk-contribution is non-convex, but the surrogate

    minimize  ½ wᵀ Σ w − (1/n) Σ_i ln(w_i),   w > 0

is convex (it uses the exponential/log cone, which Clarabel supports). Its
unique optimum ``w*`` satisfies the first-order condition ``Σw* = (1/n)/w*_i``
component-wise, i.e. each asset's *marginal* risk contribution ``w_i·(Σw)_i`` is
equal across assets. Normalizing ``w ← w / 1ᵀw`` preserves that property and
yields the fully-invested equal-risk-contribution portfolio. The log term also
keeps every weight strictly positive, so long-only is automatic.

Scope (this sprint): risk_parity is solved as a standalone problem. Only
budget / long_only (both redundant here and ignored) are accepted; any other
constraint raises "unsupported with risk_parity".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import cvxpy as cp
import numpy as np

from core.compile_context import CompiledProblem, validate_inputs
from core.exceptions import CompilationError
from core.irbase import ProblemClassImpact, _IRModel

if TYPE_CHECKING:
    from core.ir import PortfolioSpec

_SUPPORTED_KINDS = {"budget", "long_only"}


class RiskParity(_IRModel):
    """Equal risk contribution via the convex log-barrier surrogate."""

    kind: Literal["risk_parity"] = "risk_parity"
    problem_class_impact: ClassVar[ProblemClassImpact] = "convex"


def build(node: RiskParity, spec: PortfolioSpec, mu: np.ndarray, sigma: np.ndarray) -> CompiledProblem:
    validate_inputs(spec, mu, sigma)
    n = len(spec.universe)

    kinds = {c.kind for c in spec.constraints}
    unsupported = kinds - _SUPPORTED_KINDS
    if unsupported:
        raise CompilationError(
            "risk_parity is solved standalone this sprint and accepts only "
            f"budget / long_only; unsupported with risk_parity: {sorted(unsupported)}."
        )

    w = cp.Variable(n, name="w_rp")
    risk = 0.5 * cp.quad_form(w, cp.psd_wrap(sigma))
    barrier = (1.0 / n) * cp.sum(cp.log(w))  # log domain enforces w > 0
    objective = cp.Minimize(risk - barrier)
    # No hard constraints: the surrogate's optimum is interior; budget/long_only
    # are handled by the post-solve normalization and the log domain.
    problem = cp.Problem(objective, [])

    def _recover() -> np.ndarray:
        raw = np.asarray(w.value, dtype=float)
        return raw / float(np.sum(raw))

    return CompiledProblem(
        problem=problem,
        weights=w,
        constraint_objs={},
        spec=spec,
        weight_recovery=_recover,
    )
