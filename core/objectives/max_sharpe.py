"""MaxSharpe objective via the Charnes–Cooper transform.

Maximizing the Sharpe ratio ``(μ̃ᵀw)/√(wᵀΣw)`` (with excess returns
``μ̃ = μ − r_f·1``) is non-convex as stated. For a long-only, fully-invested
book it linearizes under the Charnes–Cooper change of variables: set
``y = w / (μ̃ᵀw)`` (valid when the optimal ``μ̃ᵀw > 0``). Then

    minimize   yᵀ Σ y
    subject to μ̃ᵀ y = 1,  y ≥ 0,  (homogenized box bounds),

and the Sharpe-optimal weights are recovered as ``w = y / 1ᵀy`` (which is
automatically fully invested, so an explicit budget is implicit).

Scope and precondition (documented for the sprint):

* Transforming *arbitrary* user constraints through this change of variables is
  subtle, so this sprint supports MaxSharpe with **budget + long-only + box
  only**. Any other constraint raises a clear "unsupported with max_sharpe"
  error. Long-only is *required* (it is what makes ``y ≥ 0`` valid).
* Precondition: a positive-excess-return portfolio must exist. If every
  excess return is ≤ 0 the transform is infeasible (no portfolio beats the
  risk-free rate) and we raise immediately with that explanation.

Limitation flagged in the README roadmap: box bounds are homogenized exactly,
but richer constraints (group caps, turnover, factor/TE) are *not* supported
with max_sharpe in this sprint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import cvxpy as cp
import numpy as np
from pydantic import Field

from core.compile_context import CompiledProblem, validate_inputs
from core.exceptions import CompilationError
from core.irbase import ProblemClassImpact, _IRModel

if TYPE_CHECKING:
    from core.ir import PortfolioSpec

_SUPPORTED_KINDS = {"budget", "long_only", "box"}


class MaxSharpe(_IRModel):
    """Maximize the Sharpe ratio (Charnes–Cooper); long-only book only."""

    kind: Literal["max_sharpe"] = "max_sharpe"
    risk_free_rate: float = Field(
        default=0.0, description="Annualized risk-free rate r_f subtracted from μ to form excess returns."
    )
    problem_class_impact: ClassVar[ProblemClassImpact] = "convex"


def build(node: MaxSharpe, spec: PortfolioSpec, mu: np.ndarray, sigma: np.ndarray) -> CompiledProblem:
    validate_inputs(spec, mu, sigma)
    n = len(spec.universe)

    kinds = {c.kind for c in spec.constraints}
    unsupported = kinds - _SUPPORTED_KINDS
    if unsupported:
        raise CompilationError(
            "max_sharpe supports only budget + long_only + box constraints this "
            f"sprint; unsupported with max_sharpe: {sorted(unsupported)}."
        )
    if "long_only" not in kinds:
        raise CompilationError(
            "max_sharpe requires a long_only constraint in this sprint (the "
            "Charnes–Cooper y ≥ 0 transform assumes a long-only book)."
        )

    mu_excess = np.asarray(mu, dtype=float) - node.risk_free_rate
    if float(np.max(mu_excess)) <= 0.0:
        raise CompilationError(
            "max_sharpe is infeasible: no asset has positive excess return over "
            f"the risk-free rate ({node.risk_free_rate:g}), so no portfolio beats it."
        )

    ticker_index = {t: i for i, t in enumerate(spec.universe)}
    y = cp.Variable(n, name="y_sharpe")
    sum_y = cp.sum(y)
    # Normalization that fixes the scale of the transformed variable.
    hard: list[cp.Constraint] = [mu_excess @ y == 1]
    constraint_objs: dict[str, cp.Constraint] = {}

    for c in spec.constraints:
        if c.kind == "budget":
            # Fully-invested is implicit in the w = y/1ᵀy recovery; nothing to add.
            continue
        if c.kind == "long_only":
            con = y >= 0
            constraint_objs[c.id] = con
            hard.append(con)
        elif c.kind == "box":
            if c.tickers is None:
                target = y
            else:
                idx = np.array([ticker_index[t] for t in c.tickers], dtype=int)
                target = y[idx]
            # Homogenized box: lower·κ ≤ y_i ≤ upper·κ with κ = 1ᵀy.
            con = cp.hstack([c.upper * sum_y - target, target - c.lower * sum_y]) >= 0
            constraint_objs[c.id] = con
            hard.append(con)

    objective = cp.Minimize(cp.quad_form(y, cp.psd_wrap(sigma)))
    problem = cp.Problem(objective, hard)

    def _recover() -> np.ndarray:
        raw = np.asarray(y.value, dtype=float)
        return raw / float(np.sum(raw))

    return CompiledProblem(
        problem=problem,
        weights=y,
        constraint_objs=constraint_objs,
        spec=spec,
        weight_recovery=_recover,
    )
