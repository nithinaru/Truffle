"""Truffle intermediate representation (IR).

This module is the heart of the project: a typed, validated description of a
portfolio optimization problem. Everything downstream — compiler, solver,
duals, explanation — consumes ``PortfolioSpec`` instances. The LLM layer
(Sprint 2) will *only* be allowed to emit IR; it will never touch math.

Design rules (from BLUEPRINT.md sections 4 & 5):

* Every constraint and objective declares ``problem_class_impact`` so the
  compiler can aggregate the problem class (``"convex"`` vs ``"mip"``).
* Every ``Constraint`` carries a unique ``id``. The compiler stores a
  ``{id -> cvxpy.Constraint}`` map so :mod:`core.duals` can recover shadow
  prices and name them back to the user.
* Discriminated unions are keyed on a ``kind`` ``Literal`` field; Pydantic v2
  picks the concrete model from the tag without any runtime ``isinstance``
  acrobatics in the compiler.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import Field, model_validator

from core.constraints.cvar_limit import CVaRLimit
from core.constraints.factor_exposure import FactorExposure
from core.constraints.group_cap import GroupCap
from core.constraints.tracking_error_cap import TrackingErrorCap
from core.constraints.transaction_cost import TransactionCost
from core.constraints.turnover_cap import TurnoverCap

# Base IR model + helpers live in the dependency-light core.irbase leaf module so
# the per-node constraint modules above can subclass _IRModel without cycling
# back through this module.
from core.irbase import ProblemClassImpact, _IRModel, _new_id

__all__ = [
    "Box",
    "Budget",
    "CVaRLimit",
    "Constraint",
    "FactorExposure",
    "GroupCap",
    "LongOnly",
    "MeanVariance",
    "MinCVaR",
    "MinVariance",
    "Objective",
    "PortfolioSpec",
    "ProblemClassImpact",
    "TrackingErrorCap",
    "TransactionCost",
    "TurnoverCap",
]


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------


class MinVariance(_IRModel):
    """Minimum-variance objective: ``min wᵀ Σ w``.

    No expected-return input required; this is the recommended default per the
    blueprint's "expected returns are the weakest input" principle.
    """

    kind: Literal["min_variance"] = "min_variance"
    problem_class_impact: ClassVar[ProblemClassImpact] = "convex"


class MeanVariance(_IRModel):
    """Markowitz mean-variance objective: ``min wᵀ Σ w − λ μᵀ w``.

    ``risk_aversion`` (λ) trades expected return against variance. Larger λ
    pushes the optimum toward higher-expected-return portfolios; λ = 0 reduces
    to min-variance.
    """

    kind: Literal["mean_variance"] = "mean_variance"
    risk_aversion: float = Field(gt=0.0, description="λ in min wᵀΣw − λμᵀw.")
    problem_class_impact: ClassVar[ProblemClassImpact] = "convex"


class MinCVaR(_IRModel):
    """Minimum-CVaR objective via the Rockafellar–Uryasev LP reformulation.

    For S equally-likely return scenarios ``r_s`` (per-period, *return*
    convention, not loss), the CVaR at level ``cvar_alpha`` of portfolio
    losses ``L_s = -r_s · w`` reduces to the LP

        min  t + (1/((1-α)S)) Σ z_s
        s.t. z_s ≥ −r_s·w − t,   z_s ≥ 0

    The optimal ``t`` is the VaR; the objective value is the CVaR. Per
    BLUEPRINT §5, this is the crown-jewel objective that differentiates
    Truffle from PyPortfolioOpt-style libraries.
    """

    kind: Literal["min_cvar"] = "min_cvar"
    cvar_alpha: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description="Confidence level α (0 < α < 1). Typical values: 0.90, 0.95, 0.99.",
    )
    problem_class_impact: ClassVar[ProblemClassImpact] = "convex"


Objective = Annotated[
    MinVariance | MeanVariance | MinCVaR,
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class Budget(_IRModel):
    """Sum-of-weights constraint: ``Σ w_i = total`` (default fully invested)."""

    kind: Literal["budget"] = "budget"
    id: str = Field(default_factory=lambda: _new_id("budget"))
    total: float = Field(default=1.0, description="Right-hand side of Σw = total.")
    problem_class_impact: ClassVar[ProblemClassImpact] = "convex"


class LongOnly(_IRModel):
    """Long-only: ``w_i ≥ 0`` for every asset."""

    kind: Literal["long_only"] = "long_only"
    id: str = Field(default_factory=lambda: _new_id("longonly"))
    problem_class_impact: ClassVar[ProblemClassImpact] = "convex"


class Box(_IRModel):
    """Per-asset bounds: ``lower ≤ w_i ≤ upper``.

    If ``tickers`` is ``None`` the bound applies to every asset. If provided,
    it must be a subset of the spec's ``universe`` (semantic check happens in
    :class:`PortfolioSpec`).
    """

    kind: Literal["box"] = "box"
    id: str = Field(default_factory=lambda: _new_id("box"))
    lower: float = Field(description="Lower bound on weight.")
    upper: float = Field(description="Upper bound on weight.")
    tickers: list[str] | None = Field(
        default=None,
        description="If set, restrict this bound to the listed tickers; else applies universe-wide.",
    )
    problem_class_impact: ClassVar[ProblemClassImpact] = "convex"

    @model_validator(mode="after")
    def _check_bounds(self) -> Box:
        if self.lower > self.upper:
            raise ValueError(f"Box bound lower={self.lower} > upper={self.upper}.")
        return self


Constraint = Annotated[
    Budget
    | LongOnly
    | Box
    | GroupCap
    | TurnoverCap
    | TransactionCost
    | CVaRLimit
    | TrackingErrorCap
    | FactorExposure,
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Top-level spec
# ---------------------------------------------------------------------------


class PortfolioSpec(_IRModel):
    """Top-level description of a portfolio optimization problem.

    The IR is intentionally narrow in Sprint 1: only constraints/objectives
    that exercise the compiler-and-duals plumbing end to end. Later sprints
    add CVaR, cardinality, group caps, etc.
    """

    universe: list[str] = Field(min_length=1, description="Asset tickers; order is canonical.")
    objective: Objective
    constraints: list[Constraint] = Field(default_factory=list)
    current_weights: dict[str, float] | None = Field(
        default=None,
        description=(
            "Pre-trade holdings as {ticker -> weight}, used by turnover and "
            "transaction-cost terms. Convention: any universe ticker absent "
            "from this mapping is treated as weight 0.0; None means the whole "
            "vector is 0.0, i.e. the portfolio is being built fresh from cash "
            "(full deployment). This is the single-shot input; per-rebalance "
            "threading is a later sprint."
        ),
    )

    @model_validator(mode="after")
    def _validate_semantics(self) -> PortfolioSpec:
        # Unique tickers — duplicate tickers would silently double-count.
        seen: set[str] = set()
        for t in self.universe:
            if t in seen:
                raise ValueError(f"Duplicate ticker in universe: {t!r}.")
            seen.add(t)

        # Constraint ids are unique (else dual mapping is ambiguous).
        ids = [c.id for c in self.constraints]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate constraint ids: {ids}.")

        # Box tickers must be in the universe.
        universe_set = set(self.universe)
        for c in self.constraints:
            if isinstance(c, Box) and c.tickers is not None:
                missing = sorted(set(c.tickers) - universe_set)
                if missing:
                    raise ValueError(
                        f"Box constraint {c.id} references tickers not in universe: {missing}."
                    )

        # current_weights keys must be a subset of the universe — a weight on
        # an unknown ticker is almost certainly a user/agent mistake we should
        # surface, not silently drop.
        if self.current_weights is not None:
            unknown = sorted(set(self.current_weights) - universe_set)
            if unknown:
                raise ValueError(
                    f"current_weights references tickers not in universe: {unknown}."
                )
        return self

    def w_prev_vector(self) -> list[float]:
        """Return ``current_weights`` aligned to ``universe`` order.

        Missing tickers (and the ``None`` case) resolve to ``0.0``, encoding
        the "fresh from cash" convention documented on ``current_weights``.
        """
        cw = self.current_weights or {}
        return [float(cw.get(t, 0.0)) for t in self.universe]

    @property
    def problem_class(self) -> ProblemClassImpact:
        """Aggregate problem class: ``"mip"`` if anything forces it, else ``"convex"``."""
        impacts: list[ProblemClassImpact] = [type(self.objective).problem_class_impact]
        impacts.extend(type(c).problem_class_impact for c in self.constraints)
        return "mip" if "mip" in impacts else "convex"
