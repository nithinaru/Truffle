"""Typed, deterministic amendments to a :class:`~core.ir.PortfolioSpec`.

Patch types live in ``core`` because they are part of the solver-facing data
contract.  Agent schemas re-export these names for backwards compatibility,
but core report models can import them without depending on the agent layer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.ir import Constraint, Objective, PortfolioSpec

__all__ = ["SpecPatch", "apply_patch"]


class SpecPatch(BaseModel):
    """A typed amendment to the current portfolio spec.

    Operations apply in a fixed order: remove constraints, replace the
    objective, add constraints, then set the universe.  Estimation and
    backtest overrides remain reserved for their respective future config
    models and are intentionally not applied to ``PortfolioSpec`` yet.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["spec_patch"] = "spec_patch"
    add_constraints: list[Constraint] = Field(default_factory=list)
    remove_constraint_ids: list[str] = Field(default_factory=list)
    replace_objective: Objective | None = None
    set_universe: list[str] | None = None
    estimation_overrides: dict[str, float | int | str] = Field(default_factory=dict)
    backtest_overrides: dict[str, float | int | str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _at_least_one_change(self) -> SpecPatch:
        if not (
            self.add_constraints
            or self.remove_constraint_ids
            or self.replace_objective is not None
            or self.set_universe is not None
            or self.estimation_overrides
            or self.backtest_overrides
        ):
            raise ValueError(
                "SpecPatch is empty — model should have returned a Clarification "
                "if no concrete change was warranted."
            )
        return self


def apply_patch(spec: PortfolioSpec, patch: SpecPatch) -> PortfolioSpec:
    """Return a validated spec with ``patch`` applied deterministically.

    ``current_weights`` is carried forward unchanged unless the universe is
    replaced.  On a universe replacement, holdings that remain in the new
    universe keep their weights and removed holdings are treated as
    liquidated.  This preserves rebalance context without retaining stale
    ticker keys that would make the amended spec invalid.
    """
    remove_ids = set(patch.remove_constraint_ids)
    constraints = [constraint for constraint in spec.constraints if constraint.id not in remove_ids]
    objective = patch.replace_objective if patch.replace_objective is not None else spec.objective
    constraints.extend(patch.add_constraints)
    universe = patch.set_universe if patch.set_universe is not None else spec.universe
    current_weights = spec.current_weights
    if patch.set_universe is not None and current_weights is not None:
        universe_set = set(universe)
        current_weights = {
            ticker: weight
            for ticker, weight in current_weights.items()
            if ticker in universe_set
        }

    return PortfolioSpec(
        universe=universe,
        objective=objective,
        constraints=constraints,
        current_weights=current_weights,
    )
