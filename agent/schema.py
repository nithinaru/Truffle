"""Parse-result schema returned by the LLM layer.

The LLM is restricted to emitting exactly one of:

* :class:`FreshSpec` — a brand-new portfolio spec (no prior context).
* :class:`SpecPatch` — a typed amendment to an existing spec.
* :class:`Clarification` — a single follow-up question when the request is
  ambiguous (vague quantities, contradictions, unknown tickers).

These three are wrapped in the :class:`ParseResult` discriminated union
keyed on ``kind``. Pydantic v2 validates the JSON against this schema so
the model cannot return free-form natural language at the *math* boundary.

Why discrimination matters: every downstream branch in the chat loop is a
pure function of ``ParseResult.kind``. No string sniffing, no fragile regex.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.ir import Constraint, Objective, PortfolioSpec


class _AgentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Clarification(_AgentModel):
    """Exactly one follow-up question for the user.

    Returned when the LLM cannot turn the message into a valid spec or
    amendment without guessing — vague quantities ("not too concentrated"),
    contradictions ("at least 8%, no more than 5%"), unknown tickers, etc.

    The parse policy is "one question max"; the loop will surface ``question``
    verbatim and wait for the next user message.

    Attributes:
        kind: Discriminator tag.
        question: Plain-English question shown to the user. Required.
        partial_spec: Optional best-effort partial spec for context — never
            consumed as a final spec; only used in render to show what was
            understood so far.
        reason: Short machine-readable label for the ambiguity category
            (``"vague_quantity"``, ``"contradiction"``, ``"unknown_ticker"``,
            ``"missing_universe"``, ``"other"``).
    """

    kind: Literal["clarification"] = "clarification"
    question: str = Field(min_length=1)
    partial_spec: PortfolioSpec | None = None
    reason: str = Field(min_length=1)


class FreshSpec(_AgentModel):
    """A complete new portfolio spec inferred from the user message.

    Returned when there is no prior spec to amend (first message of the
    conversation, or after the user has explicitly reset).
    """

    kind: Literal["fresh_spec"] = "fresh_spec"
    spec: PortfolioSpec


class SpecPatch(_AgentModel):
    """A typed amendment to the current spec.

    The patch is minimal-by-design — the chat loop is the only thing that
    applies it. Fields:

    * ``add_constraints``: append these to ``spec.constraints``.
    * ``remove_constraint_ids``: drop any constraint whose ``id`` is listed.
    * ``replace_objective``: if set, swap the spec's objective.
    * ``set_universe``: rare, but supported (e.g. "swap in MSFT instead of META").
    * ``estimation_overrides`` / ``backtest_overrides``: reserved for Sprint 3+
      typed config keys; kept as bare dicts in Sprint 2 so we don't lock the
      shape before the BacktestConfig model exists.

    Operations apply in this order: remove → replace_objective → add → set_universe.
    """

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


ParseResult = Annotated[
    FreshSpec | SpecPatch | Clarification,
    Field(discriminator="kind"),
]


class ParseEnvelope(_AgentModel):
    """Top-level wrapper used as the model's structured-output schema.

    The Anthropic SDK's tool-use forced-output path needs a single concrete
    JSON object; a top-level discriminated union doesn't render cleanly as
    one tool input schema, so we wrap it in this envelope.
    """

    result: ParseResult


def apply_patch(spec: PortfolioSpec, patch: SpecPatch) -> PortfolioSpec:
    """Apply a SpecPatch to a spec and return the new (validated) PortfolioSpec.

    Operations are applied in a fixed order (remove, replace_objective, add,
    set_universe) so two patches with the same fields always produce the same
    result regardless of dict-iteration order.
    """
    constraints = [c for c in spec.constraints if c.id not in set(patch.remove_constraint_ids)]
    objective = patch.replace_objective if patch.replace_objective is not None else spec.objective
    constraints = constraints + list(patch.add_constraints)
    universe = patch.set_universe if patch.set_universe is not None else list(spec.universe)
    return PortfolioSpec(
        universe=universe,
        objective=objective,
        constraints=constraints,
    )
