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

from pydantic import BaseModel, ConfigDict, Field

from core.ir import PortfolioSpec
from core.patch import SpecPatch, apply_patch

__all__ = [
    "Clarification",
    "FreshSpec",
    "ParseEnvelope",
    "ParseResult",
    "SpecPatch",
    "apply_patch",
]


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
