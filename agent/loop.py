"""Chat-session state machine.

State carried across turns:

* ``current_spec``: the spec the agent will amend. ``None`` until the first
  successful FreshSpec.
* ``pending_spec``: a spec awaiting user confirmation. Once confirmed it
  becomes the new ``current_spec`` and the solve runs.
* ``prices`` and ``sectors`` come from the CLI flags; the loop never reads
  the network.

Per-turn flow:

1. ``handle_user_message(text)`` → parses, decides one of:
   - Clarification → emit question, do not solve.
   - FreshSpec / SpecPatch → render echo/diff, stash as pending, wait for
     "y / n / edit" reply.
2. ``confirm_pending(decision)`` → on "y" solves + narrates and clears the
   pending; on "n" / "edit" discards the pending.

This thin layer is testable without a real CLI: the integration test in
test_agent_loop.py drives it with a fake client and asserts the sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from agent.client import LLMClient
from agent.explain import explain, template_summary
from agent.parse import parse_user_message
from agent.render import render_patch, render_spec
from agent.schema import (
    Clarification,
    FreshSpec,
    SpecPatch,
    apply_patch,
)
from core.exceptions import (
    GroundingFailedError,
    InfeasibleError,
    ParseFailedError,
    SolverError,
    UnboundedError,
)
from core.ir import PortfolioSpec
from core.report import SolutionReport
from core.solve import solve_spec

RESET_PHRASES = frozenset({"start over", "new portfolio", "reset", "/reset"})


@dataclass
class _Pending:
    """A spec awaiting user confirmation, plus the diff text to re-render."""

    spec: PortfolioSpec
    echo: str
    from_patch: SpecPatch | None = None
    prior_spec: PortfolioSpec | None = None


@dataclass
class TurnResult:
    """Typed return value of one chat turn — drives terminal rendering."""

    kind: Literal["clarification", "echo", "solved", "error", "info", "noop"]
    text: str
    pending_spec: PortfolioSpec | None = None
    report: SolutionReport | None = None
    explanation: str | None = None
    extra: dict[str, str] = field(default_factory=dict)


class ChatSession:
    """Coordinates parse → echo → confirm → solve → narrate.

    The session owns one ``LLMClient`` and one ``prices`` panel for its
    lifetime. ``handle_user_message`` and ``confirm_pending`` are the only
    public entrypoints the CLI calls.
    """

    def __init__(
        self,
        *,
        client: LLMClient,
        prices: pd.DataFrame,
        sectors: dict[str, str] | None = None,
    ) -> None:
        self._client = client
        self._prices = prices
        self._sectors = sectors or {}
        self.current_spec: PortfolioSpec | None = None
        self.pending: _Pending | None = None

    @property
    def universe_metadata(self) -> dict[str, object]:
        meta: dict[str, object] = {"tickers": list(self._prices.columns)}
        if self._sectors:
            meta["sectors"] = dict(self._sectors)
        return meta

    def reset(self) -> None:
        self.current_spec = None
        self.pending = None

    def handle_user_message(self, text: str) -> TurnResult:
        """Process one user message; return a TurnResult for rendering."""
        normalized = text.strip().lower()
        if normalized in RESET_PHRASES:
            self.reset()
            return TurnResult(kind="info", text="Session reset. Tell me about the new portfolio.")

        try:
            parse = parse_user_message(
                text,
                client=self._client,
                current_spec=self.current_spec,
                universe_metadata=self.universe_metadata,
            )
        except ParseFailedError as e:
            return TurnResult(kind="error", text=f"I could not parse that request.\n{e}")

        if isinstance(parse, Clarification):
            return TurnResult(kind="clarification", text=parse.question)
        if isinstance(parse, FreshSpec):
            spec = parse.spec
            echo = render_spec(spec)
            self.pending = _Pending(spec=spec, echo=echo)
            return TurnResult(kind="echo", text=echo, pending_spec=spec)
        if isinstance(parse, SpecPatch):
            if self.current_spec is None:
                # Patch arriving with no spec is a parser bug — treat as a
                # clarification rather than crash.
                return TurnResult(
                    kind="clarification",
                    text="Tell me about the portfolio you want before amending it.",
                )
            try:
                new_spec = apply_patch(self.current_spec, parse)
            except ValueError as e:
                return TurnResult(kind="error", text=f"The patch would produce an invalid spec.\n{e}")
            diff = render_patch(parse, self.current_spec, new_spec)
            self.pending = _Pending(
                spec=new_spec,
                echo=diff,
                from_patch=parse,
                prior_spec=self.current_spec,
            )
            return TurnResult(kind="echo", text=diff, pending_spec=new_spec)
        raise AssertionError(f"Unknown ParseResult: {type(parse).__name__}")

    def confirm_pending(self, decision: str) -> TurnResult:
        """Apply a y/n/edit decision to the pending spec.

        - 'y' / 'yes': commit pending as current and run the solve + narrate.
        - 'n' / 'no' / 'edit': discard pending, do not solve.
        - Anything else: treat as 'no' to keep the loop safe.
        """
        if self.pending is None:
            return TurnResult(kind="info", text="Nothing pending to confirm.")

        d = decision.strip().lower()
        if d in {"n", "no", "edit"}:
            self.pending = None
            return TurnResult(
                kind="info",
                text="Okay, discarded. Tell me what to change and I will re-render.",
            )
        if d not in {"y", "yes"}:
            # Safe default: do not solve on ambiguous confirmation.
            return TurnResult(
                kind="info",
                text="Please answer with 'y' to confirm, 'n' to discard, or 'edit' to amend.",
            )

        pending = self.pending
        self.pending = None
        self.current_spec = pending.spec

        try:
            _, report = solve_spec(pending.spec, self._prices)
        except InfeasibleError as e:
            return TurnResult(kind="error", text=str(e))
        except UnboundedError as e:
            return TurnResult(kind="error", text=str(e))
        except SolverError as e:
            return TurnResult(kind="error", text=str(e))

        try:
            explanation, _ = explain(report, client=self._client)
        except GroundingFailedError:
            explanation = template_summary(report)
        return TurnResult(
            kind="solved",
            text="Solved.",
            report=report,
            explanation=explanation,
        )
