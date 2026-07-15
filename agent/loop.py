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
from agent.diagnose import explain_conflict, template_conflict_summary
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
from core.report import ConflictReport, SolutionReport
from core.solve import solve_spec

RESET_PHRASES = frozenset({"start over", "new portfolio", "reset", "/reset"})

# Universe-size guard for mixed-integer solves: branch-and-bound can grow
# expensive when the universe is large relative to the cardinality cap. At or
# above this universe size we both warn the user and pass the solver a
# wall-clock limit (so the demo stays responsive and any early stop is reported
# via the optimality gap).
MIP_UNIVERSE_GUARD = 30
MIP_TIME_LIMIT_S = 30.0


def _mip_guard(spec: PortfolioSpec) -> tuple[float | None, str | None]:
    """Return ``(time_limit_s, warning)`` for a mixed-integer spec.

    ``(None, None)`` for continuous specs and for small mixed-integer ones; the
    continuous UX is therefore completely untouched.
    """
    if spec.problem_class != "mip":
        return None, None
    n = len(spec.universe)
    if n >= MIP_UNIVERSE_GUARD:
        warn = (
            f"Note: {n} names is large for a mixed-integer search, so the solver "
            f"is capped at {MIP_TIME_LIMIT_S:g}s; I'll report the optimality gap "
            "it reaches (0 means proven optimal)."
        )
        return MIP_TIME_LIMIT_S, warn
    return None, None


@dataclass
class _Pending:
    """A spec awaiting user confirmation, plus the diff text to re-render."""

    spec: PortfolioSpec
    echo: str
    from_patch: SpecPatch | None = None
    prior_spec: PortfolioSpec | None = None
    time_limit_s: float | None = None


@dataclass
class _ConflictState:
    """The server-owned diagnosis whose verified repairs may be selected."""

    spec: PortfolioSpec
    report: ConflictReport


@dataclass
class TurnResult:
    """Typed return value of one chat turn — drives terminal rendering."""

    kind: Literal["clarification", "echo", "solved", "conflict", "error", "info", "noop"]
    text: str
    pending_spec: PortfolioSpec | None = None
    report: SolutionReport | None = None
    conflict_report: ConflictReport | None = None
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
        benchmarks: dict[str, dict[str, float]] | None = None,
        factors: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self._client = client
        self._prices = prices
        self._sectors = sectors or {}
        self._benchmarks = benchmarks
        self._factors = factors
        self.current_spec: PortfolioSpec | None = None
        self.pending: _Pending | None = None
        self.conflict: _ConflictState | None = None

    @property
    def universe_metadata(self) -> dict[str, object]:
        meta: dict[str, object] = {"tickers": list(self._prices.columns)}
        if self._sectors:
            meta["sectors"] = dict(self._sectors)
        return meta

    def reset(self) -> None:
        self.current_spec = None
        self.pending = None
        self.conflict = None

    def handle_user_message(self, text: str) -> TurnResult:
        """Process one user message; return a TurnResult for rendering."""
        normalized = text.strip().lower()
        if normalized in RESET_PHRASES:
            self.reset()
            return TurnResult(kind="info", text="Session reset. Tell me about the new portfolio.")
        if self.conflict is not None:
            selectors = {
                selector
                for repair in self.conflict.report.repairs
                for selector in (repair.repair_id, str(repair.rank))
            }
            if text.strip() in selectors:
                return self.select_repair(text.strip())

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
            self.conflict = None
            echo = render_spec(spec)
            time_limit, guard_warn = _mip_guard(spec)
            if guard_warn:
                echo = f"{echo}\n\n{guard_warn}"
            self.pending = _Pending(spec=spec, echo=echo, time_limit_s=time_limit)
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
            self.conflict = None
            time_limit, guard_warn = _mip_guard(new_spec)
            if guard_warn:
                diff = f"{diff}\n\n{guard_warn}"
            self.pending = _Pending(
                spec=new_spec,
                echo=diff,
                from_patch=parse,
                prior_spec=self.current_spec,
                time_limit_s=time_limit,
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
            _, report = solve_spec(
                pending.spec,
                self._prices,
                sectors=self._sectors or None,
                benchmarks=self._benchmarks,
                factors=self._factors,
                time_limit_s=pending.time_limit_s,
                diagnose=True,
            )
        except InfeasibleError as e:
            if not isinstance(e.conflict_report, ConflictReport):
                # Do not leave an older repair menu live if the optional
                # diagnosis pass failed for this newly confirmed spec.
                self.conflict = None
                return TurnResult(kind="error", text=str(e))
            conflict_report = e.conflict_report
            self.conflict = _ConflictState(spec=pending.spec, report=conflict_report)
            try:
                explanation, _ = explain_conflict(conflict_report, client=self._client)
            except Exception:
                # Narration is optional presentation over a deterministic
                # report.  Grounding failures and text-provider failures both
                # fall back locally; neither may crash or alter repair data.
                explanation = template_conflict_summary(conflict_report)
            return TurnResult(
                kind="conflict",
                text="The confirmed constraints are infeasible.",
                conflict_report=conflict_report,
                explanation=explanation,
            )
        except UnboundedError as e:
            return TurnResult(kind="error", text=str(e))
        except SolverError as e:
            return TurnResult(kind="error", text=str(e))

        try:
            explanation, _ = explain(report, client=self._client)
        except GroundingFailedError:
            explanation = template_summary(report)
        self.conflict = None
        return TurnResult(
            kind="solved",
            text="Solved.",
            report=report,
            explanation=explanation,
        )

    def select_repair(self, selection: str | int) -> TurnResult:
        """Apply a server-owned verified repair and return a new spec echo.

        Selection never auto-solves.  The repaired spec must pass the same
        deterministic echo and explicit confirmation gate as every other
        amendment.
        """

        if self.conflict is None:
            return TurnResult(kind="info", text="There is no conflict repair to select.")
        token = str(selection).strip()
        repair = next(
            (
                candidate
                for candidate in self.conflict.report.repairs
                if token in {candidate.repair_id, str(candidate.rank)}
            ),
            None,
        )
        if repair is None:
            return TurnResult(
                kind="info",
                text="Choose one of the listed repair numbers; no change was applied.",
                conflict_report=self.conflict.report,
            )
        prior = self.conflict.spec
        try:
            repaired = apply_patch(prior, repair.patch)
        except ValueError as exc:
            return TurnResult(kind="error", text=f"The verified repair no longer applies.\n{exc}")
        echo = (
            render_patch(repair.patch, prior, repaired)
            + "\n\n"
            + render_spec(repaired)
        )
        time_limit, guard_warn = _mip_guard(repaired)
        if guard_warn:
            echo = f"{echo}\n\n{guard_warn}"
        self.pending = _Pending(
            spec=repaired,
            echo=echo,
            from_patch=repair.patch,
            prior_spec=prior,
            time_limit_s=time_limit,
        )
        return TurnResult(kind="echo", text=echo, pending_spec=repaired)
