"""LLM-narrated explanation of a SolutionReport, with a grounded fallback.

The deterministic ``template_summary`` is the trust fallback: if the model's
narration fails grounding verification twice in a row, the chat loop emits
the template instead. The template never references a number that is not
in the report, by construction.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from agent.client import LLMClient
from agent.grounding import GroundingResult, verify
from core.exceptions import GroundingFailedError
from core.report import SolutionReport

_PROMPT_PATH = Path(__file__).parent / "prompts" / "explain_system.md"


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _report_as_json(report: SolutionReport) -> str:
    """Serialize the report for the model's context."""
    data: dict[str, Any] = {
        "weights": report.weights,
        "objective_kind": report.objective_kind,
        "objective_value": report.objective_value,
        "var": report.var,
        "solver": report.solver,
        "solve_time_ms": report.solve_time_ms,
        "status": report.status,
        "n_assets": report.n_assets,
        "nonzero_names": report.nonzero_names,
        "binding": [asdict(b) for b in report.binding],
        "duals_conditional": report.duals_conditional,
        "selected_names": report.selected_names,
        "optimality_gap": report.optimality_gap,
    }
    return json.dumps(data, indent=2)


def template_summary(report: SolutionReport) -> str:
    """Deterministic fallback narration: every number is copied from the report verbatim."""
    lines: list[str] = []
    obj_phrase = {
        "min_variance": "Minimum-variance",
        "mean_variance": "Mean-variance",
        "min_cvar": "Minimum-CVaR",
    }.get(report.objective_kind, report.objective_kind)
    lines.append(
        f"{obj_phrase} solve returned status {report.status} on the {report.n_assets}-asset universe "
        f"in {report.solve_time_ms:.1f} ms using {report.solver}."
    )
    lines.append(f"Objective value: {report.objective_value:.6f}.")
    if report.var is not None:
        lines.append(f"VaR at the optimum: {report.var:.6f}.")
    lines.append(f"Nonzero positions: {report.nonzero_names} of {report.n_assets}.")
    if report.duals_conditional:
        # A MIP solve: name the selected count and flag the conditionality up
        # front so the deterministic fallback also honours the Slice-3 rule.
        lines.append(
            f"This is a mixed-integer (cardinality) solve: {report.nonzero_names} names "
            "were selected, and the shadow prices below are conditional — they hold "
            "with the selected names held fixed, not globally."
        )
    if report.binding:
        binders = ", ".join(
            f"{b.human_name} (shadow price {b.shadow_price:.6f})" for b in report.binding
        )
        prefix = "Conditional binding constraints" if report.duals_conditional else "Binding constraints"
        lines.append(f"{prefix}: {binders}.")
    else:
        lines.append("No constraints are binding at the optimum (all shadow prices ~ 0).")
    return " ".join(lines)


def _user_payload(report: SolutionReport, repair_note: str | None = None) -> str:
    parts = ["SOLUTION_REPORT (JSON):", _report_as_json(report)]
    if repair_note:
        parts.extend(["", "REPAIR_INSTRUCTION:", repair_note])
    return "\n".join(parts)


def explain(
    report: SolutionReport,
    *,
    client: LLMClient,
    max_attempts: int = 2,
) -> tuple[str, GroundingResult]:
    """Generate a grounded narration of ``report``.

    Args:
        report: The structured solver output.
        client: LLMClient (production or fake).
        max_attempts: Total tries (initial + repairs). Default 2.

    Returns:
        ``(explanation_text, grounding_result)``. ``grounding_result.ok``
        is guaranteed to be ``True`` for the returned text (otherwise we
        raise so the caller can swap in the template).

    Raises:
        GroundingFailedError: when both attempts fail verification. The
            chat loop catches this and prints ``template_summary(report)``
            so the user always sees a grounded narration.
    """
    system = _load_system_prompt()
    last_unmatched: list[str] = []

    for attempt in range(max_attempts):
        repair_note: str | None = None
        if attempt > 0 and last_unmatched:
            repair_note = (
                "Your previous response contained numerals that are NOT in the report: "
                f"{', '.join(last_unmatched)}. "
                "Regenerate. You MUST only cite values that appear (or are renderings of values that appear) "
                "in the SolutionReport. Do not invent rounded approximations."
            )
        text = client.call_text(
            system=system,
            messages=[{"role": "user", "content": _user_payload(report, repair_note)}],
        )
        result = verify(text, report)
        if result.ok:
            return text, result
        last_unmatched = result.unmatched

    raise GroundingFailedError(
        f"Could not ground explanation in {max_attempts} attempts. "
        f"Unmatched numerals on last attempt: {last_unmatched}."
    )
