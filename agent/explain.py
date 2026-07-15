"""LLM-narrated explanation of a SolutionReport, with a grounded fallback.

The deterministic ``template_summary`` is the trust fallback: if the model's
narration fails grounding verification twice in a row, the chat loop emits
the template instead. The template never references a number that is not
in the report, by construction.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.client import LLMClient
from agent.grounding import GroundingResult, verify
from core.exceptions import GroundingFailedError
from core.report import SolutionReport

_PROMPT_PATH = Path(__file__).parent / "prompts" / "explain_system.md"


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _report_as_json(report: SolutionReport) -> str:
    """Serialize the report for the model's context."""
    return json.dumps(report.to_dict(), indent=2)


def template_summary(report: SolutionReport) -> str:
    """Deterministic fallback narration: every number is copied from the report verbatim."""
    lines: list[str] = []
    obj_phrase = {
        "min_variance": "Minimum-variance",
        "mean_variance": "Mean-variance",
        "min_cvar": "Minimum-CVaR",
        "max_sharpe": "Maximum-Sharpe",
        "risk_parity": "Risk-parity",
        "min_tracking_error": "Minimum-tracking-error",
    }.get(report.objective_kind, report.objective_kind)
    status_label = (
        "CVXPY adapter status"
        if report.problem_class == "mip" or report.selected_names is not None
        else "status"
    )
    lines.append(
        f"{obj_phrase} solve returned {status_label} {report.status} on the "
        f"{report.n_assets}-asset universe in {report.solve_time_ms:.1f} ms "
        f"using {report.solver}."
    )
    if report.objective_decomposition is None:
        lines.append(f"Objective value: {report.objective_value:.6f}.")
        if report.var is not None:
            lines.append(f"VaR at the optimum: {report.var:.6f}.")
    else:
        objective = report.objective_decomposition
        lines.append(
            f"Solver objective score: {objective.solver_value:.6f} "
            f"{objective.solver_unit}."
        )
        if report.metrics:
            rendered_metrics = ", ".join(
                f"{metric.label} {metric.value:.6f} {metric.unit}"
                for metric in report.metrics
            )
            lines.append(f"Portfolio metrics: {rendered_metrics}.")
    lines.append(f"Nonzero positions: {report.nonzero_names} of {report.n_assets}.")
    if report.problem_class == "mip" or report.selected_names is not None:
        # A MIP solve: name the selected count and flag the conditionality up
        # front so the deterministic fallback also honours the Slice-3 rule.
        selected_count = len(report.selected_names or [])
        if report.duals_conditional and (report.sensitivities or report.binding):
            lines.append(
                "This is a mixed-integer (cardinality) solve: "
                f"{selected_count} names were selected, and the sensitivities below are "
                "conditional — they hold with the selected names held fixed, not globally."
            )
        else:
            lines.append(
                "This is a mixed-integer (cardinality) solve: "
                f"{selected_count} names were selected."
            )
        if report.termination_reason == "time_limit":
            if report.optimality_gap is None:
                lines.append(
                    "The time-limit result has no validated relative gap and must "
                    "not be treated as an optimal portfolio."
                )
            else:
                lines.append(
                    "The solver stopped at the time limit with a validated feasible "
                    f"incumbent; optimality was not proven and the relative gap is "
                    f"{report.optimality_gap:.6f}."
                )
        elif report.optimality_gap is not None:
            lines.append(
                f"The solver declared optimality within backend tolerance with relative gap "
                f"{report.optimality_gap:.6f}."
            )
    if report.binding:
        binders: list[str] = []
        for binding in report.binding:
            row = (
                ""
                if binding.row_label is None or binding.side is None
                else f", row {binding.row_label} {binding.side}"
            )
            unit = (
                ""
                if binding.sensitivity_unit is None
                else f" {binding.sensitivity_unit}"
            )
            binders.append(
                f"{binding.human_name}{row} "
                f"(shadow price {binding.shadow_price:.6f}{unit})"
            )
        prefix = "Conditional binding constraints" if report.duals_conditional else "Binding constraints"
        lines.append(f"{prefix}: {', '.join(binders)}.")
    else:
        if report.sensitivity_note:
            lines.append(report.sensitivity_note)
        elif report.sensitivities:
            lines.append("No reported constraint row has a material sensitivity.")
        else:
            lines.append("No row-aware constraint sensitivities are available for this solve.")
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
