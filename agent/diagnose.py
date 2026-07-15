"""Grounded narration for deterministic infeasibility reports."""

from __future__ import annotations

import json
import re
from pathlib import Path

from agent.client import LLMClient
from agent.grounding import GroundingResult, verify
from core.exceptions import GroundingFailedError
from core.report import ConflictReport

_PROMPT_PATH = Path(__file__).parent / "prompts" / "diagnose_system.md"

# Conflict narration is descriptive only.  Actionable repair text is rendered
# later from server-owned ``Repair`` objects, so any free-form response that
# drifts into advice or amendment language is rejected and replaced by the
# deterministic fallback.  False positives are safe: they only select the
# fallback, never suppress a verified repair choice.
_REPAIR_ACTION_RE = re.compile(
    r"""\b(?:
        adjust(?:ed|ing)?|allow(?:ed|ing)?|amend(?:ed|ing)?|
        appl(?:y|ied|ying)|choos(?:e|ing)|chang(?:e|ed|ing)|
        decreas(?:e|ed|ing)|discard(?:ed|ing)?|dropp?(?:ed|ing)?|drop|
        eliminat(?:e|ed|ing)|fix(?:ed|ing)?|increas(?:e|ed|ing)|larger|
        loosen(?:ed|ing)?|lower(?:ed|ing)?|modif(?:y|ied|ying)|option(?:s)?|
        ought|rais(?:e|ed|ing)|recommend(?:ed|ing|ation)?|
        relax(?:ed|ing)?|remov(?:e|ed|ing)|repair(?:ed|ing|s)?|
        replac(?:e|ed|ing)|revis(?:e|ed|ing)|select(?:ed|ing|ion)?|
        should|smaller|suggest(?:ed|ing|ion)?|tighten(?:ed|ing)?|tr(?:y|ied|ying)
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)


def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _report_as_json(report: ConflictReport) -> str:
    """Serialize only facts that free-form narration may reference.

    In particular, repairs, patches, repair ranks/targets, and elastic solver
    internals never cross the LLM boundary.  The CLI (or another caller)
    renders those directly from the typed ``ConflictReport``.
    """

    payload = {
        "kind": report.kind,
        "solver_status": report.solver_status,
        "n_assets": report.n_assets,
        "minimality_status": report.minimality_status,
        "conflict_scope": report.conflict_scope,
        "conflict_set": [
            {
                "constraint_id": member.constraint_id,
                "constraint_kind": member.constraint_kind,
                "human_name": member.human_name,
                "relaxability": member.relaxability,
                "parameters": [
                    value.model_dump(mode="json") for value in member.parameters
                ],
            }
            for member in report.conflict_set
        ],
        "evidence": [evidence.model_dump(mode="json") for evidence in report.evidence],
    }
    return json.dumps(payload, indent=2)


def _verify_descriptive_narration(
    text: str, report: ConflictReport
) -> GroundingResult:
    """Require grounded numerals and reject free-form repair language."""

    grounded = verify(text, report)
    failures = list(grounded.unmatched)
    if _REPAIR_ACTION_RE.search(text):
        failures.append("<repair-action-language>")
    return GroundingResult(ok=not failures, unmatched=failures)


def template_conflict_summary(report: ConflictReport) -> str:
    """Deterministic fallback assembled only from trusted report fields."""

    names = ", ".join(member.human_name for member in report.conflict_set)
    if report.minimality_status == "verified_iis":
        opening = f"The verified irreducible conflict contains: {names}."
    else:
        opening = (
            f"These constraints appear to conflict, but minimality was not verified: {names}."
        )
    parts = [opening]
    parts.extend(evidence.text for evidence in report.evidence)
    structural = [
        member.human_name
        for member in report.conflict_set
        if member.relaxability != "relaxable"
    ]
    if structural:
        parts.append(
            "The conflict includes structural or non-negotiable constraints: "
            + ", ".join(structural)
            + "."
        )
    if report.repairs:
        parts.append("Verified repair choices are listed separately from this narration.")
    else:
        parts.append("No verified single-change repair is available from this diagnosis.")
    return " ".join(parts)


def explain_conflict(
    report: ConflictReport,
    *,
    client: LLMClient,
    max_attempts: int = 2,
) -> tuple[str, GroundingResult]:
    """Narrate a conflict report, retrying once on ungrounded numerals."""

    last_failures: list[str] = []
    for attempt in range(max_attempts):
        payload = ["CONFLICT_REPORT (JSON):", _report_as_json(report)]
        if attempt and last_failures:
            payload.extend(
                [
                    "",
                    "REPAIR_INSTRUCTION:",
                    "Your prior response failed these narration trust checks: "
                    + ", ".join(last_failures)
                    + ". Regenerate using only trusted report facts and do not "
                    "describe, recommend, or imply any repair action.",
                ]
            )
        text = client.call_text(
            system=_load_system_prompt(),
            messages=[{"role": "user", "content": "\n".join(payload)}],
        )
        result = _verify_descriptive_narration(text, report)
        if result.ok:
            return text, result
        last_failures = result.unmatched
    raise GroundingFailedError(
        "Could not ground conflict explanation after "
        f"{max_attempts} attempts. Failed checks: {last_failures}."
    )
