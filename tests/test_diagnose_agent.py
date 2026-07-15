"""Grounding and chat repair-loop tests for Sprint 5."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from agent.diagnose import explain_conflict, template_conflict_summary
from agent.grounding import verify
from agent.loop import ChatSession
from core.diagnose import DiagnosisData, diagnose
from core.exceptions import GroundingFailedError, InfeasibleError
from core.ir import Box, Budget, LongOnly, MinVariance, PortfolioSpec
from core.solve import solve_spec


class FakeClient:
    def __init__(self, tool_response: dict[str, Any]) -> None:
        self.tool_response = tool_response
        self.tool_calls = 0
        self.text_calls = 0

    def call_tool(self, **_: Any) -> dict[str, Any]:
        self.tool_calls += 1
        return self.tool_response

    def call_text(self, **_: Any) -> str:
        self.text_calls += 1
        if self.text_calls == 1:
            return "The confirmed constraints have no common feasible allocation."
        return "The confirmed portfolio problem solved successfully."


class TextClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.payloads: list[str] = []

    def call_text(self, **kwargs: Any) -> str:
        self.payloads.append(kwargs["messages"][0]["content"])
        return self.responses.pop(0)

    def call_tool(self, **_: Any) -> dict[str, Any]:
        raise AssertionError("Conflict narration must not invoke a tool.")


class BlankNarrationClient(FakeClient):
    def call_text(self, **_: Any) -> str:
        self.text_calls += 1
        return "   "


def _spec() -> PortfolioSpec:
    return PortfolioSpec(
        universe=["A", "B", "C"],
        current_weights={"A": 1 / 3, "B": 1 / 3, "C": 1 / 3},
        objective=MinVariance(),
        constraints=[
            Budget(id="budget"),
            LongOnly(id="long"),
            Box(id="box", lower=0.0, upper=0.30),
        ],
    )


def _prices() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0002, 0.01, size=(90, 3))
    values = 100.0 * np.cumprod(1.0 + returns, axis=0)
    return pd.DataFrame(values, columns=["A", "B", "C"])


def _report():
    spec = _spec()
    data = DiagnosisData(
        mu=np.zeros(3),
        sigma=np.eye(3),
        scenarios=None,
        w_prev=np.array(spec.w_prev_vector()),
        sectors=None,
        benchmark_weights=None,
        factor_loadings=None,
    )
    return diagnose(spec, data)


def test_conflict_template_is_grounded_and_fabrication_is_rejected() -> None:
    report = _report()
    assert verify(template_conflict_summary(report), report).ok
    rejected = verify("Raise the cap to 47.5%.", report)
    assert not rejected.ok
    assert "47.5%" in rejected.unmatched

    # Repair targets are intentionally unavailable to free-form narration;
    # they are rendered from the typed report by deterministic server code.
    applied = report.repairs[0].changes[0].applied_value * 100
    repair_target = verify(f"A feasible target is {applied:g}%.", report)
    assert not repair_target.ok


def test_conflict_narration_payload_redacts_repairs() -> None:
    report = _report()
    client = TextClient(["The confirmed constraints have no common feasible allocation."])

    text, result = explain_conflict(report, client=client, max_attempts=1)

    assert result.ok
    assert text
    payload = client.payloads[0]
    assert '"repairs"' not in payload
    assert '"patch"' not in payload
    assert "applied_value" not in payload
    assert report.repairs[0].description not in payload


@pytest.mark.parametrize("narration", ["", "Relax the position cap."])
def test_conflict_narration_rejects_blank_or_free_form_repair_action(
    narration: str,
) -> None:
    client = TextClient([narration])
    with pytest.raises(GroundingFailedError):
        explain_conflict(_report(), client=client, max_attempts=1)


def test_chat_applies_verified_repair_re_echoes_then_solves() -> None:
    spec = _spec()
    client = FakeClient(
        {"result": {"kind": "fresh_spec", "spec": spec.model_dump(mode="json")}}
    )
    session = ChatSession(client=client, prices=_prices())

    echo = session.handle_user_message("Build this portfolio")
    assert echo.kind == "echo"
    conflict = session.confirm_pending("yes")
    assert conflict.kind == "conflict"
    assert conflict.conflict_report is not None
    assert client.tool_calls == 1

    invalid = session.select_repair("999")
    assert invalid.kind == "info"
    assert session.pending is None

    repaired_echo = session.handle_user_message("1")
    assert repaired_echo.kind == "echo"
    assert client.tool_calls == 1  # repair selection is deterministic, not parsed again
    assert session.pending is not None
    assert session.pending.spec.current_weights == spec.current_weights
    box = next(c for c in session.pending.spec.constraints if c.id == "box")
    assert isinstance(box, Box)
    assert box.upper == 0.34

    solved = session.confirm_pending("yes")
    assert solved.kind == "solved"
    assert solved.report is not None


def test_chat_replaces_blank_conflict_narration_with_grounded_fallback() -> None:
    spec = _spec()
    client = BlankNarrationClient(
        {"result": {"kind": "fresh_spec", "spec": spec.model_dump(mode="json")}}
    )
    session = ChatSession(client=client, prices=_prices())

    session.handle_user_message("Build this portfolio")
    conflict = session.confirm_pending("yes")

    assert conflict.kind == "conflict"
    assert conflict.conflict_report is not None
    assert conflict.explanation is not None
    assert conflict.explanation.strip()
    assert verify(conflict.explanation, conflict.conflict_report).ok
    assert client.text_calls == 2


def test_diagnosis_failure_preserves_primary_infeasible_error(monkeypatch) -> None:
    import core.diagnose as diagnosis_module  # noqa: PLC0415

    def fail_diagnosis(*_: Any, **__: Any) -> None:
        raise RuntimeError("secondary diagnosis failed")

    monkeypatch.setattr(diagnosis_module, "diagnose_infeasibility", fail_diagnosis)

    with pytest.raises(InfeasibleError) as caught:
        solve_spec(_spec(), _prices(), diagnose=True)

    assert "Solver reports the problem is infeasible" in str(caught.value)
    assert caught.value.conflict_report is None
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_chat_returns_primary_error_when_optional_diagnosis_fails(monkeypatch) -> None:
    import core.diagnose as diagnosis_module  # noqa: PLC0415

    def fail_diagnosis(*_: Any, **__: Any) -> None:
        raise RuntimeError("secondary diagnosis failed")

    monkeypatch.setattr(diagnosis_module, "diagnose_infeasibility", fail_diagnosis)
    spec = _spec()
    client = FakeClient(
        {"result": {"kind": "fresh_spec", "spec": spec.model_dump(mode="json")}}
    )
    session = ChatSession(client=client, prices=_prices())

    session.handle_user_message("Build this portfolio")
    result = session.confirm_pending("yes")

    assert result.kind == "error"
    assert "Solver reports the problem is infeasible" in result.text
    assert session.conflict is None
    assert client.text_calls == 0
