"""Slice 3 tests: conditional-dual narration is flagged and stays grounded.

When ``duals_conditional`` is true the narration must (a) name the selected
count and (b) state the conditionality — and still pass grounding.verify. We
check both the deterministic fallback (template_summary) and a hand-written
LLM-style narration routed through explain() with a fake client.
"""

from __future__ import annotations

from agent.explain import explain, template_summary
from agent.grounding import verify
from core.report import BindingConstraint, SolutionReport


def _conditional_report() -> SolutionReport:
    return SolutionReport(
        weights={"AAA": 0.45, "CCC": 0.35, "DDD": 0.20, "BBB": 0.0, "EEE": 0.0},
        objective_kind="min_variance",
        objective_value=0.012300,
        solver="SCIP",
        solve_time_ms=21.0,
        status="optimal",
        n_assets=5,
        nonzero_names=3,
        binding=[BindingConstraint("tech", "the Tech group cap", 0.001100)],
        duals_conditional=True,
        selected_names=["AAA", "CCC", "DDD"],
        optimality_gap=0.0,
    )


class _FakeClient:
    def __init__(self, text: str) -> None:
        self._text = text

    def call_text(self, **_kwargs) -> str:
        return self._text

    def call_tool(self, **_kwargs):
        raise NotImplementedError


def test_template_summary_flags_conditionality_and_count() -> None:
    report = _conditional_report()
    text = template_summary(report)
    # Names the selected count.
    assert "3 names" in text or "3 of 5" in text
    # Flags conditionality explicitly.
    assert "conditional" in text.lower()
    assert "held fixed" in text.lower()
    # And the deterministic fallback is, by construction, grounded.
    assert verify(text, report).ok


def test_llm_conditional_narration_grounds() -> None:
    report = _conditional_report()
    narration = (
        "This was a mixed-integer cardinality solve, so the shadow prices are "
        "conditional: with the 3 selected names held fixed, the Tech group cap is "
        "the binding constraint at a conditional shadow price of 0.001100. The "
        "minimum-variance objective reached 0.012300 across the 5-name universe; "
        "the solver proved optimality (gap 0)."
    )
    client = _FakeClient(narration)
    text, result = explain(report, client=client)
    assert result.ok
    assert "conditional" in text.lower()
    assert "3 selected names" in text


def test_conditional_count_passes_grounding_when_selected_only() -> None:
    # "13 selected names" style count grounds via the selected/nonzero count.
    report = SolutionReport(
        weights={f"T{i}": (0.05 if i < 13 else 0.0) for i in range(20)},
        objective_kind="min_cvar",
        objective_value=0.0184,
        solver="HiGHS",
        solve_time_ms=120.0,
        status="optimal",
        n_assets=20,
        nonzero_names=13,
        var=0.0152,
        binding=[BindingConstraint("cap_tech", "the Tech group cap", 0.0011)],
        duals_conditional=True,
        selected_names=[f"T{i}" for i in range(13)],
        optimality_gap=0.0,
    )
    narration = (
        "With the 13 selected names held fixed, the Tech group cap's conditional "
        "shadow price is about 11 bps. Minimum-CVaR reached 0.0184 (VaR 0.0152)."
    )
    assert verify(narration, report).ok
