"""Tests for agent/grounding.py and agent/explain.py.

We build a SolutionReport by hand, then check:

* A correct explanation containing percentage and bps renderings of true
  values passes verify.
* An explanation containing a fabricated number fails verify.
* explain() succeeds when the (fake) client returns a grounded narration.
* explain() raises GroundingFailedError when the client always returns
  ungrounded text; the template_summary fallback contains the right facts.
"""

from __future__ import annotations

import pytest

from agent.explain import explain, template_summary
from agent.grounding import verify
from core.exceptions import GroundingFailedError
from core.report import BindingConstraint, SolutionReport


def _report() -> SolutionReport:
    return SolutionReport(
        weights={"AAA": 0.35, "BBB": 0.3747, "CCC": 0.1991, "DDD": 0.0762, "EEE": 0.0000},
        objective_kind="min_variance",
        objective_value=0.023513,
        var=None,
        solver="Clarabel",
        solve_time_ms=8.4,
        status="optimal",
        n_assets=5,
        nonzero_names=4,
        binding=[
            BindingConstraint(constraint_id="fully_invested", human_name="the budget constraint",
                              shadow_price=0.056272),
            BindingConstraint(constraint_id="cap_lowvol_asset", human_name="the AAA position cap",
                              shadow_price=0.026418),
        ],
    )


def test_verify_accepts_correct_explanation_with_percent_and_bps() -> None:
    r = _report()
    # 0.0264 -> 2.64% and 264 bps; 0.056272 -> 5.6272%; 4 of 5 names.
    text = (
        "The minimum-variance solve produced an objective value of 0.023513. "
        "The AAA position cap is binding with a shadow price of about 264 bps. "
        "The budget constraint also binds, shadow price 5.6272%. "
        "4 of 5 names received nonzero weight."
    )
    res = verify(text, r)
    assert res.ok, res.unmatched


def test_verify_rejects_fabricated_number() -> None:
    r = _report()
    text = "The portfolio Sharpe ratio is 1.43 with a max drawdown of 14.5%."
    res = verify(text, r)
    assert not res.ok
    assert any("1.43" in u or "14.5" in u for u in res.unmatched)


def test_verify_handles_leading_dot_and_scientific_notation_as_whole_tokens() -> None:
    r = _report()
    accepted = verify(
        "The objective is .023513, equivalently 2.3513e-2.",
        r,
    )
    assert accepted.ok, accepted.unmatched

    rejected = verify("Fabricated values: .475 and 1e3.", r)
    assert not rejected.ok
    assert "0.475" in rejected.unmatched
    assert "1000" in rejected.unmatched


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_verify_rejects_blank_narration(text: str) -> None:
    result = verify(text, _report())
    assert not result.ok
    assert result.unmatched == ["<blank>"]


def test_verify_allows_small_integers_for_counts() -> None:
    r = _report()
    text = "Truffle solved a 5-asset problem with 2 binding constraints."
    res = verify(text, r)
    assert res.ok, res.unmatched


def test_template_summary_is_self_grounded() -> None:
    r = _report()
    text = template_summary(r)
    res = verify(text, r)
    assert res.ok, res.unmatched


class _FakeTextClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def call_text(self, *, system, messages, **_kwargs):
        if not self._responses:
            raise AssertionError("FakeTextClient out of responses.")
        return self._responses.pop(0)

    def call_tool(self, **_kwargs):
        raise NotImplementedError


def test_explain_succeeds_with_grounded_response() -> None:
    r = _report()
    good = (
        "Minimum-variance solve, objective value 0.023513. "
        "Two constraints bind: the budget (shadow price 5.6272%) and "
        "the AAA position cap (264 bps). 4 of 5 names hold weight."
    )
    client = _FakeTextClient([good])
    text, result = explain(r, client=client)
    assert result.ok
    assert "0.023513" in text


def test_explain_raises_when_repeated_ungrounded_responses() -> None:
    r = _report()
    bad = "Sharpe 1.43, max drawdown 14.5%."
    client = _FakeTextClient([bad, bad])
    with pytest.raises(GroundingFailedError):
        explain(r, client=client, max_attempts=2)
