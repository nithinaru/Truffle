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

from dataclasses import replace

import pytest

from agent.explain import explain, template_summary
from agent.grounding import verify
from core.exceptions import GroundingFailedError
from core.report import BindingConstraint, SolutionReport
from core.report_semantics import PortfolioMetric


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


def test_verify_accepts_correct_explanation_with_typed_raw_sensitivities() -> None:
    r = _report()
    text = (
        "The minimum-variance solve produced an objective value of 0.023513. "
        "The AAA position cap is binding with a sensitivity magnitude of 0.026418. "
        "The budget constraint also binds with sensitivity magnitude 0.056272. "
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


def test_solution_grounding_enforces_metric_units() -> None:
    report = replace(
        _report(),
        objective_value=0.9,
        weights={"AAA": 0.35, "BBB": 0.65},
        n_assets=2,
        nonzero_names=2,
        binding=[],
        metrics=(
            PortfolioMetric(
                key="variance",
                label="Portfolio variance",
                value=0.04,
                unit="fraction_squared_per_year",
                definition="annualized portfolio variance",
            ),
            PortfolioMetric(
                key="expected_return",
                label="Expected return",
                value=0.08,
                unit="fraction_per_year",
                definition="annualized expected return",
            ),
        ),
    )
    assert verify("Variance is 0.04 and expected return is 8%.", report).ok
    rejected = verify("Variance is 4%.", report)
    assert not rejected.ok
    assert "4%" in rejected.unmatched

    cross_field = verify("Variance is 8%.", report)
    assert not cross_field.ok
    assert "8%" in cross_field.unmatched

    ambiguous = replace(
        report,
        metrics=(
            replace(report.metrics[0], value=0.08),
            report.metrics[1],
        ),
    )
    # Explicit field labels disambiguate equal raw values with different units.
    assert verify("Expected return is 8%.", ambiguous).ok
    assert not verify("Variance is 8%.", ambiguous).ok


def test_solution_grounding_does_not_convert_sensitivity_rates_to_bps() -> None:
    report = replace(
        _report(),
        # A real 40% weight must not authorize percentage or bps renderings of
        # an unrelated raw sensitivity derivative.
        weights={"AAA": 0.4, "BBB": 0.6},
        n_assets=2,
        nonzero_names=2,
        objective_value=0.9,
        binding=[
            BindingConstraint(
                "cap",
                "the cap",
                0.001,
                row_label="AAA",
                side="upper",
                sensitivity_unit="annualized_variance_per_portfolio_weight_fraction",
            )
        ],
    )
    assert verify("The signed sensitivity magnitude is 0.001.", report).ok
    assert verify("The portfolio weight is 40%.", report).ok
    for prose, rendered in (
        ("The sensitivity is 40%.", "40%"),
        ("The sensitivity is 4000 bps.", "4000bps"),
    ):
        rejected = verify(prose, report)
        assert not rejected.ok
        assert rendered in rejected.unmatched


def test_relative_gap_allows_percent_but_not_basis_points() -> None:
    report = replace(
        _report(),
        problem_class="mip",
        selected_names=["AAA", "BBB", "CCC", "DDD"],
        optimality_gap=0.125,
        incumbent_validated=True,
        binding=[],
    )
    assert verify("The relative gap is 12.5%.", report).ok
    assert not verify("The relative gap is 1250 bps.", report).ok


def test_unreported_small_integer_is_not_globally_allowlisted() -> None:
    result = verify("There are 12 constraints.", _report())
    assert not result.ok
    assert "12" in result.unmatched


def test_reported_count_cannot_ground_an_unrelated_field() -> None:
    result = verify("The Sharpe ratio is 5.", _report())
    assert not result.ok
    assert "5" in result.unmatched


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
        "Two constraints bind: the budget (sensitivity 0.056272) and "
        "the AAA position cap (sensitivity 0.026418). 4 of 5 names hold weight."
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
