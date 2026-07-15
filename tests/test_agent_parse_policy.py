"""Offline contract tests for parser handling of execution-workflow language."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.parse import _load_system_prompt
from agent.schema import Clarification, FreshSpec, ParseEnvelope


def _minimum_variance_spec(*, constraints: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "universe": ["AAA", "BBB"],
        "objective": {"kind": "min_variance"},
        "constraints": constraints
        if constraints is not None
        else [{"kind": "budget"}, {"kind": "long_only"}],
    }


def test_prompt_separates_portfolio_ir_from_execution_workflows() -> None:
    prompt = _load_system_prompt()

    assert "Historical walk-forward backtesting" in prompt
    assert "deterministic local paper replay" in prompt
    assert "outside the single-period portfolio IR" in prompt
    assert "both a complete portfolio-construction request" in prompt
    assert "return the complete `fresh_spec` or `spec_patch`" in prompt
    assert "live-shadow workflow" in prompt
    assert "must never submit an order" in prompt
    assert "Broker-hosted paper trading requires a separate, explicit" in prompt
    assert "Real-money order submission is unsupported" in prompt


def test_prompt_forbids_operational_result_and_constraint_kinds() -> None:
    prompt = _load_system_prompt()

    assert "Exactly one of:" in prompt
    assert "`fresh_spec`" in prompt
    assert "`spec_patch`" in prompt
    assert "`clarification`" in prompt
    for forbidden_kind in (
        "`backtest`",
        "`paper_trade`",
        "`live_shadow`",
        "`broker_order`",
        "`real_money`",
    ):
        assert forbidden_kind in prompt
    assert "Never invent operational schema" in prompt
    assert "not in the IR yet" not in prompt
    assert "this sprint" not in prompt


def test_combined_construction_and_backtest_stays_a_fresh_spec() -> None:
    canned = {
        "result": {
            "kind": "fresh_spec",
            "spec": _minimum_variance_spec(),
        }
    }

    envelope = ParseEnvelope.model_validate(canned)

    assert isinstance(envelope.result, FreshSpec)
    assert envelope.result.spec.universe == ["AAA", "BBB"]


def test_pure_live_shadow_request_can_only_be_a_clarification() -> None:
    canned = {
        "result": {
            "kind": "clarification",
            "question": (
                "Should I route the confirmed spec to the separate "
                "non-submitting live-shadow workflow?"
            ),
            "reason": "unsupported_feature",
        }
    }

    envelope = ParseEnvelope.model_validate(canned)

    assert isinstance(envelope.result, Clarification)
    assert envelope.result.reason == "unsupported_feature"


@pytest.mark.parametrize("kind", ["backtest", "paper_trade", "live_shadow", "broker_order"])
def test_parse_envelope_rejects_invented_operational_result_kinds(kind: str) -> None:
    with pytest.raises(ValidationError):
        ParseEnvelope.model_validate({"result": {"kind": kind}})


@pytest.mark.parametrize("kind", ["backtest", "paper_trade", "live_shadow", "broker_order"])
def test_parse_envelope_rejects_invented_operational_constraints(kind: str) -> None:
    canned = {
        "result": {
            "kind": "fresh_spec",
            "spec": _minimum_variance_spec(constraints=[{"kind": kind}]),
        }
    }

    with pytest.raises(ValidationError):
        ParseEnvelope.model_validate(canned)


def test_parse_envelope_rejects_workflow_configuration_on_fresh_spec() -> None:
    canned = {
        "result": {
            "kind": "fresh_spec",
            "spec": _minimum_variance_spec(),
            "backtest": {"rebalance_frequency": "monthly"},
        }
    }

    with pytest.raises(ValidationError):
        ParseEnvelope.model_validate(canned)
