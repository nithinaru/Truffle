"""Tests for agent/schema.py: ParseResult dispatch and patch application."""

from __future__ import annotations

import pytest

from agent.schema import (
    Clarification,
    FreshSpec,
    ParseEnvelope,
    SpecPatch,
    apply_patch,
)
from core.ir import (
    Box,
    Budget,
    LongOnly,
    MeanVariance,
    MinCVaR,
    MinVariance,
    PortfolioSpec,
)


def _base_spec() -> PortfolioSpec:
    return PortfolioSpec(
        universe=["AAA", "BBB", "CCC"],
        objective=MinVariance(),
        constraints=[
            Budget(id="b"),
            LongOnly(id="lo"),
            Box(id="cap_aaa", lower=0.0, upper=0.5, tickers=["AAA"]),
        ],
    )


def test_envelope_dispatches_clarification() -> None:
    env = ParseEnvelope.model_validate(
        {
            "result": {
                "kind": "clarification",
                "question": "What do you mean by 'not too concentrated'?",
                "reason": "vague_quantity",
            }
        }
    )
    assert isinstance(env.result, Clarification)
    assert env.result.partial_spec is None


def test_envelope_dispatches_fresh_spec() -> None:
    env = ParseEnvelope.model_validate(
        {
            "result": {
                "kind": "fresh_spec",
                "spec": {
                    "universe": ["AAPL", "MSFT"],
                    "objective": {"kind": "min_variance"},
                    "constraints": [{"kind": "budget"}, {"kind": "long_only"}],
                },
            }
        }
    )
    assert isinstance(env.result, FreshSpec)
    assert env.result.spec.universe == ["AAPL", "MSFT"]


def test_envelope_dispatches_spec_patch() -> None:
    env = ParseEnvelope.model_validate(
        {
            "result": {
                "kind": "spec_patch",
                "remove_constraint_ids": ["cap_aaa"],
                "replace_objective": {"kind": "min_cvar", "cvar_alpha": 0.9},
            }
        }
    )
    assert isinstance(env.result, SpecPatch)
    assert env.result.remove_constraint_ids == ["cap_aaa"]
    assert isinstance(env.result.replace_objective, MinCVaR)


def test_empty_spec_patch_rejected() -> None:
    with pytest.raises(ValueError, match="SpecPatch is empty"):
        SpecPatch()


def test_apply_patch_removes_then_replaces_then_adds() -> None:
    spec = _base_spec()
    patch = SpecPatch(
        remove_constraint_ids=["cap_aaa"],
        replace_objective=MeanVariance(risk_aversion=2.0),
        add_constraints=[Box(id="cap_all", lower=0.0, upper=0.4)],
    )
    new_spec = apply_patch(spec, patch)
    assert isinstance(new_spec.objective, MeanVariance)
    ids = [c.id for c in new_spec.constraints]
    assert "cap_aaa" not in ids
    assert "cap_all" in ids
    assert "b" in ids and "lo" in ids


def test_apply_patch_universe_swap() -> None:
    spec = _base_spec()
    # Drop the box-on-AAA first; otherwise the validator will complain that
    # AAA is no longer in the universe.
    patch = SpecPatch(
        remove_constraint_ids=["cap_aaa"],
        set_universe=["XXX", "YYY"],
    )
    new_spec = apply_patch(spec, patch)
    assert new_spec.universe == ["XXX", "YYY"]


def test_clarification_rejects_empty_question() -> None:
    with pytest.raises(ValueError):
        Clarification(question="", reason="vague_quantity")


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValueError):
        ParseEnvelope.model_validate(
            {"result": {"kind": "clarification", "question": "?", "reason": "x", "extra": 1}}
        )
