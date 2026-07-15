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
from core.patch import SpecPatch as CoreSpecPatch
from core.patch import apply_patch as core_apply_patch


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


def test_agent_schema_reexports_core_patch_api() -> None:
    assert SpecPatch is CoreSpecPatch
    assert apply_patch is core_apply_patch


def test_empty_spec_patch_rejected() -> None:
    with pytest.raises(ValueError, match="SpecPatch is empty"):
        SpecPatch()


def test_apply_patch_removes_then_replaces_then_adds() -> None:
    spec = _base_spec()
    patch = SpecPatch(
        remove_constraint_ids=["cap_aaa"],
        replace_objective=MeanVariance(risk_aversion=2.0),
        add_constraints=[Box(id="cap_aaa", lower=0.0, upper=0.4)],
    )
    new_spec = apply_patch(spec, patch)
    assert isinstance(new_spec.objective, MeanVariance)
    assert [constraint.id for constraint in new_spec.constraints] == ["b", "lo", "cap_aaa"]
    assert isinstance(new_spec.constraints[-1], Box)
    assert new_spec.constraints[-1].upper == 0.4


def test_apply_patch_preserves_current_weights() -> None:
    spec = PortfolioSpec(
        universe=["AAA", "BBB", "CCC"],
        objective=MinVariance(),
        constraints=[Budget(id="b"), LongOnly(id="lo")],
        current_weights={"AAA": 0.6, "BBB": 0.3, "CCC": 0.1},
    )

    new_spec = apply_patch(
        spec,
        SpecPatch(replace_objective=MeanVariance(risk_aversion=2.0)),
    )

    assert new_spec.current_weights == spec.current_weights


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


def test_apply_patch_universe_swap_liquidates_removed_holdings() -> None:
    spec = PortfolioSpec(
        universe=["AAA", "BBB", "CCC"],
        objective=MinVariance(),
        constraints=[Budget(id="b"), LongOnly(id="lo")],
        current_weights={"AAA": 0.6, "BBB": 0.3, "CCC": 0.1},
    )

    new_spec = apply_patch(spec, SpecPatch(set_universe=["BBB", "DDD"]))

    assert new_spec.universe == ["BBB", "DDD"]
    assert new_spec.current_weights == {"BBB": 0.3}
    assert new_spec.w_prev_vector() == [0.3, 0.0]


def test_clarification_rejects_empty_question() -> None:
    with pytest.raises(ValueError):
        Clarification(question="", reason="vague_quantity")


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValueError):
        ParseEnvelope.model_validate(
            {"result": {"kind": "clarification", "question": "?", "reason": "x", "extra": 1}}
        )
