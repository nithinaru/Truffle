"""Slice 5 tests: the parser accepts the Sprint-3 vocabulary (mock client only).

No live API calls. Each test feeds a canned tool response that maps a natural
phrase to a new node and asserts the parse path returns the correctly-typed IR,
i.e. the schema round-trips every node the updated prompt instructs the model to
emit. Also covers the clarify-when-ambiguous policy for tracking error.
"""

from __future__ import annotations

from typing import Any

from agent.parse import parse_user_message
from agent.schema import Clarification, FreshSpec, SpecPatch
from core.ir import (
    CVaRLimit,
    FactorExposure,
    GroupCap,
    MaxSharpe,
    MinTrackingError,
    MinVariance,
    PortfolioSpec,
    RiskParity,
    TrackingErrorCap,
    TransactionCost,
    TurnoverCap,
)


class FakeClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    def call_tool(self, **_kwargs):
        return self._response

    def call_text(self, **_kwargs):
        raise NotImplementedError


def _fresh(spec: dict[str, Any]) -> FakeClient:
    return FakeClient({"result": {"kind": "fresh_spec", "spec": spec}})


def _patch(**fields: Any) -> FakeClient:
    return FakeClient({"result": {"kind": "spec_patch", **fields}})


_META = {"tickers": ["AAA", "BBB", "CCC", "DDD"]}
_BASE = PortfolioSpec(universe=["AAA", "BBB", "CCC", "DDD"], objective=MinVariance())


def test_parse_group_cap_fresh_spec() -> None:
    client = _fresh(
        {
            "universe": ["AAA", "BBB", "CCC", "DDD"],
            "objective": {"kind": "min_variance"},
            "constraints": [
                {"kind": "budget"},
                {"kind": "long_only"},
                {"kind": "group_cap", "id": "tech", "group": "Tech", "max_weight": 0.25},
            ],
        }
    )
    result = parse_user_message("cap tech at 25%", client=client, universe_metadata=_META)
    assert isinstance(result, FreshSpec)
    assert any(isinstance(c, GroupCap) and c.group == "Tech" for c in result.spec.constraints)


def test_parse_turnover_cap_patch() -> None:
    client = _patch(add_constraints=[{"kind": "turnover_cap", "id": "to", "max_turnover": 0.20}])
    result = parse_user_message(
        "keep monthly turnover under 20%", client=client, current_spec=_BASE, universe_metadata=_META
    )
    assert isinstance(result, SpecPatch)
    assert isinstance(result.add_constraints[0], TurnoverCap)


def test_parse_transaction_cost_patch() -> None:
    client = _patch(add_constraints=[{"kind": "transaction_cost", "id": "tx", "bps": 10.0}])
    result = parse_user_message(
        "I'll pay 10bps to trade", client=client, current_spec=_BASE, universe_metadata=_META
    )
    assert isinstance(result, SpecPatch)
    assert isinstance(result.add_constraints[0], TransactionCost)


def test_parse_cvar_limit_patch() -> None:
    client = _patch(
        add_constraints=[{"kind": "cvar_limit", "id": "cv", "alpha": 0.95, "max_cvar": 0.03}]
    )
    result = parse_user_message(
        "keep CVaR under 3%", client=client, current_spec=_BASE, universe_metadata=_META
    )
    assert isinstance(result, SpecPatch)
    assert isinstance(result.add_constraints[0], CVaRLimit)


def test_parse_tracking_error_ambiguous_clarification() -> None:
    client = FakeClient(
        {
            "result": {
                "kind": "clarification",
                "question": "Minimize tracking error to SP500, or cap it at 4%?",
                "reason": "other",
            }
        }
    )
    result = parse_user_message(
        "track the S&P within 4% tracking error", client=client, universe_metadata=_META
    )
    assert isinstance(result, Clarification)


def test_parse_tracking_error_cap_patch() -> None:
    client = _patch(
        add_constraints=[
            {"kind": "tracking_error_cap", "id": "te", "benchmark": "SP500", "max_te": 0.04}
        ]
    )
    result = parse_user_message(
        "cap tracking error vs SP500 at 4%", client=client, current_spec=_BASE, universe_metadata=_META
    )
    assert isinstance(result, SpecPatch)
    assert isinstance(result.add_constraints[0], TrackingErrorCap)


def test_parse_factor_exposure_patch() -> None:
    client = _patch(
        add_constraints=[
            {"kind": "factor_exposure", "id": "fx", "factor": "value", "max_exposure": 0.2}
        ]
    )
    result = parse_user_message(
        "limit my value-factor exposure", client=client, current_spec=_BASE, universe_metadata=_META
    )
    assert isinstance(result, SpecPatch)
    assert isinstance(result.add_constraints[0], FactorExposure)


def test_parse_risk_parity_objective() -> None:
    client = _fresh(
        {
            "universe": ["AAA", "BBB", "CCC", "DDD"],
            "objective": {"kind": "risk_parity"},
            "constraints": [{"kind": "long_only"}],
        }
    )
    result = parse_user_message(
        "equal risk contribution from each holding", client=client, universe_metadata=_META
    )
    assert isinstance(result, FreshSpec)
    assert isinstance(result.spec.objective, RiskParity)


def test_parse_max_sharpe_objective() -> None:
    client = _fresh(
        {
            "universe": ["AAA", "BBB", "CCC", "DDD"],
            "objective": {"kind": "max_sharpe", "risk_free_rate": 0.0},
            "constraints": [{"kind": "budget"}, {"kind": "long_only"}],
        }
    )
    result = parse_user_message(
        "maximize risk-adjusted return", client=client, universe_metadata=_META
    )
    assert isinstance(result, FreshSpec)
    assert isinstance(result.spec.objective, MaxSharpe)


def test_parse_min_tracking_error_objective() -> None:
    client = _fresh(
        {
            "universe": ["AAA", "BBB", "CCC", "DDD"],
            "objective": {"kind": "min_tracking_error", "benchmark": "SP500"},
            "constraints": [{"kind": "budget"}, {"kind": "long_only"}],
        }
    )
    result = parse_user_message(
        "track the S&P as closely as possible", client=client, universe_metadata=_META
    )
    assert isinstance(result, FreshSpec)
    assert isinstance(result.spec.objective, MinTrackingError)
