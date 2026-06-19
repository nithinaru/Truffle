"""Slice 5 parse tests: the parser accepts cardinality vocabulary (mock client).

No live API calls. Canned tool responses map "no more than 15 holdings" /
"at most 10 names" to a Cardinality node and assert the schema round-trips.
"""

from __future__ import annotations

from typing import Any

from agent.parse import parse_user_message
from agent.schema import FreshSpec, SpecPatch
from core.ir import Cardinality, MinVariance, PortfolioSpec

_META = {"tickers": ["AAA", "BBB", "CCC", "DDD", "EEE"]}
_BASE = PortfolioSpec(universe=["AAA", "BBB", "CCC", "DDD", "EEE"], objective=MinVariance())


class FakeClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    def call_tool(self, **_kwargs):
        return self._response

    def call_text(self, **_kwargs):
        raise NotImplementedError


def test_parse_no_more_than_15_holdings_patch() -> None:
    client = FakeClient(
        {"result": {"kind": "spec_patch", "add_constraints": [
            {"kind": "cardinality", "id": "card", "max_names": 15}
        ]}}
    )
    result = parse_user_message(
        "no more than 15 holdings", client=client, current_spec=_BASE, universe_metadata=_META
    )
    assert isinstance(result, SpecPatch)
    node = result.add_constraints[0]
    assert isinstance(node, Cardinality)
    assert node.max_names == 15


def test_parse_at_most_10_names_fresh_spec() -> None:
    wide = [f"T{i:02d}" for i in range(12)]
    client = FakeClient(
        {"result": {"kind": "fresh_spec", "spec": {
            "universe": wide,
            "objective": {"kind": "min_variance"},
            "constraints": [
                {"kind": "budget"},
                {"kind": "long_only"},
                {"kind": "cardinality", "id": "card", "max_names": 10},
            ],
        }}}
    )
    result = parse_user_message(
        "minimize variance, long only, fully invested, at most 10 names",
        client=client,
        universe_metadata={"tickers": wide},
    )
    assert isinstance(result, FreshSpec)
    cards = [c for c in result.spec.constraints if isinstance(c, Cardinality)]
    assert len(cards) == 1 and cards[0].max_names == 10


def test_parse_cardinality_with_min_position() -> None:
    client = FakeClient(
        {"result": {"kind": "spec_patch", "add_constraints": [
            {"kind": "cardinality", "id": "card", "max_names": 4, "min_position": 0.05}
        ]}}
    )
    result = parse_user_message(
        "at most 4 names and nothing under 5%",
        client=client,
        current_spec=_BASE,
        universe_metadata=_META,
    )
    assert isinstance(result, SpecPatch)
    node = result.add_constraints[0]
    assert isinstance(node, Cardinality)
    assert node.max_names == 4
    assert node.min_position == 0.05
