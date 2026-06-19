"""Slice 4 tests: mixed-integer UX in the spec echo and chat loop.

The echo must warn — before solving — that the problem is mixed-integer (and
which constraint caused it), that the solve is slower, and that shadow prices
will be conditional. A universe-size guard adds a time-limit warning. The
continuous UX must be untouched.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from agent.loop import (
    MIP_TIME_LIMIT_S,
    MIP_UNIVERSE_GUARD,
    ChatSession,
    _mip_guard,
)
from agent.render import render_spec
from agent.schema import ParseEnvelope
from core.ir import Budget, Cardinality, LongOnly, MinVariance, PortfolioSpec

EXAMPLES = Path(__file__).parent.parent / "examples"
UNIVERSE = ["AAA", "BBB", "CCC", "DDD", "EEE"]


@pytest.fixture
def prices() -> pd.DataFrame:
    return pd.read_csv(EXAMPLES / "prices_sample.csv", parse_dates=[0], index_col=0)


class FakeClient:
    def __init__(self, *, tool: list[dict], text: list[str]) -> None:
        self._tool = list(tool)
        self._text = list(text)

    def call_tool(self, **_kwargs):
        return self._tool.pop(0)

    def call_text(self, **_kwargs):
        return self._text.pop(0)


def _card_fresh_payload() -> dict:
    return {
        "result": {
            "kind": "fresh_spec",
            "spec": {
                "universe": UNIVERSE,
                "objective": {"kind": "min_cvar", "cvar_alpha": 0.95},
                "constraints": [
                    {"kind": "budget"},
                    {"kind": "long_only"},
                    {"kind": "cardinality", "id": "card", "max_names": 3},
                ],
            },
        }
    }


def test_render_spec_warns_on_mip() -> None:
    spec = PortfolioSpec(
        universe=UNIVERSE,
        objective=MinVariance(),
        constraints=[Budget(), LongOnly(), Cardinality(max_names=2)],
    )
    text = render_spec(spec)
    assert "Problem class: MIP" in text
    assert "MIXED-INTEGER" in text
    assert "cardinality" in text.lower()
    assert "conditional" in text.lower()


def test_render_spec_continuous_has_no_mip_warning() -> None:
    spec = PortfolioSpec(
        universe=UNIVERSE, objective=MinVariance(), constraints=[Budget(), LongOnly()]
    )
    text = render_spec(spec)
    assert "Problem class: CONVEX" in text
    assert "MIXED-INTEGER" not in text


def test_mip_guard_thresholds() -> None:
    small = PortfolioSpec(
        universe=["A", "B", "C"],
        objective=MinVariance(),
        constraints=[Cardinality(max_names=2)],
    )
    assert _mip_guard(small) == (None, None)

    big_universe = [f"T{i}" for i in range(MIP_UNIVERSE_GUARD)]
    big = PortfolioSpec(
        universe=big_universe,
        objective=MinVariance(),
        constraints=[Cardinality(max_names=5)],
    )
    time_limit, warn = _mip_guard(big)
    assert time_limit == MIP_TIME_LIMIT_S
    assert warn is not None and "gap" in warn

    # Continuous spec, even with a big universe, gets no guard.
    cont = PortfolioSpec(
        universe=big_universe, objective=MinVariance(), constraints=[Budget()]
    )
    assert _mip_guard(cont) == (None, None)


def test_echo_includes_mip_advisory(prices: pd.DataFrame) -> None:
    client = FakeClient(tool=[_card_fresh_payload()], text=[])
    session = ChatSession(client=client, prices=prices)
    res = session.handle_user_message("min CVaR, at most 3 names")
    assert res.kind == "echo"
    assert "MIXED-INTEGER" in res.text
    assert "conditional" in res.text.lower()


def test_mip_solve_through_loop_surfaces_conditional(prices: pd.DataFrame) -> None:
    # cardinality + min-CVaR => MILP (HiGHS, always available).
    from core.solve import solve_spec  # noqa: PLC0415

    payload = _card_fresh_payload()
    spec = ParseEnvelope.model_validate(payload).result.spec
    _, rep = solve_spec(spec, prices)
    narration = (
        f"Mixed-integer solve: {rep.nonzero_names} names were selected and the "
        "shadow prices are conditional, holding the selected names fixed. The "
        f"minimum-CVaR objective reached {rep.objective_value:.6f}."
    )
    client = FakeClient(tool=[payload], text=[narration])
    session = ChatSession(client=client, prices=prices)
    session.handle_user_message("min CVaR, at most 3 names")
    out = session.confirm_pending("y")

    assert out.kind == "solved"
    assert out.report is not None
    assert out.report.duals_conditional is True
    assert out.report.selected_names is not None
    assert out.report.nonzero_names <= 3
    assert out.report.optimality_gap is not None
    assert "conditional" in out.explanation.lower()
