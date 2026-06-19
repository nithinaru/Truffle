"""Slice 2/3 spec-echo coverage: every new node renders deterministically.

The spec echo is the trust handshake (BLUEPRINT §2), so each Sprint-3 node must
appear in render_spec with its id and key numbers. We also load every shipped
example YAML to guarantee the IR accepts them.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agent.render import render_spec
from core.ir import (
    Budget,
    CVaRLimit,
    FactorExposure,
    GroupCap,
    LongOnly,
    MinVariance,
    PortfolioSpec,
    TrackingErrorCap,
    TransactionCost,
    TurnoverCap,
)

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_render_includes_every_new_constraint() -> None:
    spec = PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=[
            Budget(),
            LongOnly(),
            GroupCap(id="gc", group="Tech", max_weight=0.3, min_weight=0.1),
            TurnoverCap(id="tc", max_turnover=0.2),
            TransactionCost(id="tx", bps=10.0),
            CVaRLimit(id="cv", alpha=0.95, max_cvar=0.03),
            TrackingErrorCap(id="te", benchmark="bench", max_te=0.04),
            FactorExposure(id="fx", factor="value", max_exposure=0.15),
        ],
    )
    text = render_spec(spec)
    for token in ["gc", "group cap", "tc", "turnover", "tx", "transaction cost",
                  "cv", "CVaR", "te", "tracking-error", "fx", "factor exposure"]:
        assert token in text, token


def test_all_sprint3_example_specs_load() -> None:
    for name in [
        "spec_group_cap.yaml",
        "spec_turnover_txcost.yaml",
        "spec_cvar_limit.yaml",
        "spec_tracking_error.yaml",
        "spec_factor_exposure.yaml",
    ]:
        payload = yaml.safe_load((_EXAMPLES / name).read_text())
        spec = PortfolioSpec.model_validate(payload)
        assert spec.problem_class == "convex"
        # Each renders without error in the spec echo.
        assert render_spec(spec)
