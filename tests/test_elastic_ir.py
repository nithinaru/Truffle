"""IR metadata used by Sprint 5 elastic infeasibility diagnosis."""

from __future__ import annotations

import pytest

from core.ir import (
    Box,
    Budget,
    Cardinality,
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


def test_elastic_field_round_trips_through_constraint_union() -> None:
    spec = PortfolioSpec.model_validate(
        {
            "universe": ["AAA", "BBB"],
            "objective": {"kind": "min_variance"},
            "constraints": [
                {"kind": "budget"},
                {"kind": "box", "lower": 0.0, "upper": 0.8, "elastic": False},
            ],
        }
    )

    assert spec.constraints[0].elastic is None
    assert spec.constraints[1].elastic is False
    assert spec.constraints[1].is_elastic is False
    assert spec.constraints[1].effective_elastic is False
    assert spec.model_dump()["constraints"][1]["elastic"] is False


def test_objective_schema_does_not_gain_elasticity() -> None:
    assert "elastic" in Budget.model_json_schema()["properties"]
    assert "elastic" not in MinVariance.model_json_schema()["properties"]
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        MinVariance.model_validate({"kind": "min_variance", "elastic": False})


@pytest.mark.parametrize("node", [Budget(), LongOnly(), TransactionCost(bps=5.0)])
def test_never_relaxable_nodes_reject_elastic_true(node: object) -> None:
    node_type = type(node)
    payload = node.model_dump()  # type: ignore[attr-defined]
    payload["elastic"] = True
    with pytest.raises(ValueError, match="never elastic"):
        node_type.model_validate(payload)


@pytest.mark.parametrize("node", [Budget(elastic=False), LongOnly(elastic=False)])
def test_structural_constraint_false_override_remains_hard(node: object) -> None:
    assert node.is_elastic is False  # type: ignore[attr-defined]
    assert node.effective_elastic is False  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("node", "raw_scale", "big_m"),
    [
        (Box(lower=-0.25, upper=0.4), 0.4, 1.0),
        (GroupCap(group="Tech", max_weight=0.3), 0.3, 1.0),
        (TurnoverCap(max_turnover=0.2), 0.2, 2.0),
        (CVaRLimit(alpha=0.95, max_cvar=-0.04), 0.04, None),
        (TrackingErrorCap(benchmark="SPX", max_te=0.05), 0.05, None),
        (FactorExposure(factor="value", min_exposure=-0.2, max_exposure=0.3), 0.5, None),
        (Cardinality(max_names=7), 7.0, None),
    ],
)
def test_relaxable_constraint_metadata(node: object, raw_scale: float, big_m: float | None) -> None:
    assert node.elastic_default is True  # type: ignore[attr-defined]
    assert node.is_elastic is True  # type: ignore[attr-defined]
    assert node.effective_elastic is True  # type: ignore[attr-defined]
    assert node.slack_scale == pytest.approx(raw_scale)  # type: ignore[attr-defined]
    assert node.natural_slack_scale() == pytest.approx(raw_scale)  # type: ignore[attr-defined]
    assert node.big_m == big_m  # type: ignore[attr-defined]


def test_scale_helper_has_positive_fallback_for_degenerate_bound() -> None:
    node = FactorExposure(factor="market", min_exposure=0.0, max_exposure=0.0)
    assert node.slack_scale == 0.0
    assert node.natural_slack_scale() == 1.0


def test_cardinality_diagnostic_big_m_uses_universe_size() -> None:
    node = Cardinality(max_names=3)
    assert node.diagnostic_big_m(universe_size=12) == 12.0
    with pytest.raises(ValueError, match="positive universe_size"):
        node.diagnostic_big_m()


@pytest.mark.parametrize(
    "node",
    [
        Cardinality(max_names=3, min_names=2),
        Cardinality(max_names=3, min_position=0.1),
    ],
)
def test_cardinality_lower_rows_make_the_instance_structural(node: Cardinality) -> None:
    assert node.elasticity_supported is False
    assert node.is_elastic is False
    assert node.effective_elastic is False


def test_cardinality_lower_rows_reject_explicit_elastic_true() -> None:
    with pytest.raises(ValueError, match="max_names-only"):
        Cardinality(max_names=3, min_names=2, elastic=True)
