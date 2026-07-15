"""Offline parse-benchmark tests; no production client is ever constructed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agent.schema import Clarification, FreshSpec, SpecPatch
from core.ir import Box, Budget, LongOnly, MeanVariance, MinVariance, PortfolioSpec
from evaluation import (
    BenchmarkCase,
    EvaluationDataError,
    evaluate_with_client,
    load_benchmark,
    load_predictions,
    normalize_parse_result,
    report_json,
    score_predictions,
)
from evaluation.run import main

ROOT = Path(__file__).parents[1]
STARTER_BENCHMARK = ROOT / "evaluation" / "benchmark.jsonl"


def _spec(
    *,
    universe: list[str] | None = None,
    objective: MinVariance | MeanVariance | None = None,
    constraints: list[Budget | LongOnly | Box] | None = None,
) -> PortfolioSpec:
    return PortfolioSpec(
        universe=universe or ["AAA", "BBB", "CCC"],
        objective=objective or MinVariance(),
        constraints=constraints or [Budget(id="budget"), LongOnly(id="long")],
    )


def _case(case_id: str, expected, *, category: str = "fresh_spec") -> BenchmarkCase:
    return BenchmarkCase.model_validate(
        {
            "case_id": case_id,
            "category": category,
            "user_text": f"prompt for {case_id}",
            "universe_metadata": {"tickers": ["AAA", "BBB", "CCC"]},
            "expected": expected.model_dump(mode="json"),
        }
    )


class CannedClient:
    """A deliberately network-incapable client with ordered raw responses."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.tool_calls = 0
        self.text_calls = 0

    def call_tool(self, **_kwargs: Any) -> dict[str, Any]:
        self.tool_calls += 1
        if not self.responses:
            raise AssertionError("No canned response remains; a live fallback is forbidden.")
        return self.responses.pop(0)

    def call_text(self, **_kwargs: Any) -> str:
        self.text_calls += 1
        raise AssertionError("The parse evaluator must never request free-form or live text.")


def test_starter_benchmark_is_modest_labeled_and_fully_validated() -> None:
    cases = load_benchmark(STARTER_BENCHMARK)
    assert len(cases) == 12
    assert {case.suite for case in cases} == {"starter_v1"}
    assert {case.category for case in cases} == {
        "fresh_spec",
        "patch",
        "ambiguity",
        "adversarial",
        "infeasibility_prone",
    }
    assert all(case.expected.kind in {"fresh_spec", "spec_patch", "clarification"} for case in cases)


def test_benchmark_models_are_frozen() -> None:
    case = load_benchmark(STARTER_BENCHMARK)[0]
    with pytest.raises(ValidationError, match="frozen"):
        case.case_id = "changed"  # type: ignore[misc]


def test_loader_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    row = STARTER_BENCHMARK.read_text(encoding="utf-8").splitlines()[0]
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(f"{row}\n{row}\n", encoding="utf-8")
    with pytest.raises(EvaluationDataError, match="duplicate case_id"):
        load_benchmark(duplicate)


def test_prediction_loader_validates_results_and_unique_ids(tmp_path: Path) -> None:
    row = {
        "case_id": "one",
        "result": {"kind": "clarification", "question": "Which cap?", "reason": "other"},
    }
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    loaded = load_predictions(path)
    assert isinstance(loaded["one"].result, Clarification)

    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(EvaluationDataError, match="duplicate case_id"):
        load_predictions(path)


def test_semantic_exact_match_ignores_constraint_ids_and_order() -> None:
    expected = FreshSpec(
        spec=_spec(
            constraints=[Budget(id="expected_budget"), LongOnly(id="expected_long")]
        )
    )
    actual = FreshSpec(
        spec=_spec(constraints=[LongOnly(id="actual_long"), Budget(id="actual_budget")])
    )
    report = score_predictions([_case("same", expected)], {"same": actual})
    assert report.cases[0].semantic_exact_match is True
    assert report.aggregate.semantic_exact_match.accuracy == 1.0
    assert report.aggregate.constraints_micro.f1 == 1.0


def test_semantic_exact_match_keeps_universe_objective_and_constraint_fields() -> None:
    expected = FreshSpec(
        spec=_spec(
            universe=["AAA", "BBB", "CCC"],
            objective=MeanVariance(risk_aversion=2.0),
            constraints=[
                Budget(id="b1"),
                LongOnly(id="l1"),
                Box(id="cap1", lower=0.0, upper=0.3),
            ],
        )
    )
    actual = FreshSpec(
        spec=_spec(
            universe=["BBB", "AAA", "CCC"],
            objective=MeanVariance(risk_aversion=3.0),
            constraints=[
                Budget(id="b2"),
                LongOnly(id="l2"),
                Box(id="cap2", lower=0.0, upper=0.4),
            ],
        )
    )
    result = score_predictions([_case("different", expected)], {"different": actual}).cases[0]
    assert result.semantic_exact_match is False
    paths = {error.path for error in result.errors}
    assert "$.spec.universe[0]" in paths
    assert "$.spec.objective.risk_aversion" in paths
    assert any(path.endswith(".upper") for path in paths)


def test_patch_removal_compares_the_constraint_not_generated_id() -> None:
    current = _spec(
        constraints=[
            Budget(id="budget"),
            LongOnly(id="long"),
            Box(id="cap", lower=0.0, upper=0.4),
        ]
    )
    expected = SpecPatch(remove_constraint_ids=["cap"])
    case = BenchmarkCase(
        case_id="remove",
        category="patch",
        user_text="remove cap",
        current_spec=current,
        expected=expected,
    )
    normalized = normalize_parse_result(expected, current_spec=current)
    assert normalized["remove_constraints"] == [
        {
            "constraint": {
                "kind": "box",
                "elastic": None,
                "lower": 0.0,
                "upper": 0.4,
                "tickers": None,
            }
        }
    ]
    report = score_predictions([case], {"remove": SpecPatch(remove_constraint_ids=["cap"])})
    assert report.cases[0].semantic_exact_match
    assert report.aggregate.constraints_micro.true_positive == 1


def test_constraint_multiset_micro_counts_substantive_misses() -> None:
    expected = FreshSpec(
        spec=_spec(constraints=[Budget(id="b1"), LongOnly(id="l1")])
    )
    actual = FreshSpec(
        spec=_spec(
            constraints=[Budget(id="b2"), Box(id="cap", lower=0.0, upper=0.5)]
        )
    )
    metric = score_predictions([_case("micro", expected)], {"micro": actual}).aggregate.constraints_micro
    assert (metric.true_positive, metric.false_positive, metric.false_negative) == (1, 1, 1)
    assert metric.precision == metric.recall == metric.f1 == 0.5


def test_parse_kind_and_clarification_metrics_are_separate() -> None:
    clarification = Clarification(question="Which cap?", reason="vague_quantity")
    fresh = FreshSpec(spec=_spec())
    cases = [
        _case("expected_clarification", clarification, category="ambiguity"),
        _case("expected_fresh", fresh),
    ]
    predictions = {"expected_clarification": fresh, "expected_fresh": clarification}
    aggregate = score_predictions(cases, predictions).aggregate
    assert aggregate.parse_kind.accuracy == 0.0
    assert (
        aggregate.clarification.true_positive,
        aggregate.clarification.false_positive,
        aggregate.clarification.false_negative,
    ) == (0, 1, 1)
    assert aggregate.clarification.f1 == 0.0


def test_missing_prediction_stays_in_denominators_with_case_error() -> None:
    expected = FreshSpec(spec=_spec())
    report = score_predictions([_case("missing", expected)], {})
    assert report.aggregate.prediction_count == 0
    assert report.aggregate.failed_prediction_count == 1
    assert report.aggregate.parse_kind.accuracy == 0.0
    assert report.cases[0].errors[0].code == "missing_prediction"
    assert report.aggregate.constraints_micro.false_negative == 2


def test_injected_fake_client_runs_existing_parser_without_live_calls() -> None:
    expected = FreshSpec(spec=_spec())
    case = _case("canned", expected)
    raw = {"result": expected.model_dump(mode="json")}
    client = CannedClient([raw])
    report = evaluate_with_client([case], client=client)
    assert report.cases[0].semantic_exact_match
    assert client.tool_calls == 1
    assert client.text_calls == 0
    assert client.responses == []


def test_injected_client_failure_is_reported_not_dropped() -> None:
    case = _case("failure", FreshSpec(spec=_spec()))
    client = CannedClient([])
    report = evaluate_with_client([case], client=client)
    assert report.aggregate.failed_prediction_count == 1
    assert report.cases[0].errors[0].code == "parse_exception"
    assert "AssertionError" in report.cases[0].errors[0].message


def test_report_json_is_deterministic_and_has_no_wall_clock_fields() -> None:
    expected = FreshSpec(spec=_spec())
    report = score_predictions([_case("stable", expected)], {"stable": expected})
    first = report_json(report)
    second = report_json(report)
    assert first == second
    assert "timestamp" not in first
    assert "duration" not in first
    assert "elapsed" not in first


def test_offline_module_entrypoint_scores_prediction_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = _case("offline", FreshSpec(spec=_spec()))
    benchmark_path = tmp_path / "benchmark.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    benchmark_path.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    prediction_path.write_text(
        json.dumps(
            {
                "case_id": case.case_id,
                "result": case.expected.model_dump(mode="json"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["--benchmark", str(benchmark_path), "--predictions", str(prediction_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["aggregate"]["semantic_exact_match"]["accuracy"] == 1.0
    assert "created_at" not in payload
