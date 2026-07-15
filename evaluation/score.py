"""Offline scoring and injected-client execution for parse benchmarks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from agent.client import LLMClient
from agent.parse import parse_user_message
from agent.schema import Clarification, ParseResult
from evaluation.models import (
    AccuracyMetric,
    BenchmarkCase,
    CaseResult,
    EvaluationAggregate,
    EvaluationError,
    EvaluationReport,
    PrecisionRecallF1,
)
from evaluation.normalize import constraint_multiset, normalize_parse_result


def _accuracy(correct: int, total: int) -> AccuracyMetric:
    return AccuracyMetric(correct=correct, total=total, accuracy=correct / total if total else None)


def _prf(true_positive: int, false_positive: int, false_negative: int) -> PrecisionRecallF1:
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative
    return PrecisionRecallF1(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=true_positive / precision_denominator if precision_denominator else None,
        recall=true_positive / recall_denominator if recall_denominator else None,
        f1=2 * true_positive / f1_denominator if f1_denominator else None,
    )


def _diff(expected: Any, actual: Any, path: str = "$") -> list[EvaluationError]:
    errors: list[EvaluationError] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() - actual.keys()):
            errors.append(
                EvaluationError(
                    code="missing_field",
                    path=f"{path}.{key}",
                    expected=expected[key],
                    message="Expected field is absent from the prediction.",
                )
            )
        for key in sorted(actual.keys() - expected.keys()):
            errors.append(
                EvaluationError(
                    code="unexpected_field",
                    path=f"{path}.{key}",
                    actual=actual[key],
                    message="Prediction contains an unexpected semantic field.",
                )
            )
        for key in sorted(expected.keys() & actual.keys()):
            errors.extend(_diff(expected[key], actual[key], f"{path}.{key}"))
        return errors
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            errors.append(
                EvaluationError(
                    code="length_mismatch",
                    path=f"{path}.length",
                    expected=len(expected),
                    actual=len(actual),
                    message="Expected and predicted list lengths differ.",
                )
            )
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=False)):
            errors.extend(_diff(expected_item, actual_item, f"{path}[{index}]"))
        return errors
    if expected != actual:
        errors.append(
            EvaluationError(
                code="value_mismatch",
                path=path,
                expected=expected,
                actual=actual,
                message="Expected and predicted semantic values differ.",
            )
        )
    return errors


def _matched_count(expected: Counter[str], actual: Counter[str]) -> int:
    return sum((expected & actual).values())


def score_predictions(
    cases: Sequence[BenchmarkCase],
    predictions: Mapping[str, ParseResult],
    *,
    failures: Mapping[str, str] | None = None,
) -> EvaluationReport:
    """Score already-validated predictions without creating or calling a client.

    Missing predictions remain in every case-level denominator.  Unexpected
    prediction IDs are rejected so a typo cannot silently inflate or obscure
    the reported coverage.
    """
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Benchmark cases must have unique case IDs.")
    unknown = sorted(set(predictions) - set(case_ids))
    if unknown:
        raise ValueError(f"Predictions contain unknown case IDs: {unknown}.")
    failure_map = dict(failures or {})
    unknown_failures = sorted(set(failure_map) - set(case_ids))
    if unknown_failures:
        raise ValueError(f"Failures contain unknown case IDs: {unknown_failures}.")

    results: list[CaseResult] = []
    kind_correct = 0
    exact_correct = 0
    constraint_tp = 0
    constraint_fp = 0
    constraint_fn = 0
    clarification_tp = 0
    clarification_fp = 0
    clarification_fn = 0

    for case in cases:
        expected = case.expected
        actual = predictions.get(case.case_id)
        normalized_expected = normalize_parse_result(expected, current_spec=case.current_spec)
        expected_constraints = constraint_multiset(expected, current_spec=case.current_spec)
        actual_constraints: Counter[str] = Counter()
        errors: list[EvaluationError]

        if actual is None:
            normalized_actual = None
            failure = failure_map.get(case.case_id)
            errors = [
                EvaluationError(
                    code="parse_exception" if failure is not None else "missing_prediction",
                    path="$",
                    expected=normalized_expected,
                    actual=None,
                    message=failure or "No prediction was supplied for this benchmark case.",
                )
            ]
            actual_kind = None
            semantic_match = False
        else:
            normalized_actual = normalize_parse_result(actual, current_spec=case.current_spec)
            actual_constraints = constraint_multiset(actual, current_spec=case.current_spec)
            errors = _diff(normalized_expected, normalized_actual)
            actual_kind = actual.kind
            semantic_match = not errors

        kind_match = actual_kind == expected.kind
        kind_correct += int(kind_match)
        exact_correct += int(semantic_match)
        matched = _matched_count(expected_constraints, actual_constraints)
        constraint_tp += matched
        constraint_fp += sum(actual_constraints.values()) - matched
        constraint_fn += sum(expected_constraints.values()) - matched

        expected_clarification = isinstance(expected, Clarification)
        actual_clarification = isinstance(actual, Clarification)
        clarification_tp += int(expected_clarification and actual_clarification)
        clarification_fp += int(not expected_clarification and actual_clarification)
        clarification_fn += int(expected_clarification and not actual_clarification)

        results.append(
            CaseResult(
                case_id=case.case_id,
                category=case.category,
                expected_kind=expected.kind,
                actual_kind=actual_kind,
                parse_kind_match=kind_match,
                semantic_exact_match=semantic_match,
                expected_constraint_count=sum(expected_constraints.values()),
                actual_constraint_count=sum(actual_constraints.values()),
                matched_constraint_count=matched,
                expected_normalized=normalized_expected,
                actual_normalized=normalized_actual,
                errors=tuple(errors),
            )
        )

    total = len(cases)
    aggregate = EvaluationAggregate(
        case_count=total,
        prediction_count=len(predictions),
        failed_prediction_count=total - len(predictions),
        semantic_exact_match=_accuracy(exact_correct, total),
        parse_kind=_accuracy(kind_correct, total),
        constraints_micro=_prf(constraint_tp, constraint_fp, constraint_fn),
        clarification=_prf(clarification_tp, clarification_fp, clarification_fn),
    )
    return EvaluationReport(aggregate=aggregate, cases=tuple(results))


def evaluate_with_client(
    cases: Sequence[BenchmarkCase],
    *,
    client: LLMClient,
) -> EvaluationReport:
    """Run each case through ``parse_user_message`` using an injected client.

    The evaluator never constructs a production client and has no API-key or
    network path.  Callers choosing a live implementation must construct and
    inject it explicitly outside this package.
    """
    predictions: dict[str, ParseResult] = {}
    failures: dict[str, str] = {}
    for case in cases:
        try:
            predictions[case.case_id] = parse_user_message(
                case.user_text,
                client=client,
                current_spec=case.current_spec,
                universe_metadata=dict(case.universe_metadata),
            )
        except Exception as exc:  # Evaluation must retain failed cases in denominators.
            # Exception text from remote clients can contain request IDs or timing
            # details.  Retain the stable class without making reports depend on
            # those nondeterministic values.
            failures[case.case_id] = f"parse_user_message raised {type(exc).__name__}."
    return score_predictions(cases, predictions, failures=failures)
