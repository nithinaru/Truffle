"""Offline-first natural-language parse evaluation."""

from evaluation.io import EvaluationDataError, load_benchmark, load_predictions, report_json
from evaluation.models import (
    AccuracyMetric,
    BenchmarkCase,
    CaseResult,
    EvaluationAggregate,
    EvaluationError,
    EvaluationReport,
    PrecisionRecallF1,
    PredictionRecord,
)
from evaluation.normalize import constraint_multiset, normalize_parse_result
from evaluation.score import evaluate_with_client, score_predictions

__all__ = [
    "AccuracyMetric",
    "BenchmarkCase",
    "CaseResult",
    "EvaluationAggregate",
    "EvaluationDataError",
    "EvaluationError",
    "EvaluationReport",
    "PrecisionRecallF1",
    "PredictionRecord",
    "constraint_multiset",
    "evaluate_with_client",
    "load_benchmark",
    "load_predictions",
    "normalize_parse_result",
    "report_json",
    "score_predictions",
]
