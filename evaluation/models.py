"""Typed data contracts for the offline natural-language parse benchmark.

The benchmark deliberately stores expected :class:`agent.schema.ParseResult`
objects, not prose labels.  Loading the corpus therefore exercises the same
Pydantic boundary as production parsing before any score is calculated.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from agent.schema import ParseResult
from core.ir import PortfolioSpec

ParseKind = Literal["fresh_spec", "spec_patch", "clarification"]
CaseCategory = Literal[
    "fresh_spec",
    "patch",
    "ambiguity",
    "adversarial",
    "infeasibility_prone",
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BenchmarkCase(_FrozenModel):
    """One fully validated prompt/expectation pair in the starter suite."""

    suite: Literal["starter_v1"] = "starter_v1"
    case_id: str = Field(min_length=1)
    category: CaseCategory
    user_text: str = Field(min_length=1)
    current_spec: PortfolioSpec | None = None
    universe_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    expected: ParseResult


class PredictionRecord(_FrozenModel):
    """One offline prediction JSONL row."""

    case_id: str = Field(min_length=1)
    result: ParseResult


class EvaluationError(_FrozenModel):
    """A stable, machine-readable difference or execution failure."""

    code: Literal[
        "missing_prediction",
        "parse_exception",
        "missing_field",
        "unexpected_field",
        "length_mismatch",
        "value_mismatch",
    ]
    path: str
    expected: JsonValue | None = None
    actual: JsonValue | None = None
    message: str


class AccuracyMetric(_FrozenModel):
    correct: int = Field(ge=0)
    total: int = Field(ge=0)
    accuracy: float | None = Field(default=None, ge=0.0, le=1.0)


class PrecisionRecallF1(_FrozenModel):
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    f1: float | None = Field(default=None, ge=0.0, le=1.0)


class CaseResult(_FrozenModel):
    """Deterministic score and diagnostics for a single benchmark case."""

    case_id: str
    category: CaseCategory
    expected_kind: ParseKind
    actual_kind: ParseKind | None
    parse_kind_match: bool
    semantic_exact_match: bool
    expected_constraint_count: int = Field(ge=0)
    actual_constraint_count: int = Field(ge=0)
    matched_constraint_count: int = Field(ge=0)
    expected_normalized: dict[str, JsonValue]
    actual_normalized: dict[str, JsonValue] | None
    errors: tuple[EvaluationError, ...] = ()


class EvaluationAggregate(_FrozenModel):
    """Corpus-level metrics; all denominators and confusion counts are exposed."""

    case_count: int = Field(ge=0)
    prediction_count: int = Field(ge=0)
    failed_prediction_count: int = Field(ge=0)
    semantic_exact_match: AccuracyMetric
    parse_kind: AccuracyMetric
    constraints_micro: PrecisionRecallF1
    clarification: PrecisionRecallF1


class EvaluationReport(_FrozenModel):
    """Stable report schema with no timestamps or wall-clock measurements."""

    schema_version: Literal["1"] = "1"
    benchmark_suite: Literal["starter_v1"] = "starter_v1"
    aggregate: EvaluationAggregate
    cases: tuple[CaseResult, ...]
