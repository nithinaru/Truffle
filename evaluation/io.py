"""Strict JSONL readers and deterministic report serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from evaluation.models import BenchmarkCase, EvaluationReport, PredictionRecord


class EvaluationDataError(ValueError):
    """Raised for malformed, duplicate, or internally inconsistent JSONL."""


def _read_jsonl[RecordT: BaseModel](
    path: str | Path, model: type[RecordT]
) -> tuple[RecordT, ...]:
    source = Path(path)
    records: list[RecordT] = []
    seen: set[str] = set()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationDataError(f"Could not read {source}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationDataError(
                f"{source}:{line_number}: invalid JSON: {exc.msg}."
            ) from exc
        try:
            record = model.model_validate(raw)
        except ValidationError as exc:
            raise EvaluationDataError(
                f"{source}:{line_number}: record failed validation: {exc}"
            ) from exc
        case_id = str(record.model_dump()["case_id"])
        if case_id in seen:
            raise EvaluationDataError(
                f"{source}:{line_number}: duplicate case_id {case_id!r}."
            )
        seen.add(case_id)
        records.append(record)
    if not records:
        raise EvaluationDataError(f"{source}: expected at least one JSONL record.")
    return tuple(records)


def load_benchmark(path: str | Path) -> tuple[BenchmarkCase, ...]:
    """Load and validate a benchmark, including unique case IDs."""
    return _read_jsonl(path, BenchmarkCase)


def load_predictions(path: str | Path) -> dict[str, PredictionRecord]:
    """Load validated offline predictions keyed by unique case ID."""
    records = _read_jsonl(path, PredictionRecord)
    return {record.case_id: record for record in records}


def report_json(report: EvaluationReport) -> str:
    """Serialize a report canonically; repeated runs produce identical bytes."""
    return json.dumps(
        report.model_dump(mode="json"),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
