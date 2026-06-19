"""Slice 4 tests: CSV data loaders and universe alignment/validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.inputs import align_named, load_named_series, load_sectors

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_load_sectors_sample() -> None:
    sectors = load_sectors(_EXAMPLES / "sectors_sample.csv")
    assert sectors["AAA"] == "Tech"
    assert sectors["EEE"] == "Healthcare"


def test_load_named_series_wide_benchmark() -> None:
    benches = load_named_series(_EXAMPLES / "benchmark_sample.csv", label="Benchmark")
    assert set(benches) == {"bench"}
    assert benches["bench"]["AAA"] == 0.30


def test_load_named_series_multiple_factors() -> None:
    factors = load_named_series(_EXAMPLES / "factors_sample.csv", label="Factor")
    assert set(factors) == {"value", "momentum"}
    assert factors["momentum"]["DDD"] == 1.0


def test_align_named_orders_to_universe() -> None:
    named = {"bench": {"B": 0.6, "A": 0.4}}
    aligned = align_named(named, ["A", "B"], label="Benchmark")
    np.testing.assert_allclose(aligned["bench"], [0.4, 0.6])


def test_align_named_errors_on_missing_ticker() -> None:
    named = {"bench": {"A": 1.0}}  # missing B
    with pytest.raises(ValueError, match="missing values for universe tickers"):
        align_named(named, ["A", "B"], label="Benchmark")


def test_align_named_none_passthrough() -> None:
    assert align_named(None, ["A"], label="Factor") is None


def test_load_named_series_requires_ticker_first_column(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("name,bench\nAAA,0.5\n")
    with pytest.raises(ValueError, match="'ticker' as its first column"):
        load_named_series(bad, label="Benchmark")
