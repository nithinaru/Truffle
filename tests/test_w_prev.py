"""Slice 1 tests: single-shot current_weights / w_prev plumbing.

The zero-vector default ("fresh from cash") is the convention turnover and
transaction cost both lean on, so it is pinned down explicitly here.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.compiler import resolve_w_prev
from core.ir import Budget, LongOnly, MinVariance, PortfolioSpec


def test_w_prev_vector_defaults_to_zero_when_none() -> None:
    spec = PortfolioSpec(
        universe=["A", "B", "C"],
        objective=MinVariance(),
        constraints=[Budget(), LongOnly()],
    )
    assert spec.current_weights is None
    assert spec.w_prev_vector() == [0.0, 0.0, 0.0]


def test_w_prev_vector_aligns_to_universe_order_with_zero_fill() -> None:
    spec = PortfolioSpec(
        universe=["A", "B", "C"],
        objective=MinVariance(),
        constraints=[],
        current_weights={"C": 0.3, "A": 0.7},  # out of order, B missing
    )
    # Aligned to universe order; missing B -> 0.0.
    assert spec.w_prev_vector() == [0.7, 0.0, 0.3]


def test_current_weights_rejects_unknown_ticker() -> None:
    with pytest.raises(ValueError, match="not in universe"):
        PortfolioSpec(
            universe=["A", "B"],
            objective=MinVariance(),
            constraints=[],
            current_weights={"A": 0.5, "ZZZ": 0.5},
        )


def test_resolve_w_prev_none_is_zero_vector() -> None:
    np.testing.assert_array_equal(resolve_w_prev(None, 4), np.zeros(4))


def test_resolve_w_prev_rejects_wrong_length() -> None:
    with pytest.raises(Exception, match="does not match universe size"):
        resolve_w_prev(np.array([0.1, 0.2]), 3)
