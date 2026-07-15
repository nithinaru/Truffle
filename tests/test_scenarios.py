"""Focused tests for deterministic bootstrap scenario generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.scenarios import (
    block_bootstrap_scenarios,
    historical_scenarios,
    iid_bootstrap_scenarios,
)


def _prices() -> pd.DataFrame:
    # The one-period rows are deliberately distinct and cross-asset paired.
    returns = np.array(
        [
            [0.10, -0.05],
            [-0.20, 0.10],
            [0.25, 0.20],
            [-0.10, -0.25],
        ]
    )
    values = np.vstack([np.array([100.0, 200.0]), np.cumprod(1.0 + returns, axis=0) * [100, 200]])
    return pd.DataFrame(values, columns=["A", "B"])


def test_iid_bootstrap_samples_complete_one_period_rows_with_replacement() -> None:
    prices = _prices()
    observed = historical_scenarios(prices)

    actual = iid_bootstrap_scenarios(prices, 8, seed=19)
    expected_indices = np.random.default_rng(19).integers(0, len(observed), size=8)

    assert actual.shape == (8, 2)
    np.testing.assert_allclose(actual, observed[expected_indices])
    # Sampling complete rows protects contemporaneous cross-asset relationships.
    assert all(any(np.allclose(row, source) for source in observed) for row in actual)


def test_iid_bootstrap_is_reproducible_for_a_seed() -> None:
    first = iid_bootstrap_scenarios(_prices(), 20, seed=7)
    second = iid_bootstrap_scenarios(_prices(), 20, seed=7)
    other_seed = iid_bootstrap_scenarios(_prices(), 20, seed=8)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other_seed)


def test_block_bootstrap_compounds_contiguous_historical_blocks() -> None:
    prices = _prices()
    returns = historical_scenarios(prices)
    n_scenarios = 6
    horizon = 2

    actual = block_bootstrap_scenarios(
        prices,
        n_scenarios,
        horizon,
        seed=23,
    )
    starts = np.random.default_rng(23).integers(
        0,
        len(returns) - horizon + 1,
        size=n_scenarios,
    )
    expected = np.vstack(
        [np.prod(1.0 + returns[start : start + horizon], axis=0) - 1.0 for start in starts]
    )

    assert actual.shape == (n_scenarios, 2)
    np.testing.assert_allclose(actual, expected)


def test_one_period_blocks_match_iid_sampling_for_the_same_seed() -> None:
    iid = iid_bootstrap_scenarios(_prices(), 12, seed=5)
    blocks = block_bootstrap_scenarios(_prices(), 12, 1, seed=5)

    np.testing.assert_allclose(blocks, iid)


@pytest.mark.parametrize("invalid_count", [0, -1, 1.5, True])
def test_bootstraps_reject_invalid_sample_counts(invalid_count: object) -> None:
    prices = _prices()

    with pytest.raises(ValueError, match="n_scenarios must be a positive integer"):
        iid_bootstrap_scenarios(prices, invalid_count, seed=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_scenarios must be a positive integer"):
        block_bootstrap_scenarios(prices, invalid_count, 2, seed=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_length", [0, -1, 1.5, True])
def test_block_bootstrap_rejects_invalid_block_length(invalid_length: object) -> None:
    with pytest.raises(ValueError, match="block_length must be a positive integer"):
        block_bootstrap_scenarios(
            _prices(),
            2,
            invalid_length,  # type: ignore[arg-type]
            seed=1,
        )


def test_block_bootstrap_rejects_horizon_longer_than_history() -> None:
    with pytest.raises(ValueError, match="must not exceed.*historical return rows"):
        block_bootstrap_scenarios(_prices(), 2, 5, seed=1)


@pytest.mark.parametrize(
    "bad_prices",
    [
        pd.DataFrame({"A": [100.0, 0.0]}),
        pd.DataFrame({"A": [100.0, np.nan]}),
        pd.DataFrame({"A": [100.0, np.inf]}),
    ],
)
def test_all_scenario_generators_reject_non_positive_or_non_finite_prices(
    bad_prices: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="finite and strictly positive"):
        historical_scenarios(bad_prices)
    with pytest.raises(ValueError, match="finite and strictly positive"):
        iid_bootstrap_scenarios(bad_prices, 1, seed=1)
    with pytest.raises(ValueError, match="finite and strictly positive"):
        block_bootstrap_scenarios(bad_prices, 1, 1, seed=1)
