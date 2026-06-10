"""Tests for data/estimation.py.

We verify the Ledoit-Wolf shrinkage estimator actually shrinks toward its
identity-scaled target: when N is comparable to T, the eigenvalues of the
shrunk covariance are pulled in from the sample covariance's eigenvalue
spread.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.estimation import estimate_moments


def _synthetic_prices(
    n_assets: int,
    n_days: int,
    seed: int = 42,
    daily_vol: float = 0.02,
) -> pd.DataFrame:
    """Geometric random walks; uncorrelated assets, log returns iid N(0, vol²)."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0, daily_vol, size=(n_days, n_assets))
    log_prices = np.cumsum(log_returns, axis=0)
    prices = 100.0 * np.exp(log_prices)
    cols = [f"T{i}" for i in range(n_assets)]
    return pd.DataFrame(prices, columns=cols)


def test_estimate_moments_shapes_and_psd() -> None:
    prices = _synthetic_prices(n_assets=5, n_days=300)
    mu, sigma = estimate_moments(prices)
    assert mu.shape == (5,)
    assert sigma.shape == (5, 5)
    # Symmetric.
    np.testing.assert_allclose(sigma, sigma.T, atol=1e-12)
    # Positive-definite (all eigenvalues > 0).
    assert np.linalg.eigvalsh(sigma).min() > 0


def test_ledoit_wolf_shrinks_eigenvalue_spread() -> None:
    """With N close to T, sample covariance has high eigenvalue dispersion.
    Ledoit-Wolf pulls eigenvalues toward (trace/N) * I, shrinking the spread.
    Compare condition numbers: shrunk should be smaller than sample."""
    prices = _synthetic_prices(n_assets=20, n_days=40)  # N/T = 0.5, noisy regime
    log_prices = np.log(prices.to_numpy(dtype=float))
    log_returns = np.diff(log_prices, axis=0)
    sample_cov = np.cov(log_returns, rowvar=False)
    _, shrunk_cov = estimate_moments(prices, periods_per_year=1)  # avoid annualization scaling

    cond_sample = np.linalg.cond(sample_cov)
    cond_shrunk = np.linalg.cond(shrunk_cov)
    # Shrinking toward c * I should reduce condition number — often by orders of magnitude.
    assert cond_shrunk < cond_sample, (cond_sample, cond_shrunk)


def test_annualization_scales_linearly() -> None:
    prices = _synthetic_prices(n_assets=3, n_days=500)
    _, sigma_252 = estimate_moments(prices, periods_per_year=252)
    _, sigma_1 = estimate_moments(prices, periods_per_year=1)
    np.testing.assert_allclose(sigma_252, sigma_1 * 252, rtol=1e-12)


def test_rejects_non_positive_prices() -> None:
    bad = pd.DataFrame({"A": [10.0, 0.0, 5.0], "B": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="strictly positive"):
        estimate_moments(bad)


def test_rejects_single_observation() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        estimate_moments(pd.DataFrame({"A": [10.0]}))
