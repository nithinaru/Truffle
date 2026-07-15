"""Expected-return and covariance estimation from a price panel.

Why Ledoit–Wolf shrinkage (BLUEPRINT §5):

    When the number of assets N is close to the number of observations T, the
    sample covariance matrix is a noisy estimate — eigenvalues are dispersed
    far from their true values, the smallest are biased toward zero, and the
    inverse (which Markowitz weights amplify) explodes. Ledoit–Wolf shrinks
    the sample covariance toward a structured target (here a multiple of the
    identity) with an analytically optimal shrinkage intensity that minimizes
    expected Frobenius distance to the true covariance. The result is always
    well-conditioned and positive-definite, and it dominates the sample
    covariance in mean-squared error whenever N/T is non-trivial.

Annualization:

    We compute *log* returns r_t = ln(p_t / p_{t-1}) and treat them as
    approximately IID over the lookback window. Mean log return is scaled by
    ``periods_per_year`` (default 252 for daily prices); covariance is scaled
    by the same factor. This is the standard daily→annual convention used in
    every practitioner library.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

DEFAULT_PERIODS_PER_YEAR = 252


def estimate_moments(
    prices: pd.DataFrame,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate annualized expected returns and covariance from a price panel.

    Args:
        prices: ``DataFrame`` of asset prices, indexed by date, one column
            per ticker. Column order is the canonical asset order returned
            by the caller — match it against ``PortfolioSpec.universe``.
        periods_per_year: Annualization factor. 252 for daily, 52 for
            weekly, 12 for monthly.

    Returns:
        ``(mu, sigma)`` where ``mu`` is the annualized mean log-return
        vector (shape ``(n,)``) and ``sigma`` is the annualized
        Ledoit–Wolf shrunk covariance matrix (shape ``(n, n)``,
        symmetric positive-definite).

    Raises:
        ValueError: if the price panel has fewer than 2 rows (no returns)
            or contains non-positive prices (log undefined).
    """
    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, Integral)
        or periods_per_year < 1
    ):
        raise ValueError("periods_per_year must be a positive integer.")
    annualization = int(periods_per_year)
    if prices.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 price observations to compute returns; got {prices.shape[0]}."
        )
    if (prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive (log return undefined).")

    log_prices = np.log(prices.to_numpy(dtype=float))
    log_returns = np.diff(log_prices, axis=0)  # shape (T-1, N)

    mu = log_returns.mean(axis=0) * annualization

    # Ledoit–Wolf shrinks toward a (μ_trace · I) target. Always PSD by
    # construction, which is what the compiler's `cp.psd_wrap` is promising.
    lw = LedoitWolf().fit(log_returns)
    sigma = lw.covariance_ * annualization

    # Defensive symmetrization — Ledoit–Wolf returns symmetric matrices in
    # principle, but downstream `cp.quad_form` is allergic to even 1e-15
    # asymmetry on some platforms.
    sigma = 0.5 * (sigma + sigma.T)
    return mu, sigma
