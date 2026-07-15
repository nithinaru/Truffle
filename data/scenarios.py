"""Scenario generation for CVaR (Rockafellar–Uryasev).

CVaR is evaluated as a *sample average over scenarios* — each row of the
returned matrix is one realization of asset returns. Historical and IID
scenarios are one-period simple returns. Block-bootstrap scenarios are
explicitly multi-period: each row compounds one contiguous historical block.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd


def _simple_returns(prices: pd.DataFrame) -> np.ndarray:
    """Validate a price panel and return its one-period simple returns."""
    if prices.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 price observations to build scenarios; got {prices.shape[0]}."
        )
    if prices.shape[1] < 1:
        raise ValueError("Need at least one asset column to build scenarios.")

    try:
        arr = prices.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Prices must be numeric, finite, and strictly positive.") from exc

    if not np.isfinite(arr).all() or np.any(arr <= 0):
        raise ValueError("Prices must be finite and strictly positive.")
    return arr[1:] / arr[:-1] - 1.0


def _validate_positive_count(value: int, *, name: str) -> int:
    """Return an integer count, rejecting booleans and non-positive values."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return int(value)


def historical_scenarios(prices: pd.DataFrame) -> np.ndarray:
    """Return the historical scenario matrix.

    Args:
        prices: ``DataFrame`` indexed by date with one column per asset,
            in the canonical universe order. Must have at least 2 rows.

    Returns:
        ``np.ndarray`` of shape ``(S, N)`` of per-period *simple* returns
        (``p_t / p_{t-1} − 1``). We use simple returns here, not log
        returns: portfolio P&L over one period equals ``w · r_simple``
        only for simple returns; using log returns would mis-state CVaR
        for any non-trivial holding.

    Raises:
        ValueError: if the panel is too small or contains non-positive prices.
    """
    return _simple_returns(prices)


def iid_bootstrap_scenarios(
    prices: pd.DataFrame,
    n_scenarios: int,
    *,
    seed: int,
) -> np.ndarray:
    """Sample one-period historical simple-return rows with replacement.

    One sampled row is one complete cross-asset observation, so contemporaneous
    dependence between assets is retained. Temporal ordering is intentionally
    discarded; use :func:`block_bootstrap_scenarios` when a multi-period horizon
    and within-block time dependence are required.

    Args:
        prices: Positive, finite prices with dates in rows and assets in columns.
            At least two observations and one asset are required.
        n_scenarios: Number of one-period return rows to sample. Must be positive.
        seed: Seed passed to ``numpy.random.default_rng``. The same inputs and seed
            produce the same matrix.

    Returns:
        Array of shape ``(n_scenarios, n_assets)``. Values are one-period simple
        returns in fractional units (``0.01`` means a 1% return).

    Raises:
        ValueError: if the price panel or ``n_scenarios`` is invalid.
    """
    sample_count = _validate_positive_count(n_scenarios, name="n_scenarios")
    returns = _simple_returns(prices)
    rng = np.random.default_rng(seed)
    row_indices = rng.integers(0, returns.shape[0], size=sample_count)
    return returns[row_indices]


def block_bootstrap_scenarios(
    prices: pd.DataFrame,
    n_scenarios: int,
    block_length: int,
    *,
    seed: int,
) -> np.ndarray:
    """Sample and compound contiguous historical return blocks.

    Each scenario chooses one valid block start uniformly with replacement, keeps
    the following ``block_length`` one-period rows in their historical order, and
    compounds each asset as ``prod(1 + r_t) - 1``. Thus ``block_length`` is also
    the scenario horizon. This retains the observed ordering *within* a sampled
    block; it does not claim to preserve dependence across independently sampled
    scenarios.

    Args:
        prices: Positive, finite prices with dates in rows and assets in columns.
            At least two observations and one asset are required.
        n_scenarios: Number of block-compounded scenarios to sample. Must be
            positive.
        block_length: Horizon in price periods for every output row. Must be a
            positive integer no larger than the number of historical return rows
            (``len(prices) - 1``).
        seed: Seed passed to ``numpy.random.default_rng``. The same inputs and seed
            produce the same matrix.

    Returns:
        Array of shape ``(n_scenarios, n_assets)``. Values are
        ``block_length``-period compounded simple returns in fractional units.

    Raises:
        ValueError: if the price panel, sample count, or block length is invalid.

    Note:
        These rows have different financial units from the one-period rows returned
        by :func:`historical_scenarios` and :func:`iid_bootstrap_scenarios` whenever
        ``block_length != 1``. CVaR limits and reported results must use the same
        horizon as the selected scenarios.
    """
    sample_count = _validate_positive_count(n_scenarios, name="n_scenarios")
    horizon = _validate_positive_count(block_length, name="block_length")
    returns = _simple_returns(prices)
    if horizon > returns.shape[0]:
        raise ValueError(
            "block_length must not exceed the number of historical return rows "
            f"({returns.shape[0]}); got {horizon}."
        )

    rng = np.random.default_rng(seed)
    max_start = returns.shape[0] - horizon
    starts = rng.integers(0, max_start + 1, size=sample_count)
    offsets = np.arange(horizon)
    blocks = returns[starts[:, np.newaxis] + offsets]
    return np.prod(1.0 + blocks, axis=1) - 1.0
