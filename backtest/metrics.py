"""Pure performance and risk metrics used by historical evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Integral

import numpy as np

from backtest.tearsheet import SeriesMetrics


def _finite_vector(values: Iterable[float], *, label: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{label} requires at least one observation.")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} observations must all be finite.")
    return array


def _period_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("periods_per_year must be a positive integer.")
    return int(value)


def _wealth_path_returns(values: Iterable[float], *, label: str) -> np.ndarray:
    sample = _finite_vector(values, label=label)
    if (sample < -1.0).any():
        raise ValueError(f"{label} cannot contain a return below -100%.")
    return sample


def empirical_cvar(returns: Iterable[float], alpha: float = 0.95) -> float:
    """Return equal-probability empirical CVaR of losses.

    This evaluates the exact finite-sample Rockafellar--Uryasev tail average.
    When ``(1-alpha) * n`` is fractional, the boundary loss receives exactly
    the fractional mass needed by the tail instead of rounding the number of
    observations up or down.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    sample = _finite_vector(returns, label="CVaR")
    losses = np.sort(-sample)[::-1]
    tail_mass = (1.0 - alpha) * losses.size
    whole = int(math.floor(tail_mass))
    fraction = tail_mass - whole

    weighted_loss = float(losses[:whole].sum()) if whole else 0.0
    if fraction > 1e-12:
        weighted_loss += fraction * float(losses[whole])
    elif whole == 0:
        # Numerically, a positive tail mass below one can round to zero.
        weighted_loss = tail_mass * float(losses[0])
    return weighted_loss / tail_mass


def annualized_return(returns: Iterable[float], periods_per_year: int = 252) -> float:
    sample = _wealth_path_returns(returns, label="Annualized return")
    periods = _period_count(periods_per_year)
    growth = float(np.prod(1.0 + sample))
    if growth == 0.0:
        return -1.0
    return growth ** (periods / sample.size) - 1.0


def annualized_volatility(
    returns: Iterable[float], periods_per_year: int = 252
) -> float:
    sample = _finite_vector(returns, label="Annualized volatility")
    periods = _period_count(periods_per_year)
    if sample.size < 2:
        return 0.0
    return float(np.std(sample, ddof=1) * math.sqrt(periods))


def sharpe_ratio(
    returns: Iterable[float],
    *,
    periods_per_year: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> float | None:
    sample = _finite_vector(returns, label="Sharpe ratio")
    periods = _period_count(periods_per_year)
    if not math.isfinite(annual_risk_free_rate):
        raise ValueError("annual_risk_free_rate must be finite.")
    if annual_risk_free_rate <= -1.0:
        raise ValueError("annual_risk_free_rate must be greater than -1.")
    if sample.size < 2:
        return None
    volatility = float(np.std(sample, ddof=1))
    if volatility <= 1e-15:
        return None
    periodic_rf = (1.0 + annual_risk_free_rate) ** (1.0 / periods) - 1.0
    return float(np.mean(sample - periodic_rf) / volatility * math.sqrt(periods))


def max_drawdown(returns: Iterable[float]) -> float:
    """Return the most negative peak-to-trough drawdown."""

    sample = _wealth_path_returns(returns, label="Maximum drawdown")
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + sample)))
    peaks = np.maximum.accumulate(wealth)
    drawdowns = np.divide(
        wealth,
        peaks,
        out=np.ones_like(wealth),
        where=peaks != 0.0,
    ) - 1.0
    return float(drawdowns.min())


def series_metrics(
    returns: Iterable[float],
    *,
    periods_per_year: int = 252,
    annual_risk_free_rate: float = 0.0,
    cvar_alpha: float = 0.95,
) -> SeriesMetrics:
    """Build the deterministic metric bundle for one return stream."""

    sample = _finite_vector(returns, label="Series metrics")
    return SeriesMetrics(
        observations=int(sample.size),
        total_return=float(np.prod(1.0 + sample) - 1.0),
        annualized_return=annualized_return(sample, periods_per_year),
        annualized_volatility=annualized_volatility(sample, periods_per_year),
        sharpe=sharpe_ratio(
            sample,
            periods_per_year=periods_per_year,
            annual_risk_free_rate=annual_risk_free_rate,
        ),
        max_drawdown=max_drawdown(sample),
        realized_cvar=empirical_cvar(sample, cvar_alpha),
    )
