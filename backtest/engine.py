"""Deterministic walk-forward evaluation with an explicit close-to-close clock.

The accounting convention in this module is deliberately conservative for a
close-only price panel:

* a signal at close ``s`` sees exactly the trailing configured window ending
  at ``s``;
* the holdings already in the account earn the return from ``s`` to ``s+1``;
* the signal target fills (and pays costs) at close ``s+1``; and
* that target first earns a return from ``s+1`` to ``s+2``.

There are no data downloads or wall-clock fields here.  Callers must supply a
fully local, already-adjusted price panel and, optionally, a strictly aligned
local market-price series.  The panel's ordered ``DatetimeIndex`` is the
authoritative, exogenous trading calendar: an absent row must mean the market
was not scheduled to trade, not that a quote was lost during ingestion.  That
lets holiday-shortened period ends be scheduled without importing an exchange
calendar and without exposing any future price value to a solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from backtest.config import BacktestConfig
from backtest.errors import BacktestError
from backtest.metrics import empirical_cvar, series_metrics
from backtest.tearsheet import (
    BacktestSummary,
    CurvePoint,
    RebalanceRecord,
    Tearsheet,
)
from core.ir import PortfolioSpec, TransactionCost
from core.solve import solve_spec


@dataclass
class _PendingRebalance:
    """Mutable accounting state for a signal until its result is frozen."""

    signal_index: int
    fill_index: int
    signal_date: date
    fill_date: date
    training_start: date
    training_end: date
    signal_weights: dict[str, float]
    target_weights: dict[str, float]
    modeled_cvar: float
    decision_turnover: float
    fill_pretrade_weights: dict[str, float] | None = None
    realized_turnover: float | None = None
    transaction_cost_fraction: float | None = None
    transaction_cost_paid: float | None = None
    realized_returns: list[float] = field(default_factory=list)


def _as_date(value: object) -> date:
    return pd.Timestamp(value).date()


def _validate_prices(spec: PortfolioSpec, prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise BacktestError("prices must be a pandas DataFrame.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise BacktestError("prices must use a DatetimeIndex of observed closes.")
    if prices.index.has_duplicates:
        raise BacktestError("prices index contains duplicate observations.")
    if not prices.index.is_monotonic_increasing:
        raise BacktestError("prices index must be strictly increasing; sort it explicitly first.")
    if prices.columns.has_duplicates:
        raise BacktestError("prices contains duplicate ticker columns.")

    # One close per civil date keeps the public date-only result unambiguous.
    civil_dates = prices.index.tz_localize(None).normalize()
    if civil_dates.has_duplicates:
        raise BacktestError("prices must contain at most one close observation per calendar date.")

    missing = [ticker for ticker in spec.universe if ticker not in prices.columns]
    if missing:
        raise BacktestError(f"prices is missing universe tickers: {missing}.")

    try:
        panel = prices.loc[:, spec.universe].astype(float)
    except (TypeError, ValueError) as exc:
        raise BacktestError("universe price columns must be numeric.") from exc
    values = panel.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise BacktestError("prices must contain only finite values (no missing observations).")
    if (values <= 0.0).any():
        raise BacktestError("prices must be strictly positive.")
    return panel


def _validate_market(market_prices: pd.Series | None, index: pd.DatetimeIndex) -> pd.Series | None:
    if market_prices is None:
        return None
    if not isinstance(market_prices, pd.Series):
        raise BacktestError("market_prices must be a pandas Series.")
    if not market_prices.index.equals(index):
        raise BacktestError(
            "market_prices must have exactly the same ordered index as prices; "
            "implicit alignment is not allowed."
        )
    try:
        market = market_prices.astype(float)
    except (TypeError, ValueError) as exc:
        raise BacktestError("market_prices must be numeric.") from exc
    values = market.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise BacktestError("market_prices must contain only finite values.")
    if (values <= 0.0).any():
        raise BacktestError("market_prices must be strictly positive.")
    return market


def _period_labels(index: pd.DatetimeIndex, frequency: str) -> pd.PeriodIndex:
    # Calendar buckets, rather than every 5/21/63 observations, make holiday
    # and missing-session behavior explicit.  A signal is the last observed
    # close before the next calendar bucket begins.
    naive = index.tz_localize(None)
    aliases = {
        "weekly": "W-FRI",
        "monthly": "M",
        "quarterly": "Q-DEC",
    }
    return naive.to_period(aliases[frequency])


def _signal_indices(
    index: pd.DatetimeIndex, *, lookback_returns: int, frequency: str
) -> tuple[int, ...]:
    labels = _period_labels(index, frequency)
    # Need L+1 closes through s, a fill at s+1, and at least one return earned
    # by the new target (s+1 -> s+2).  Comparing with the following label also
    # avoids treating a truncated final calendar bucket as a period end.
    return tuple(
        i
        for i in range(lookback_returns, len(index) - 2)
        if labels[i] != labels[i + 1]
    )


def _weights_vector(weights: dict[str, float], universe: list[str]) -> np.ndarray:
    try:
        vector = np.array(
            [float(weights.get(ticker, 0.0)) for ticker in universe],
            dtype=float,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise BacktestError("portfolio weights must all be numeric.") from exc
    if not np.isfinite(vector).all():
        raise BacktestError("portfolio weights must all be finite.")
    return vector


def _solved_weights_vector(weights: object, universe: list[str]) -> np.ndarray:
    """Validate that a solver report names the complete canonical universe."""

    if not isinstance(weights, dict):
        raise BacktestError("solver report weights must be a ticker-to-weight mapping.")
    expected = set(universe)
    actual = set(weights)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise BacktestError(
            "solver report weights do not match the backtest universe "
            f"(missing={missing}, extra={extra})."
        )
    return _weights_vector(weights, universe)


def _weights_dict(weights: np.ndarray, universe: list[str]) -> dict[str, float]:
    return {ticker: float(weights[i]) for i, ticker in enumerate(universe)}


def _portfolio_return(weights: np.ndarray, asset_returns: np.ndarray) -> float:
    value = float(weights @ asset_returns)
    if not np.isfinite(value):
        raise BacktestError("portfolio return became non-finite.")
    return value


def _drift_weights(
    weights: np.ndarray,
    asset_returns: np.ndarray,
    *,
    end_date: date,
) -> np.ndarray:
    """Drift close weights through one period, retaining implicit cash at 0%."""

    portfolio_return = _portfolio_return(weights, asset_returns)
    wealth_multiplier = 1.0 + portfolio_return
    if wealth_multiplier <= 0.0:
        raise BacktestError(
            f"Portfolio wealth was exhausted by the close on {end_date.isoformat()}; "
            "weights cannot be drifted beyond that date."
        )
    drifted = weights * (1.0 + asset_returns) / wealth_multiplier
    if not np.isfinite(drifted).all():
        raise BacktestError(f"Portfolio weights became non-finite on {end_date.isoformat()}.")
    return drifted


def _turnover(before: np.ndarray, after: np.ndarray) -> float:
    return float(np.abs(after - before).sum())


def _resolve_execution_cost(spec: PortfolioSpec, config: BacktestConfig) -> tuple[float, str]:
    if config.execution_cost_bps is not None:
        return float(config.execution_cost_bps), "config_override"
    total = float(
        sum(c.bps for c in spec.constraints if isinstance(c, TransactionCost))
    )
    return total, "spec_transaction_cost" if total > 0.0 else "none"


def _apply_cost(
    nav: float,
    turnover: float,
    cost_rate: float,
    *,
    fill_date: date,
) -> tuple[float, float, float]:
    fraction = cost_rate * turnover
    if not np.isfinite(fraction) or fraction < 0.0:
        raise BacktestError(f"Transaction cost became invalid on {fill_date.isoformat()}.")
    if fraction >= 1.0:
        raise BacktestError(
            f"Transaction cost would exhaust wealth on {fill_date.isoformat()} "
            f"(fraction={fraction:g})."
        )
    paid = nav * fraction
    return nav - paid, fraction, paid


def _frozen_record(record: _PendingRebalance, *, alpha: float) -> RebalanceRecord:
    if (
        record.fill_pretrade_weights is None
        or record.realized_turnover is None
        or record.transaction_cost_fraction is None
        or record.transaction_cost_paid is None
    ):
        raise BacktestError(
            f"Internal accounting error: signal on {record.signal_date.isoformat()} never filled."
        )
    if not record.realized_returns:
        raise BacktestError(
            f"Signal on {record.signal_date.isoformat()} has no post-fill return observation."
        )
    return RebalanceRecord(
        signal_date=record.signal_date,
        fill_date=record.fill_date,
        training_start=record.training_start,
        training_end=record.training_end,
        signal_weights=record.signal_weights,
        fill_pretrade_weights=record.fill_pretrade_weights,
        target_weights=record.target_weights,
        modeled_cvar=record.modeled_cvar,
        realized_cvar=empirical_cvar(record.realized_returns, alpha),
        holding_observations=len(record.realized_returns),
        decision_turnover=record.decision_turnover,
        realized_turnover=record.realized_turnover,
        transaction_cost_fraction=record.transaction_cost_fraction,
        transaction_cost_paid=record.transaction_cost_paid,
    )


def run_backtest(
    spec: PortfolioSpec,
    prices: pd.DataFrame,
    *,
    config: BacktestConfig | None = None,
    sectors: dict[str, str] | None = None,
    benchmarks: dict[str, dict[str, float]] | None = None,
    factors: dict[str, dict[str, float]] | None = None,
    market_prices: pd.Series | None = None,
    time_limit_s: float | None = None,
) -> Tearsheet:
    """Run a complete no-lookahead walk-forward experiment.

    Every scheduled solve is mandatory.  A dated :class:`BacktestError` is
    raised on the first failed signal rather than silently shortening or
    cherry-picking the history.  ``prices.index`` is treated as the pre-known
    trading calendar, so callers must distinguish genuine non-trading sessions
    from data outages before invoking this function.
    """

    cfg = config or BacktestConfig()
    panel = _validate_prices(spec, prices)
    market = _validate_market(market_prices, panel.index)
    signals = _signal_indices(
        panel.index,
        lookback_returns=cfg.lookback_returns,
        frequency=cfg.rebalance_frequency,
    )
    if not signals:
        minimum = cfg.lookback_returns + 3
        raise BacktestError(
            "No eligible calendar-period-end signal has a complete training window, "
            "next-close fill, and post-fill return. "
            f"At least {minimum} close observations are required, and often more for "
            f"{cfg.rebalance_frequency} scheduling."
        )

    universe = list(spec.universe)
    n_assets = len(universe)
    initial = _weights_vector(spec.current_weights or {}, universe)
    equal_target = np.full(n_assets, 1.0 / n_assets, dtype=float)
    cost_bps, cost_source = _resolve_execution_cost(spec, cfg)
    cost_rate = cost_bps / 10_000.0

    start_index = signals[0]
    signal_set = set(signals)
    current = initial.copy()
    equal_current = initial.copy()
    strategy_net_nav = 1.0
    strategy_gross_nav = 1.0
    equal_net_nav = 1.0
    market_nav = 1.0 if market is not None else None

    net_returns: list[float] = []
    gross_returns: list[float] = []
    equal_returns: list[float] = []
    market_returns: list[float] = []
    records: list[_PendingRebalance] = []
    pending: _PendingRebalance | None = None
    active: _PendingRebalance | None = None
    pending_equal_fill: int | None = None

    first_date = _as_date(panel.index[start_index])
    curves: list[CurvePoint] = [
        CurvePoint(
            date=first_date,
            strategy_net=strategy_net_nav,
            strategy_gross=strategy_gross_nav,
            equal_weight_net=equal_net_nav,
            market=market_nav,
        )
    ]

    def create_signal(signal_index: int) -> tuple[_PendingRebalance, int]:
        signal_date = _as_date(panel.index[signal_index])
        window_start = signal_index - cfg.lookback_returns
        window = panel.iloc[window_start : signal_index + 1]
        if len(window) != cfg.lookback_returns + 1:  # defensive invariant
            raise BacktestError(
                f"Signal on {signal_date.isoformat()} did not receive exactly "
                f"{cfg.lookback_returns + 1} training closes."
            )

        signal_weights = _weights_dict(current, universe)
        dated_spec = PortfolioSpec.model_validate(
            {**spec.model_dump(), "current_weights": signal_weights}
        )
        try:
            _, report = solve_spec(
                dated_spec,
                window,
                sectors=sectors,
                benchmarks=benchmarks,
                factors=factors,
                periods_per_year=cfg.periods_per_year,
                time_limit_s=time_limit_s,
            )
        except Exception as exc:
            raise BacktestError(
                f"Backtest solve failed at signal close {signal_date.isoformat()}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            target = _solved_weights_vector(report.weights, universe)
        except (AttributeError, BacktestError) as exc:
            raise BacktestError(
                f"Backtest solve produced invalid weights at signal close "
                f"{signal_date.isoformat()}: {exc}"
            ) from exc
        training_returns = window.to_numpy(dtype=float)[1:] / window.to_numpy(dtype=float)[:-1] - 1.0
        modeled_returns = training_returns @ target
        modeled_cvar = empirical_cvar(modeled_returns, cfg.cvar_alpha)
        fill_index = signal_index + cfg.execution_lag_observations
        record = _PendingRebalance(
            signal_index=signal_index,
            fill_index=fill_index,
            signal_date=signal_date,
            fill_date=_as_date(panel.index[fill_index]),
            training_start=_as_date(window.index[0]),
            training_end=_as_date(window.index[-1]),
            signal_weights=signal_weights,
            target_weights=_weights_dict(target, universe),
            modeled_cvar=modeled_cvar,
            decision_turnover=_turnover(current, target),
        )
        return record, fill_index

    # The first signal happens at the initial curve close.  Later signals are
    # created after that close's return, fill, and cost events have settled.
    pending, pending_equal_fill = create_signal(start_index)
    records.append(pending)

    values = panel.to_numpy(dtype=float)
    for end_index in range(start_index + 1, len(panel)):
        end_date = _as_date(panel.index[end_index])
        asset_returns = values[end_index] / values[end_index - 1] - 1.0

        start_net_nav = strategy_net_nav
        start_gross_nav = strategy_gross_nav
        start_equal_nav = equal_net_nav
        start_market_nav = market_nav

        gross_return = _portfolio_return(current, asset_returns)
        strategy_gross_nav *= 1.0 + gross_return
        strategy_net_nav *= 1.0 + gross_return
        equal_return = _portfolio_return(equal_current, asset_returns)
        equal_net_nav *= 1.0 + equal_return

        # The record active at the interval's starting close owns this return,
        # including the interval that ends at the next rebalance fill.
        if active is not None:
            active.realized_returns.append(gross_return)

        current = _drift_weights(current, asset_returns, end_date=end_date)
        equal_current = _drift_weights(equal_current, asset_returns, end_date=end_date)

        if pending is not None and pending.fill_index == end_index:
            target = _weights_vector(pending.target_weights, universe)
            realized_turnover = _turnover(current, target)
            strategy_net_nav, cost_fraction, cost_paid = _apply_cost(
                strategy_net_nav,
                realized_turnover,
                cost_rate,
                fill_date=end_date,
            )
            pending.fill_pretrade_weights = _weights_dict(current, universe)
            pending.realized_turnover = realized_turnover
            pending.transaction_cost_fraction = cost_fraction
            pending.transaction_cost_paid = cost_paid
            current = target
            active = pending
            pending = None

        if pending_equal_fill == end_index:
            equal_turnover = _turnover(equal_current, equal_target)
            equal_net_nav, _, _ = _apply_cost(
                equal_net_nav,
                equal_turnover,
                cost_rate,
                fill_date=end_date,
            )
            equal_current = equal_target.copy()
            pending_equal_fill = None

        if market is not None:
            market_return = float(market.iloc[end_index] / market.iloc[end_index - 1] - 1.0)
            market_nav = float(market_nav * (1.0 + market_return))
            market_returns.append(market_return)

        if end_index in signal_set:
            if pending is not None:
                raise BacktestError(
                    f"Signal schedules overlap before the prior fill on {end_date.isoformat()}."
                )
            pending, pending_equal_fill = create_signal(end_index)
            records.append(pending)

        net_period_return = strategy_net_nav / start_net_nav - 1.0
        gross_period_return = strategy_gross_nav / start_gross_nav - 1.0
        equal_period_return = equal_net_nav / start_equal_nav - 1.0
        if start_market_nav is not None and market_nav is not None:
            # The ratio is used rather than reusing the price return so the
            # summary and the displayed normalized curve are mathematically
            # identical even under floating-point accumulation.
            market_returns[-1] = market_nav / start_market_nav - 1.0
        net_returns.append(net_period_return)
        gross_returns.append(gross_period_return)
        equal_returns.append(equal_period_return)

        curves.append(
            CurvePoint(
                date=end_date,
                strategy_net=strategy_net_nav,
                strategy_gross=strategy_gross_nav,
                equal_weight_net=equal_net_nav,
                market=market_nav,
            )
        )

    frozen_records = tuple(_frozen_record(record, alpha=cfg.cvar_alpha) for record in records)
    total_turnover = float(sum(record.realized_turnover for record in frozen_records))
    total_cost_paid = float(sum(record.transaction_cost_paid for record in frozen_records))
    holding_count = sum(record.holding_observations for record in frozen_records)
    modeled_cvar = (
        float(
            sum(record.modeled_cvar * record.holding_observations for record in frozen_records)
            / holding_count
        )
        if holding_count
        else None
    )
    realized_holding_cvar = (
        float(
            sum(
                record.realized_cvar * record.holding_observations
                for record in frozen_records
                if record.realized_cvar is not None
            )
            / holding_count
        )
        if holding_count
        else None
    )
    realized_gross_cvar = empirical_cvar(gross_returns, cfg.cvar_alpha)
    realized_net_cvar = empirical_cvar(net_returns, cfg.cvar_alpha)
    summary = BacktestSummary(
        strategy=series_metrics(
            net_returns,
            periods_per_year=cfg.periods_per_year,
            annual_risk_free_rate=cfg.annual_risk_free_rate,
            cvar_alpha=cfg.cvar_alpha,
        ),
        strategy_gross=series_metrics(
            gross_returns,
            periods_per_year=cfg.periods_per_year,
            annual_risk_free_rate=cfg.annual_risk_free_rate,
            cvar_alpha=cfg.cvar_alpha,
        ),
        equal_weight=series_metrics(
            equal_returns,
            periods_per_year=cfg.periods_per_year,
            annual_risk_free_rate=cfg.annual_risk_free_rate,
            cvar_alpha=cfg.cvar_alpha,
        ),
        market=(
            series_metrics(
                market_returns,
                periods_per_year=cfg.periods_per_year,
                annual_risk_free_rate=cfg.annual_risk_free_rate,
                cvar_alpha=cfg.cvar_alpha,
            )
            if market is not None
            else None
        ),
        total_turnover=total_turnover,
        annualized_turnover=total_turnover * cfg.periods_per_year / len(net_returns),
        total_cost_paid=total_cost_paid,
        cost_drag=max(0.0, float(strategy_gross_nav - strategy_net_nav)),
        holding_weighted_modeled_cvar=modeled_cvar,
        holding_weighted_realized_cvar=realized_holding_cvar,
        realized_gross_cvar=realized_gross_cvar,
        realized_net_cvar=realized_net_cvar,
        resolved_execution_cost_bps=cost_bps,
        execution_cost_source=cost_source,
    )
    return Tearsheet(
        config=cfg,
        start_date=first_date,
        end_date=_as_date(panel.index[-1]),
        curves=tuple(curves),
        rebalances=frozen_records,
        summary=summary,
    )
