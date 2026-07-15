"""Timing, accounting, and validation tests for the walk-forward engine."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import backtest.engine as engine
from backtest import BacktestConfig, BacktestError, run_backtest
from backtest.metrics import (
    annualized_return,
    annualized_volatility,
    empirical_cvar,
    max_drawdown,
    sharpe_ratio,
)
from core.ir import Budget, LongOnly, MinVariance, PortfolioSpec, TransactionCost


def _spec(
    *,
    current_weights: dict[str, float] | None = None,
    costs: tuple[float, ...] = (),
) -> PortfolioSpec:
    constraints = [Budget(id="budget"), LongOnly(id="long")]
    constraints.extend(
        TransactionCost(id=f"cost_{i}", bps=bps) for i, bps in enumerate(costs)
    )
    return PortfolioSpec(
        universe=["A", "B"],
        objective=MinVariance(),
        constraints=constraints,
        current_weights=current_weights,
    )


def _single_month_panel() -> pd.DataFrame:
    index = pd.to_datetime(
        ["2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"]
    )
    # The large moves occur strictly after the Jan-31 signal close.
    return pd.DataFrame(
        {
            "A": [100.0, 100.0, 100.0, 110.0, 110.0],
            "B": [100.0, 100.0, 100.0, 100.0, 120.0],
        },
        index=index,
    )


def _monthly_config(**updates: object) -> BacktestConfig:
    return BacktestConfig.model_validate(
        {
            "lookback_returns": 2,
            "rebalance_frequency": "monthly",
            **updates,
        }
    )


def test_signal_window_has_exact_trailing_closes_and_no_future_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _single_month_panel()
    seen: list[pd.DataFrame] = []

    def fake_solve(spec: PortfolioSpec, window: pd.DataFrame, **_: object) -> object:
        seen.append(window.copy())
        return object(), SimpleNamespace(weights={"A": 0.0, "B": 1.0})

    monkeypatch.setattr(engine, "solve_spec", fake_solve)
    sheet = run_backtest(_spec(), prices, config=_monthly_config())

    assert len(seen) == 1
    assert list(seen[0].index) == list(prices.index[:3])
    assert len(seen[0]) == 3  # lookback_returns + one price close
    assert seen[0].index[-1] == pd.Timestamp("2024-01-31")
    assert sheet.rebalances[0].training_start.isoformat() == "2024-01-29"
    assert sheet.rebalances[0].training_end.isoformat() == "2024-01-31"


def test_real_solver_integration_produces_a_complete_delayed_rebalance() -> None:
    prices = _single_month_panel().loc[:, ["A"]]
    spec = PortfolioSpec(
        universe=["A"],
        objective=MinVariance(),
        constraints=[Budget(id="budget"), LongOnly(id="long")],
    )

    sheet = run_backtest(spec, prices, config=_monthly_config())

    assert len(sheet.rebalances) == 1
    record = sheet.rebalances[0]
    np.testing.assert_allclose(record.signal_weights["A"], 0.0)
    np.testing.assert_allclose(record.target_weights["A"], 1.0, atol=1e-7)
    assert record.signal_date.isoformat() == "2024-01-31"
    assert record.fill_date.isoformat() == "2024-02-01"
    assert record.holding_observations == 1


def test_future_price_perturbation_cannot_change_an_already_formed_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _single_month_panel()

    def window_only_target(_spec: PortfolioSpec, window: pd.DataFrame, **_: object) -> object:
        # Deliberately make the result a deterministic function of only the
        # frame handed to the solver.  If the engine leaked Feb prices into the
        # Jan31 decision, the perturbation below would change this target.
        choose_a = float(window["A"].iloc[-1]) <= float(window["B"].iloc[-1])
        weights = {"A": float(choose_a), "B": float(not choose_a)}
        return object(), SimpleNamespace(weights=weights)

    monkeypatch.setattr(engine, "solve_spec", window_only_target)
    original = run_backtest(_spec(), prices, config=_monthly_config())
    perturbed_prices = prices.copy()
    perturbed_prices.loc[pd.Timestamp("2024-02-01") :, "A"] *= 50.0
    perturbed_prices.loc[pd.Timestamp("2024-02-01") :, "B"] *= 0.02
    perturbed = run_backtest(_spec(), perturbed_prices, config=_monthly_config())

    assert original.rebalances[0].target_weights == perturbed.rebalances[0].target_weights


def test_old_holdings_earn_to_fill_target_first_earns_after_fill_and_costs_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _single_month_panel()

    def target_b(*_: object, **__: object) -> object:
        return object(), SimpleNamespace(weights={"A": 0.0, "B": 1.0})

    monkeypatch.setattr(engine, "solve_spec", target_b)
    sheet = run_backtest(
        _spec(current_weights={"A": 1.0, "B": 0.0}),
        prices,
        config=_monthly_config(execution_cost_bps=100.0),
    )

    # Jan31->Feb1 belongs to old A: +10%.  Selling A and buying B is L1=2,
    # so 100 bps per L1 unit deducts 2% at the Feb1 close.  B's +20% is only
    # earned on Feb1->Feb2.
    np.testing.assert_allclose(sheet.curves[1].strategy_gross, 1.10)
    np.testing.assert_allclose(sheet.curves[1].strategy_net, 1.10 * 0.98)
    np.testing.assert_allclose(sheet.curves[2].strategy_gross, 1.10 * 1.20)
    np.testing.assert_allclose(sheet.curves[2].strategy_net, 1.10 * 0.98 * 1.20)

    record = sheet.rebalances[0]
    assert record.signal_date.isoformat() == "2024-01-31"
    assert record.fill_date.isoformat() == "2024-02-01"
    np.testing.assert_allclose(record.decision_turnover, 2.0)
    np.testing.assert_allclose(record.realized_turnover, 2.0)
    np.testing.assert_allclose(record.transaction_cost_fraction, 0.02)
    np.testing.assert_allclose(record.transaction_cost_paid, 0.022)
    assert record.holding_observations == 1
    np.testing.assert_allclose(record.realized_cvar, -0.20)
    np.testing.assert_allclose(sheet.summary.cost_drag, 1.32 - 1.2936)


def test_realized_turnover_uses_fill_weights_not_signal_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _single_month_panel().copy()
    prices.loc[:, "A"] = [100.0, 100.0, 100.0, 200.0, 200.0]
    prices.loc[:, "B"] = 100.0
    monkeypatch.setattr(
        engine,
        "solve_spec",
        lambda *_args, **_kwargs: (
            object(),
            SimpleNamespace(weights={"A": 0.5, "B": 0.5}),
        ),
    )

    sheet = run_backtest(
        _spec(current_weights={"A": 0.5, "B": 0.5}),
        prices,
        config=_monthly_config(),
    )
    record = sheet.rebalances[0]
    np.testing.assert_allclose(record.decision_turnover, 0.0)
    np.testing.assert_allclose(
        [record.fill_pretrade_weights["A"], record.fill_pretrade_weights["B"]],
        [2 / 3, 1 / 3],
    )
    np.testing.assert_allclose(record.realized_turnover, 1 / 3)


def test_each_solve_receives_the_drifted_signal_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.to_datetime(
        [
            "2024-01-29",
            "2024-01-30",
            "2024-01-31",
            "2024-02-01",
            "2024-02-02",
            "2024-02-28",
            "2024-02-29",
            "2024-03-01",
            "2024-03-04",
        ]
    )
    prices = pd.DataFrame(
        {
            "A": [100, 100, 100, 100, 100, 200, 200, 200, 200],
            "B": [100, 100, 100, 100, 100, 100, 100, 100, 100],
        },
        index=index,
        dtype=float,
    )
    seen: list[dict[str, float]] = []

    def half_and_half(spec: PortfolioSpec, *_: object, **__: object) -> object:
        assert spec.current_weights is not None
        seen.append(dict(spec.current_weights))
        return object(), SimpleNamespace(weights={"A": 0.5, "B": 0.5})

    monkeypatch.setattr(engine, "solve_spec", half_and_half)
    sheet = run_backtest(_spec(), prices, config=_monthly_config())

    assert len(seen) == 2
    np.testing.assert_allclose([seen[0]["A"], seen[0]["B"]], [0.0, 0.0])
    # The first target filled 50/50 on Feb1.  A then doubled before the Feb29
    # signal, so it is 2/3 of the drifted account; the old 50/50 target must
    # not be threaded into the second solve.
    np.testing.assert_allclose([seen[1]["A"], seen[1]["B"]], [2 / 3, 1 / 3])
    np.testing.assert_allclose(
        [
            sheet.rebalances[1].signal_weights["A"],
            sheet.rebalances[1].signal_weights["B"],
        ],
        [2 / 3, 1 / 3],
    )
    np.testing.assert_allclose(sheet.rebalances[1].decision_turnover, 1 / 3)
    assert [record.holding_observations for record in sheet.rebalances] == [4, 1]


def test_periods_per_year_is_forwarded_to_every_training_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int] = []

    def capture_periods(*_: object, **kwargs: object) -> object:
        seen.append(int(kwargs["periods_per_year"]))
        return object(), SimpleNamespace(weights={"A": 0.5, "B": 0.5})

    monkeypatch.setattr(engine, "solve_spec", capture_periods)
    run_backtest(
        _spec(),
        _single_month_panel(),
        config=_monthly_config(periods_per_year=12),
    )
    assert seen == [12]


def test_equal_weight_uses_the_same_fill_schedule_and_cost_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _single_month_panel()

    def target_b(*_: object, **__: object) -> object:
        return object(), SimpleNamespace(weights={"A": 0.0, "B": 1.0})

    monkeypatch.setattr(engine, "solve_spec", target_b)
    sheet = run_backtest(
        _spec(current_weights={"A": 1.0, "B": 0.0}),
        prices,
        config=_monthly_config(execution_cost_bps=100.0),
    )

    # Equal weight also retains old A through the fill (+10%), then trades L1=1
    # at 1% and earns its 50% share of B's subsequent +20% move.
    np.testing.assert_allclose(sheet.curves[1].equal_weight_net, 1.10 * 0.99)
    np.testing.assert_allclose(sheet.curves[2].equal_weight_net, 1.10 * 0.99 * 1.10)


def test_costs_are_summed_from_spec_unless_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        engine,
        "solve_spec",
        lambda *_args, **_kwargs: (
            object(),
            SimpleNamespace(weights={"A": 0.5, "B": 0.5}),
        ),
    )
    prices = _single_month_panel()

    inferred = run_backtest(
        _spec(costs=(5.0, 7.0)),
        prices,
        config=_monthly_config(),
    )
    assert inferred.summary.resolved_execution_cost_bps == 12.0
    assert inferred.summary.execution_cost_source == "spec_transaction_cost"

    overridden = run_backtest(
        _spec(costs=(5.0, 7.0)),
        prices,
        config=_monthly_config(execution_cost_bps=3.0),
    )
    assert overridden.summary.resolved_execution_cost_bps == 3.0
    assert overridden.summary.execution_cost_source == "config_override"


def test_modeled_and_realized_cvar_are_both_reported_from_their_own_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _single_month_panel().copy()
    # B loses 10%, then gains 10% in training.  After its delayed fill it gains
    # 20%, so the modeled and realized loss tails have distinct, hand-checkable
    # values.
    prices.loc[:, "B"] = [100.0, 90.0, 99.0, 99.0, 118.8]

    monkeypatch.setattr(
        engine,
        "solve_spec",
        lambda *_args, **_kwargs: (
            object(),
            SimpleNamespace(weights={"A": 0.0, "B": 1.0}),
        ),
    )
    sheet = run_backtest(_spec(), prices, config=_monthly_config(cvar_alpha=0.5))
    record = sheet.rebalances[0]
    np.testing.assert_allclose(record.modeled_cvar, empirical_cvar([-0.10, 0.10], 0.5))
    np.testing.assert_allclose(record.realized_cvar, empirical_cvar([0.20], 0.5))
    assert sheet.summary.holding_weighted_modeled_cvar == record.modeled_cvar
    assert sheet.summary.holding_weighted_realized_cvar == record.realized_cvar


def test_weekly_schedule_uses_last_observed_close_in_the_calendar_week(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Friday Jan5 is absent.  Thursday is still the final observed W-FRI close
    # because the next observation starts a new calendar week.
    index = pd.to_datetime(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-08",
            "2024-01-09",
        ]
    )
    prices = pd.DataFrame({"A": 100.0, "B": 100.0}, index=index)
    signals: list[pd.Timestamp] = []

    def capture(_spec: PortfolioSpec, window: pd.DataFrame, **_: object) -> object:
        signals.append(window.index[-1])
        return object(), SimpleNamespace(weights={"A": 0.5, "B": 0.5})

    monkeypatch.setattr(engine, "solve_spec", capture)
    run_backtest(
        _spec(),
        prices,
        config=BacktestConfig(lookback_returns=2, rebalance_frequency="weekly"),
    )
    assert signals == [pd.Timestamp("2024-01-04")]


def test_market_must_be_strictly_aligned_and_is_normalized_when_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _single_month_panel()
    monkeypatch.setattr(
        engine,
        "solve_spec",
        lambda *_args, **_kwargs: (
            object(),
            SimpleNamespace(weights={"A": 0.5, "B": 0.5}),
        ),
    )
    misaligned = pd.Series([100.0] * 4, index=prices.index[:-1])
    with pytest.raises(BacktestError, match="exactly the same ordered index"):
        run_backtest(_spec(), prices, config=_monthly_config(), market_prices=misaligned)

    market = pd.Series([80.0, 90.0, 100.0, 110.0, 121.0], index=prices.index)
    sheet = run_backtest(
        _spec(),
        prices,
        config=_monthly_config(),
        market_prices=market,
    )
    np.testing.assert_allclose(sheet.curves[-1].market, 121.0 / 100.0)
    np.testing.assert_allclose(sheet.summary.market.total_return, 0.21)  # type: ignore[union-attr]


def test_metrics_match_hand_calculations_including_fractional_cvar_tail() -> None:
    # Losses are [4%, 1%, 0%, -1%, -2%].  At alpha=.70 the tail mass is 1.5
    # observations: all of 4% plus half of 1%, divided by 1.5 = 3%.
    returns = [-0.04, -0.01, 0.0, 0.01, 0.02]
    np.testing.assert_allclose(empirical_cvar(returns, alpha=0.70), 0.03)

    np.testing.assert_allclose(annualized_return([0.10, -0.10], periods_per_year=2), -0.01)
    np.testing.assert_allclose(
        annualized_volatility([0.01, -0.01], periods_per_year=2),
        0.02,
    )
    np.testing.assert_allclose(
        sharpe_ratio([0.01, 0.03], periods_per_year=2),
        2.0,
    )
    np.testing.assert_allclose(max_drawdown([0.10, -0.20, 0.10]), -0.20)
    assert sharpe_ratio([0.01, 0.01], periods_per_year=252) is None
    with pytest.raises(ValueError, match="below -100%"):
        annualized_return([-2.0, -2.0])
    with pytest.raises(ValueError, match="must be finite"):
        sharpe_ratio([0.01, 0.02], annual_risk_free_rate=float("inf"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution_cost_bps", float("inf")),
        ("annual_risk_free_rate", float("inf")),
        ("annual_risk_free_rate", float("nan")),
    ],
)
def test_backtest_config_rejects_nonfinite_numbers(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite number"):
        BacktestConfig.model_validate({field: value})


@pytest.mark.parametrize(
    ("bad_prices", "message"),
    [
        (
            _single_month_panel().sort_index(ascending=False),
            "strictly increasing",
        ),
        (
            _single_month_panel().assign(A=np.nan),
            "finite values",
        ),
    ],
)
def test_invalid_panels_fail_loudly(bad_prices: pd.DataFrame, message: str) -> None:
    with pytest.raises(BacktestError, match=message):
        run_backtest(_spec(), bad_prices, config=_monthly_config())


def test_duplicate_dates_duplicate_tickers_and_nonpositive_prices_are_rejected() -> None:
    prices = _single_month_panel()
    duplicate_date = pd.concat([prices.iloc[:1], prices])
    with pytest.raises(BacktestError, match="duplicate observations"):
        run_backtest(_spec(), duplicate_date, config=_monthly_config())

    duplicate_ticker = prices.copy()
    duplicate_ticker.columns = ["A", "A"]
    with pytest.raises(BacktestError, match="duplicate ticker columns"):
        run_backtest(_spec(), duplicate_ticker, config=_monthly_config())

    nonpositive = prices.copy()
    nonpositive.iloc[0, 0] = 0.0
    with pytest.raises(BacktestError, match="strictly positive"):
        run_backtest(_spec(), nonpositive, config=_monthly_config())


def test_solver_report_must_name_every_universe_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        engine,
        "solve_spec",
        lambda *_args, **_kwargs: (object(), SimpleNamespace(weights={"A": 1.0})),
    )
    with pytest.raises(BacktestError, match=r"invalid weights.*missing=\['B'\]"):
        run_backtest(_spec(), _single_month_panel(), config=_monthly_config())

    monkeypatch.setattr(
        engine,
        "solve_spec",
        lambda *_args, **_kwargs: (
            object(),
            SimpleNamespace(weights={"A": "not-a-number", "B": 1.0}),
        ),
    )
    with pytest.raises(BacktestError, match=r"2024-01-31.*weights must all be numeric"):
        run_backtest(_spec(), _single_month_panel(), config=_monthly_config())


def test_insufficient_history_and_solver_failure_are_never_silently_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = _single_month_panel()
    with pytest.raises(BacktestError, match="No eligible calendar-period-end signal"):
        run_backtest(
            _spec(),
            prices,
            config=BacktestConfig(lookback_returns=4, rebalance_frequency="monthly"),
        )

    def fail(*_: object, **__: object) -> object:
        raise RuntimeError("deliberate solver failure")

    monkeypatch.setattr(engine, "solve_spec", fail)
    with pytest.raises(BacktestError, match=r"2024-01-31.*deliberate solver failure"):
        run_backtest(_spec(), prices, config=_monthly_config())


def test_tearsheet_has_no_wall_clock_fields_and_is_json_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        engine,
        "solve_spec",
        lambda *_args, **_kwargs: (
            object(),
            SimpleNamespace(weights={"A": 0.5, "B": 0.5}),
        ),
    )
    kwargs = {
        "spec": _spec(),
        "prices": _single_month_panel(),
        "config": _monthly_config(),
    }
    first = run_backtest(**kwargs).model_dump_json()
    second = run_backtest(**kwargs).model_dump_json()
    assert first == second
    assert "solve_time" not in first
    assert "wall_clock" not in first
