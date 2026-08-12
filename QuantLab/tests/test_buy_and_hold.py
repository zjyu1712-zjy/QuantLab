import pandas as pd
import pytest

from quantlab.backtest import run_buy_and_hold
from quantlab.config.settings import AnalyticsConfig, BacktestConfig


def make_market_data() -> pd.DataFrame:
    prices = [10.0, 11.0, 12.0]
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3),
            "symbol": pd.Series(["510300"] * 3, dtype="string"),
            "open": prices,
            "high": [price + 0.5 for price in prices],
            "low": [price - 0.5 for price in prices],
            "close": prices,
            "volume": [1000] * 3,
        }
    )


def test_buy_and_hold_buys_first_day_and_tracks_equity() -> None:
    result = run_buy_and_hold(
        make_market_data(),
        backtest_config=BacktestConfig(
            initial_cash=1000.0,
            commission_rate=0.0,
            slippage_rate=0.0,
            lot_size=10,
        ),
        analytics_config=AnalyticsConfig(periods_per_year=2),
    )

    assert result.backtest.orders["status"].tolist() == ["FILLED"]
    assert result.backtest.trades["quantity"].tolist() == [100]
    assert result.backtest.equity_curve["equity"].tolist() == [
        1000.0,
        1100.0,
        1200.0,
    ]
    assert result.performance.cumulative_return == pytest.approx(0.20)


def test_buy_and_hold_applies_entry_costs() -> None:
    result = run_buy_and_hold(
        make_market_data(),
        backtest_config=BacktestConfig(
            initial_cash=1000.0,
            commission_rate=0.01,
            slippage_rate=0.01,
            lot_size=10,
        ),
        analytics_config=AnalyticsConfig(periods_per_year=2),
    )

    trade = result.backtest.trades.iloc[0]
    assert trade["execution_price"] == pytest.approx(10.10)
    assert trade["quantity"] == 90
    assert trade["commission"] == pytest.approx(9.09)
    assert result.backtest.equity_curve.iloc[0]["equity"] == pytest.approx(
        981.91
    )
    assert result.backtest.equity_curve.iloc[-1]["equity"] == pytest.approx(
        1161.91
    )
