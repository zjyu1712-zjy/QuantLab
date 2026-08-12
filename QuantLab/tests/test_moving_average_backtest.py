import pandas as pd
import pytest

from quantlab.backtest import run_dual_moving_average_backtest
from quantlab.config.settings import AnalyticsConfig, BacktestConfig


def make_market_data() -> pd.DataFrame:
    prices = [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0]
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=len(prices)),
            "symbol": pd.Series(
                ["510300"] * len(prices),
                dtype="string",
            ),
            "open": prices,
            "high": [price + 0.1 for price in prices],
            "low": [price - 0.1 for price in prices],
            "close": prices,
            "volume": [1000] * len(prices),
        }
    )


def test_moving_average_backtest_runs_complete_chain() -> None:
    result = run_dual_moving_average_backtest(
        make_market_data(),
        2,
        3,
        backtest_config=BacktestConfig(
            initial_cash=1000.0,
            commission_rate=0.0,
            slippage_rate=0.0,
            lot_size=10,
        ),
        analytics_config=AnalyticsConfig(periods_per_year=7),
    )

    assert result.signals["signal"].dropna().tolist() == [
        "BUY",
        "SELL",
    ]
    assert result.backtest.trades["side"].tolist() == ["BUY", "SELL"]
    assert result.backtest.trades["signal_date"].tolist() == [
        pd.Timestamp("2025-01-05"),
        pd.Timestamp("2025-01-07"),
    ]
    assert result.backtest.trades["execution_date"].tolist() == [
        pd.Timestamp("2025-01-06"),
        pd.Timestamp("2025-01-08"),
    ]
    assert result.backtest.equity_curve.iloc[-1]["equity"] == pytest.approx(
        1000.0
    )
    assert result.performance.cumulative_return == pytest.approx(0.0)
    assert result.performance.max_drawdown == pytest.approx(-0.5)


def test_moving_average_backtest_keeps_signal_and_execution_separate() -> None:
    result = run_dual_moving_average_backtest(
        make_market_data(),
        2,
        3,
        backtest_config=BacktestConfig(
            initial_cash=1000.0,
            commission_rate=0.0,
            slippage_rate=0.0,
            lot_size=10,
        ),
        analytics_config=AnalyticsConfig(periods_per_year=7),
    )

    for trade in result.backtest.trades.itertuples(index=False):
        signal_index = result.signals.index[
            result.signals["date"] == trade.signal_date
        ][0]
        execution_index = result.signals.index[
            result.signals["date"] == trade.execution_date
        ][0]
        assert execution_index == signal_index + 1
