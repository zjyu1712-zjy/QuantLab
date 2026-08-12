import pandas as pd
import pytest

from quantlab.backtest import analyze_backtest
from quantlab.config.settings import AnalyticsConfig, BacktestConfig


def make_signal_data() -> pd.DataFrame:
    opens = [10.0, 10.0, 11.0, 12.0, 13.0]
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5),
            "symbol": pd.Series(["510300"] * 5, dtype="string"),
            "open": opens,
            "high": [price + 0.5 for price in opens],
            "low": [price - 0.5 for price in opens],
            "close": opens,
            "volume": [1000] * 5,
            "signal": pd.Series(
                ["BUY", None, "SELL", None, None],
                dtype="string",
            ),
        }
    )


def test_analyze_backtest_connects_equity_and_performance() -> None:
    result = analyze_backtest(
        make_signal_data(),
        backtest_config=BacktestConfig(
            initial_cash=1000.0,
            commission_rate=0.0,
            slippage_rate=0.0,
            lot_size=10,
        ),
        analytics_config=AnalyticsConfig(periods_per_year=4),
    )

    assert result.backtest.trades["side"].tolist() == ["BUY", "SELL"]
    assert result.backtest.equity_curve["equity"].tolist() == [
        1000.0,
        1000.0,
        1100.0,
        1200.0,
        1200.0,
    ]
    assert result.performance.cumulative_return == pytest.approx(0.20)
    assert result.performance.annualized_return == pytest.approx(0.20)
    assert result.performance.max_drawdown == pytest.approx(0.0)


def test_analyze_backtest_uses_analytics_risk_free_rate() -> None:
    without_risk_free = analyze_backtest(
        make_signal_data(),
        backtest_config=BacktestConfig(
            initial_cash=1000.0,
            commission_rate=0.0,
            slippage_rate=0.0,
            lot_size=10,
        ),
        analytics_config=AnalyticsConfig(
            periods_per_year=4,
            annual_risk_free_rate=0.0,
        ),
    )
    with_risk_free = analyze_backtest(
        make_signal_data(),
        backtest_config=BacktestConfig(
            initial_cash=1000.0,
            commission_rate=0.0,
            slippage_rate=0.0,
            lot_size=10,
        ),
        analytics_config=AnalyticsConfig(
            periods_per_year=4,
            annual_risk_free_rate=0.04,
        ),
    )

    assert (
        with_risk_free.performance.sharpe_ratio
        < without_risk_free.performance.sharpe_ratio
    )
