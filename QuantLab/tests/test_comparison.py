import pandas as pd
import pytest

from quantlab.comparison import compare_three_approaches
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


def make_trade_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_id": pd.Series(["T001", "T002"], dtype="string"),
            "trade_datetime": pd.to_datetime(
                [
                    "2025-01-02 10:00:00+08:00",
                    "2025-01-06 10:00:00+08:00",
                ],
                utc=True,
            ).tz_convert("Asia/Shanghai"),
            "symbol": pd.Series(["510300", "510300"], dtype="string"),
            "side": pd.Series(["BUY", "SELL"], dtype="string"),
            "quantity": pd.Series([100, 100], dtype="int64"),
            "price": [2.0, 2.0],
            "fee": [0.5, 0.5],
        }
    )


def test_compare_three_approaches_aligns_curves_and_metrics() -> None:
    result = compare_three_approaches(
        make_market_data(),
        make_trade_data(),
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

    assert result.equity_curves.columns.tolist() == [
        "date",
        "buy_and_hold",
        "dual_moving_average",
        "personal_trades",
    ]
    assert len(result.equity_curves) == 8
    assert result.metrics["approach"].tolist() == [
        "buy_and_hold",
        "dual_moving_average",
        "personal_trades",
    ]
    assert result.metrics["final_equity"].tolist() == pytest.approx(
        [670.0, 1000.0, 999.0]
    )


def test_compare_three_approaches_calculates_return_gaps() -> None:
    result = compare_three_approaches(
        make_market_data(),
        make_trade_data(),
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

    assert result.return_gaps[
        "strategy_minus_buy_and_hold"
    ] == pytest.approx(0.33)
    assert result.return_gaps[
        "personal_minus_strategy"
    ] == pytest.approx(-0.001)
    assert result.return_gaps[
        "personal_minus_buy_and_hold"
    ] == pytest.approx(0.329)
