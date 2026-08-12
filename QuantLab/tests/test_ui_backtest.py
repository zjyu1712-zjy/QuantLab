import pandas as pd
import pytest

from quantlab.ui.backtest import (
    build_moving_average_backtest_view,
)


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


def test_build_moving_average_backtest_view() -> None:
    view = build_moving_average_backtest_view(
        make_market_data(),
        short_window=2,
        long_window=3,
        initial_cash=1000.0,
        commission_rate=0.0,
        minimum_commission=0.0,
        slippage_rate=0.0,
        sell_tax_rate=0.0,
        lot_size=10,
        periods_per_year=7,
    )

    assert view.symbol == "510300"
    assert view.signal_count == 2
    assert view.filled_trade_count == 2
    assert view.final_equity == pytest.approx(1000.0)
    assert view.total_costs == pytest.approx(0.0)
    assert len(view.chart_data) == 8
    assert {
        "sma_2",
        "sma_3",
        "signal",
        "equity",
        "net_value",
        "cumulative_costs",
    }.issubset(view.chart_data.columns)


def test_build_moving_average_backtest_view_applies_costs() -> None:
    view = build_moving_average_backtest_view(
        make_market_data(),
        short_window=2,
        long_window=3,
        initial_cash=1000.0,
        commission_rate=0.01,
        minimum_commission=0.0,
        slippage_rate=0.0,
        sell_tax_rate=0.0,
        lot_size=10,
        periods_per_year=7,
    )

    assert view.total_costs > 0
    assert view.final_equity < 1000.0
