import pandas as pd
import pytest

from quantlab.backtest import run_backtest
from quantlab.config.settings import BacktestConfig


def make_signal_data(
    signals: list[str | None],
    *,
    open_prices: list[float] | None = None,
) -> pd.DataFrame:
    length = len(signals)
    opens = open_prices or [10.0 + index for index in range(length)]

    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=length),
            "symbol": pd.Series(
                ["510300"] * length,
                dtype="string",
            ),
            "open": opens,
            "high": [price + 0.5 for price in opens],
            "low": [price - 0.5 for price in opens],
            "close": opens,
            "volume": [1000] * length,
            "signal": pd.Series(signals, dtype="string"),
        }
    )


def test_backtest_executes_signals_on_next_day_open() -> None:
    signal_data = make_signal_data(
        ["BUY", None, "SELL", None, None],
        open_prices=[10.0, 11.0, 12.0, 13.0, 14.0],
    )
    config = BacktestConfig(
        initial_cash=1000.0,
        commission_rate=0.0,
        slippage_rate=0.0,
        lot_size=10,
    )

    result = run_backtest(signal_data, config)

    assert result.trades["side"].tolist() == ["BUY", "SELL"]
    assert result.orders["status"].tolist() == ["FILLED", "FILLED"]
    assert result.orders["trade_id"].tolist() == [
        "BT000001",
        "BT000002",
    ]
    assert result.trades["signal_date"].tolist() == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-03"),
    ]
    assert result.trades["execution_date"].tolist() == [
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-04"),
    ]
    assert result.trades["quantity"].tolist() == [90, 90]
    assert result.equity_curve["quantity"].tolist() == [
        0,
        90,
        90,
        0,
        0,
    ]
    assert result.equity_curve.iloc[-1]["equity"] == pytest.approx(
        1180.0
    )


def test_backtest_applies_commission_slippage_and_tax() -> None:
    signal_data = make_signal_data(
        ["BUY", "SELL", None],
        open_prices=[9.0, 10.0, 12.0],
    )
    config = BacktestConfig(
        initial_cash=1000.0,
        commission_rate=0.01,
        slippage_rate=0.01,
        sell_tax_rate=0.01,
        lot_size=10,
    )

    result = run_backtest(signal_data, config)

    buy = result.trades.iloc[0]
    sell = result.trades.iloc[1]
    assert buy["execution_price"] == pytest.approx(10.10)
    assert buy["quantity"] == 90
    assert buy["commission"] == pytest.approx(9.09)
    assert sell["execution_price"] == pytest.approx(11.88)
    assert sell["commission"] == pytest.approx(10.692)
    assert sell["tax"] == pytest.approx(10.692)
    assert result.equity_curve.iloc[-1]["equity"] == pytest.approx(
        1129.726
    )


def test_backtest_skips_buy_when_cash_cannot_buy_one_lot() -> None:
    signal_data = make_signal_data(["BUY", None])
    config = BacktestConfig(
        initial_cash=50.0,
        commission_rate=0.0,
        slippage_rate=0.0,
        lot_size=100,
    )

    result = run_backtest(signal_data, config)

    assert result.trades.empty
    assert result.orders["status"].tolist() == ["REJECTED"]
    assert result.orders["reason"].tolist() == [
        "资金不足，无法买入一手"
    ]
    assert result.equity_curve["equity"].tolist() == [50.0, 50.0]


def test_backtest_does_not_execute_last_day_signal() -> None:
    signal_data = make_signal_data([None, "BUY"])
    config = BacktestConfig(
        initial_cash=1000.0,
        commission_rate=0.0,
        slippage_rate=0.0,
        lot_size=10,
    )

    result = run_backtest(signal_data, config)

    assert result.trades.empty
    assert result.orders["status"].tolist() == ["EXPIRED"]
    assert result.orders["reason"].tolist() == ["没有下一交易日"]
    assert pd.isna(result.orders.iloc[0]["execution_date"])
    assert result.equity_curve["quantity"].tolist() == [0, 0]


def test_backtest_records_rejected_position_signals() -> None:
    signal_data = make_signal_data(
        ["SELL", "BUY", "BUY", "SELL", None],
        open_prices=[10.0, 10.0, 11.0, 12.0, 13.0],
    )
    config = BacktestConfig(
        initial_cash=1000.0,
        commission_rate=0.0,
        slippage_rate=0.0,
        lot_size=10,
    )

    result = run_backtest(signal_data, config)

    assert result.orders["status"].tolist() == [
        "REJECTED",
        "FILLED",
        "REJECTED",
        "FILLED",
    ]
    assert result.orders["reason"].tolist() == [
        "当前空仓，无法卖出",
        "已成交",
        "已有持仓，忽略重复买入",
        "已成交",
    ]
    assert result.trades["side"].tolist() == ["BUY", "SELL"]
