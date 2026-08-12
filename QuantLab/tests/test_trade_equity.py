import pandas as pd
import pytest

from quantlab.data.market_loader import load_market_csv
from quantlab.data.trade_loader import load_trade_csv
from quantlab.trades.equity import build_trade_equity_curve


def test_build_trade_equity_curve() -> None:
    market_data = load_market_csv(
        "tests/fixtures/valid_etf.csv"
    )
    trade_data = load_trade_csv(
        "tests/fixtures/valid_trades.csv"
    )

    curve = build_trade_equity_curve(
        market_data,
        trade_data,
        initial_cash=1000.0,
    )

    assert curve["quantity"].tolist() == [0, 100, 0, 0, 0]
    assert curve["cash"].tolist() == pytest.approx(
        [1000.0, 593.5, 1001.0, 1001.0, 1001.0]
    )
    assert curve["equity"].tolist() == pytest.approx(
        [1000.0, 999.5, 1001.0, 1001.0, 1001.0]
    )
    assert curve["net_value"].tolist() == pytest.approx(
        [1.0, 0.9995, 1.001, 1.001, 1.001]
    )
    assert curve["cumulative_fees"].tolist() == pytest.approx(
        [0.0, 0.5, 1.0, 1.0, 1.0]
    )


def test_trade_equity_curve_rejects_insufficient_cash() -> None:
    market_data = load_market_csv(
        "tests/fixtures/valid_etf.csv"
    )
    trade_data = load_trade_csv(
        "tests/fixtures/valid_trades.csv"
    )

    with pytest.raises(ValueError, match="超过可用现金"):
        build_trade_equity_curve(
            market_data,
            trade_data,
            initial_cash=400.0,
        )


def test_trade_equity_curve_rejects_non_trading_date() -> None:
    market_data = load_market_csv(
        "tests/fixtures/valid_etf.csv"
    )
    trade_data = load_trade_csv(
        "tests/fixtures/valid_trades.csv"
    )
    trade_data.loc[0, "trade_datetime"] = pd.Timestamp(
        "2025-01-04 10:00:00",
        tz="Asia/Shanghai",
    )

    with pytest.raises(ValueError, match="2025-01-04"):
        build_trade_equity_curve(
            market_data,
            trade_data,
            initial_cash=1000.0,
        )


def test_trade_equity_curve_rejects_symbol_mismatch() -> None:
    market_data = load_market_csv(
        "tests/fixtures/valid_etf.csv"
    )
    trade_data = load_trade_csv(
        "tests/fixtures/valid_trades.csv"
    )
    trade_data["symbol"] = "159915"

    with pytest.raises(ValueError, match="证券代码不一致"):
        build_trade_equity_curve(
            market_data,
            trade_data,
            initial_cash=1000.0,
        )
