import pandas as pd
import pytest

from quantlab.data.market_loader import load_market_csv
from quantlab.data.trade_loader import load_trade_csv
from quantlab.trades.analysis import analyze_trades


def test_analyze_trades_end_to_end_from_csv() -> None:
    market_data = load_market_csv(
        "tests/fixtures/valid_etf.csv"
    )

    result = analyze_trades(
        market_data,
        "tests/fixtures/valid_trades.csv",
        initial_cash=1000.0,
    )

    assert len(result.trades) == 2
    assert len(result.fifo.matches) == 1
    assert result.fifo.open_lots.empty
    assert result.pnl.realized_pnl == pytest.approx(1.0)
    assert result.pnl.unrealized_pnl == 0.0
    assert result.pnl.total_pnl == pytest.approx(1.0)
    assert result.statistics is not None
    assert result.statistics.win_rate == 1.0
    assert result.statistics.profit_loss_ratio is None
    assert result.equity_curve.iloc[-1]["equity"] == pytest.approx(
        1001.0
    )


def test_analyze_trades_accepts_dataframe() -> None:
    market_data = load_market_csv(
        "tests/fixtures/valid_etf.csv"
    )
    trade_data = load_trade_csv(
        "tests/fixtures/valid_trades.csv"
    )

    result = analyze_trades(
        market_data,
        trade_data,
        initial_cash=1000.0,
    )

    assert result.trades.equals(trade_data)
    assert result.equity_curve.iloc[-1]["net_value"] == pytest.approx(
        1.001
    )


def test_analyze_trades_open_position_has_no_statistics() -> None:
    market_data = load_market_csv(
        "tests/fixtures/valid_etf.csv"
    )
    trade_data = load_trade_csv(
        "tests/fixtures/valid_trades.csv"
    ).iloc[[0]].copy()

    result = analyze_trades(
        market_data,
        trade_data,
        initial_cash=1000.0,
    )

    assert result.statistics is None
    assert result.pnl.realized_pnl == 0.0
    assert result.pnl.unrealized_pnl == pytest.approx(-0.5)
    assert result.equity_curve.iloc[-1]["equity"] == pytest.approx(
        999.5
    )
