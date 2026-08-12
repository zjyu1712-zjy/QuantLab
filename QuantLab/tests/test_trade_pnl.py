import pandas as pd
import pytest

from quantlab.data.trade_loader import load_trade_csv
from quantlab.trades import (
    match_trades_fifo,
    summarize_trade_pnl,
)


def make_open_position_trade() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_id": "B1",
                "trade_datetime": pd.Timestamp(
                    "2025-01-02 10:00:00",
                    tz="Asia/Shanghai",
                ),
                "symbol": "510300",
                "side": "BUY",
                "quantity": 100,
                "price": 10.0,
                "fee": 1.0,
            }
        ]
    ).astype(
        {
            "trade_id": "string",
            "symbol": "string",
            "side": "string",
        }
    )


def test_summarize_trade_pnl_for_open_position() -> None:
    fifo_result = match_trades_fifo(make_open_position_trade())

    summary = summarize_trade_pnl(
        fifo_result,
        {"510300": 12.0},
    )

    assert summary.realized_pnl == 0.0
    assert summary.unrealized_pnl == pytest.approx(199.0)
    assert summary.total_pnl == pytest.approx(199.0)

    position = summary.open_positions.iloc[0]
    assert position["quantity"] == 100
    assert position["cost_basis"] == pytest.approx(1001.0)
    assert position["market_value"] == pytest.approx(1200.0)
    assert position["unrealized_pnl"] == pytest.approx(199.0)


def test_summarize_trade_pnl_for_closed_position() -> None:
    trade_data = load_trade_csv("tests/fixtures/valid_trades.csv")
    fifo_result = match_trades_fifo(trade_data)

    summary = summarize_trade_pnl(fifo_result, {})

    assert summary.realized_pnl == pytest.approx(1.0)
    assert summary.unrealized_pnl == 0.0
    assert summary.total_pnl == pytest.approx(1.0)
    assert summary.open_positions.empty


def test_summarize_trade_pnl_requires_current_price() -> None:
    fifo_result = match_trades_fifo(make_open_position_trade())

    with pytest.raises(ValueError, match="缺少 510300 的当前价格"):
        summarize_trade_pnl(fifo_result, {})


def test_summarize_trade_pnl_rejects_invalid_current_price() -> None:
    fifo_result = match_trades_fifo(make_open_position_trade())

    with pytest.raises(ValueError, match="当前价格必须大于 0"):
        summarize_trade_pnl(
            fifo_result,
            {"510300": 0.0},
        )
