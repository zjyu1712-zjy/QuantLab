import pandas as pd
import pytest

from quantlab.data.trade_loader import load_trade_csv
from quantlab.trades.fifo import match_trades_fifo


def make_trade_data(records: list[dict[str, object]]) -> pd.DataFrame:
    trade_data = pd.DataFrame(records)
    trade_data["trade_id"] = trade_data["trade_id"].astype("string")
    trade_data["symbol"] = trade_data["symbol"].astype("string")
    trade_data["side"] = trade_data["side"].astype("string")
    trade_data["trade_datetime"] = pd.to_datetime(
        trade_data["trade_datetime"]
    ).dt.tz_localize("Asia/Shanghai")
    return trade_data


def test_fifo_matches_simple_round_trip_with_fees() -> None:
    trade_data = load_trade_csv("tests/fixtures/valid_trades.csv")

    result = match_trades_fifo(trade_data)

    assert len(result.matches) == 1
    assert result.open_lots.empty

    matched = result.matches.iloc[0]
    assert matched["buy_trade_id"] == "T001"
    assert matched["sell_trade_id"] == "T002"
    assert matched["quantity"] == 100
    assert matched["cost_basis"] == pytest.approx(406.50)
    assert matched["proceeds"] == pytest.approx(407.50)
    assert matched["realized_pnl"] == pytest.approx(1.00)


def test_fifo_uses_oldest_lot_and_allocates_fees() -> None:
    trade_data = make_trade_data(
        [
            {
                "trade_id": "B1",
                "trade_datetime": "2025-01-02 10:00:00",
                "symbol": "510300",
                "side": "BUY",
                "quantity": 100,
                "price": 10.0,
                "fee": 1.0,
            },
            {
                "trade_id": "B2",
                "trade_datetime": "2025-01-03 10:00:00",
                "symbol": "510300",
                "side": "BUY",
                "quantity": 100,
                "price": 12.0,
                "fee": 1.0,
            },
            {
                "trade_id": "S1",
                "trade_datetime": "2025-01-06 10:00:00",
                "symbol": "510300",
                "side": "SELL",
                "quantity": 150,
                "price": 15.0,
                "fee": 1.5,
            },
        ]
    )

    result = match_trades_fifo(trade_data)

    assert result.matches["buy_trade_id"].tolist() == ["B1", "B2"]
    assert result.matches["quantity"].tolist() == [100, 50]
    assert result.matches["buy_fee"].tolist() == pytest.approx(
        [1.0, 0.5]
    )
    assert result.matches["sell_fee"].tolist() == pytest.approx(
        [1.0, 0.5]
    )
    assert result.matches["realized_pnl"].tolist() == pytest.approx(
        [498.0, 149.0]
    )

    assert len(result.open_lots) == 1
    open_lot = result.open_lots.iloc[0]
    assert open_lot["buy_trade_id"] == "B2"
    assert open_lot["remaining_quantity"] == 50
    assert open_lot["remaining_fee"] == pytest.approx(0.5)


def test_fifo_rejects_sell_larger_than_position() -> None:
    trade_data = make_trade_data(
        [
            {
                "trade_id": "B1",
                "trade_datetime": "2025-01-02 10:00:00",
                "symbol": "510300",
                "side": "BUY",
                "quantity": 100,
                "price": 4.0,
                "fee": 0.5,
            },
            {
                "trade_id": "S1",
                "trade_datetime": "2025-01-03 10:00:00",
                "symbol": "510300",
                "side": "SELL",
                "quantity": 101,
                "price": 4.1,
                "fee": 0.5,
            },
        ]
    )

    with pytest.raises(ValueError, match="超过可用持仓 100"):
        match_trades_fifo(trade_data)
