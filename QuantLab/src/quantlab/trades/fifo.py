from collections import defaultdict, deque
from dataclasses import dataclass

import pandas as pd

from quantlab.data.trade_validator import validate_trade_data


MATCH_COLUMNS = [
    "symbol",
    "buy_trade_id",
    "sell_trade_id",
    "buy_datetime",
    "sell_datetime",
    "quantity",
    "buy_price",
    "sell_price",
    "buy_fee",
    "sell_fee",
    "cost_basis",
    "proceeds",
    "realized_pnl",
]

OPEN_LOT_COLUMNS = [
    "symbol",
    "buy_trade_id",
    "buy_datetime",
    "remaining_quantity",
    "buy_price",
    "remaining_fee",
]


@dataclass(frozen=True)
class FifoMatchResult:
    """FIFO 匹配产生的已平仓记录和剩余买入批次。"""

    matches: pd.DataFrame
    open_lots: pd.DataFrame


def match_trades_fifo(trade_data: pd.DataFrame) -> FifoMatchResult:
    """按证券代码分别执行 FIFO 匹配并计算已实现盈亏。"""
    validate_trade_data(trade_data)
    sorted_trades = trade_data.sort_values(
        by=["trade_datetime", "trade_id"],
    )

    open_lots: dict[str, deque[dict[str, object]]] = defaultdict(deque)
    match_records: list[dict[str, object]] = []

    for trade in sorted_trades.to_dict("records"):
        symbol = str(trade["symbol"])
        quantity = int(trade["quantity"])
        price = float(trade["price"])
        fee = float(trade["fee"])

        if trade["side"] == "BUY":
            open_lots[symbol].append(
                {
                    "symbol": symbol,
                    "buy_trade_id": str(trade["trade_id"]),
                    "buy_datetime": trade["trade_datetime"],
                    "remaining_quantity": quantity,
                    "buy_price": price,
                    "remaining_fee": fee,
                }
            )
            continue

        available_quantity = sum(
            int(lot["remaining_quantity"])
            for lot in open_lots[symbol]
        )
        if quantity > available_quantity:
            raise ValueError(
                f"{symbol} 卖出数量 {quantity} 超过可用持仓 "
                f"{available_quantity}"
            )

        remaining_sell_quantity = quantity

        while remaining_sell_quantity > 0:
            buy_lot = open_lots[symbol][0]
            lot_quantity = int(buy_lot["remaining_quantity"])
            matched_quantity = min(
                remaining_sell_quantity,
                lot_quantity,
            )

            buy_fee = (
                float(buy_lot["remaining_fee"])
                * matched_quantity
                / lot_quantity
            )
            sell_fee = fee * matched_quantity / quantity
            cost_basis = (
                float(buy_lot["buy_price"]) * matched_quantity
                + buy_fee
            )
            proceeds = price * matched_quantity - sell_fee

            match_records.append(
                {
                    "symbol": symbol,
                    "buy_trade_id": buy_lot["buy_trade_id"],
                    "sell_trade_id": str(trade["trade_id"]),
                    "buy_datetime": buy_lot["buy_datetime"],
                    "sell_datetime": trade["trade_datetime"],
                    "quantity": matched_quantity,
                    "buy_price": float(buy_lot["buy_price"]),
                    "sell_price": price,
                    "buy_fee": buy_fee,
                    "sell_fee": sell_fee,
                    "cost_basis": cost_basis,
                    "proceeds": proceeds,
                    "realized_pnl": proceeds - cost_basis,
                }
            )

            buy_lot["remaining_quantity"] = (
                lot_quantity - matched_quantity
            )
            buy_lot["remaining_fee"] = (
                float(buy_lot["remaining_fee"]) - buy_fee
            )
            remaining_sell_quantity -= matched_quantity

            if buy_lot["remaining_quantity"] == 0:
                open_lots[symbol].popleft()

    open_lot_records = [
        lot.copy()
        for symbol_lots in open_lots.values()
        for lot in symbol_lots
    ]

    return FifoMatchResult(
        matches=pd.DataFrame(
            match_records,
            columns=MATCH_COLUMNS,
        ),
        open_lots=pd.DataFrame(
            open_lot_records,
            columns=OPEN_LOT_COLUMNS,
        ),
    )
