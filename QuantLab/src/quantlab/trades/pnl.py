from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import pandas as pd

from quantlab.trades.fifo import FifoMatchResult


UNREALIZED_POSITION_COLUMNS = [
    "symbol",
    "buy_trade_id",
    "quantity",
    "buy_price",
    "buy_fee",
    "current_price",
    "cost_basis",
    "market_value",
    "unrealized_pnl",
]


@dataclass(frozen=True)
class TradePnlSummary:
    """真实交易的已实现、未实现和合计盈亏。"""

    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    open_positions: pd.DataFrame


def calculate_unrealized_positions(
    open_lots: pd.DataFrame,
    current_prices: Mapping[str, float],
) -> pd.DataFrame:
    """按当前价格计算每个未平仓 FIFO 批次的未实现盈亏。"""
    if open_lots.empty:
        return pd.DataFrame(columns=UNREALIZED_POSITION_COLUMNS)

    normalized_prices = {
        str(symbol): float(price)
        for symbol, price in current_prices.items()
    }
    position_records: list[dict[str, object]] = []

    for lot in open_lots.to_dict("records"):
        symbol = str(lot["symbol"])

        if symbol not in normalized_prices:
            raise ValueError(f"缺少 {symbol} 的当前价格")

        current_price = normalized_prices[symbol]
        if not isfinite(current_price) or current_price <= 0:
            raise ValueError(f"{symbol} 的当前价格必须大于 0")

        quantity = int(lot["remaining_quantity"])
        buy_price = float(lot["buy_price"])
        buy_fee = float(lot["remaining_fee"])
        cost_basis = buy_price * quantity + buy_fee
        market_value = current_price * quantity

        position_records.append(
            {
                "symbol": symbol,
                "buy_trade_id": lot["buy_trade_id"],
                "quantity": quantity,
                "buy_price": buy_price,
                "buy_fee": buy_fee,
                "current_price": current_price,
                "cost_basis": cost_basis,
                "market_value": market_value,
                "unrealized_pnl": market_value - cost_basis,
            }
        )

    return pd.DataFrame(
        position_records,
        columns=UNREALIZED_POSITION_COLUMNS,
    )


def summarize_trade_pnl(
    fifo_result: FifoMatchResult,
    current_prices: Mapping[str, float],
) -> TradePnlSummary:
    """汇总 FIFO 结果中的已实现与未实现盈亏。"""
    open_positions = calculate_unrealized_positions(
        fifo_result.open_lots,
        current_prices,
    )
    realized_pnl = float(fifo_result.matches["realized_pnl"].sum())
    unrealized_pnl = float(open_positions["unrealized_pnl"].sum())

    return TradePnlSummary(
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        total_pnl=realized_pnl + unrealized_pnl,
        open_positions=open_positions,
    )
