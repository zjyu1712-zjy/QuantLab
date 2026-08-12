from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TradeStatistics:
    """FIFO 已平仓匹配记录的交易统计。"""

    closed_trade_count: int
    winning_trade_count: int
    losing_trade_count: int
    breakeven_trade_count: int
    win_rate: float
    average_profit: float | None
    average_loss: float | None
    profit_loss_ratio: float | None


def calculate_trade_statistics(
    matches: pd.DataFrame,
    *,
    zero_tolerance: float = 1e-12,
) -> TradeStatistics:
    """计算已平仓交易的胜率、平均盈亏和盈亏比。"""
    if zero_tolerance < 0:
        raise ValueError("零值容差不能为负数")

    if "realized_pnl" not in matches.columns:
        raise ValueError("已平仓记录缺少 realized_pnl 列")

    if matches.empty:
        raise ValueError("没有已平仓交易，无法计算交易统计")

    if matches["realized_pnl"].isna().any():
        raise ValueError("已平仓盈亏不能包含缺失值")

    realized_pnl = pd.to_numeric(
        matches["realized_pnl"],
        errors="raise",
    ).astype("float64")

    winning_pnl = realized_pnl[realized_pnl > zero_tolerance]
    losing_pnl = realized_pnl[realized_pnl < -zero_tolerance]
    breakeven_count = int(
        (realized_pnl.abs() <= zero_tolerance).sum()
    )

    closed_count = len(realized_pnl)
    winning_count = len(winning_pnl)
    losing_count = len(losing_pnl)
    average_profit = (
        float(winning_pnl.mean())
        if winning_count > 0
        else None
    )
    average_loss = (
        float(losing_pnl.mean())
        if losing_count > 0
        else None
    )
    profit_loss_ratio = (
        average_profit / abs(average_loss)
        if average_profit is not None and average_loss is not None
        else None
    )

    return TradeStatistics(
        closed_trade_count=closed_count,
        winning_trade_count=winning_count,
        losing_trade_count=losing_count,
        breakeven_trade_count=breakeven_count,
        win_rate=winning_count / closed_count,
        average_profit=average_profit,
        average_loss=average_loss,
        profit_loss_ratio=profit_loss_ratio,
    )
