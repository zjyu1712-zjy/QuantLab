from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantlab.data.trade_loader import load_trade_csv
from quantlab.trades.equity import build_trade_equity_curve
from quantlab.trades.fifo import (
    FifoMatchResult,
    match_trades_fifo,
)
from quantlab.trades.pnl import (
    TradePnlSummary,
    summarize_trade_pnl,
)
from quantlab.trades.statistics import (
    TradeStatistics,
    calculate_trade_statistics,
)


@dataclass(frozen=True)
class TradeAnalysis:
    """个人真实交易的一站式分析结果。"""

    trades: pd.DataFrame
    fifo: FifoMatchResult
    pnl: TradePnlSummary
    statistics: TradeStatistics | None
    equity_curve: pd.DataFrame


def analyze_trades(
    market_data: pd.DataFrame,
    trade_source: pd.DataFrame | str | Path,
    initial_cash: float,
) -> TradeAnalysis:
    """统一执行个人交易读取、匹配、盈亏、统计和资金曲线计算。"""
    trades = (
        load_trade_csv(trade_source)
        if isinstance(trade_source, (str, Path))
        else trade_source.copy()
    )
    fifo_result = match_trades_fifo(trades)

    latest_market = market_data.copy()
    latest_market["_date"] = pd.to_datetime(
        latest_market["date"],
        errors="raise",
    )
    latest_market = (
        latest_market.sort_values("_date")
        .groupby("symbol", as_index=False)
        .tail(1)
    )
    current_prices = {
        str(row["symbol"]): float(row["close"])
        for row in latest_market.to_dict("records")
    }

    pnl_summary = summarize_trade_pnl(
        fifo_result,
        current_prices,
    )
    statistics = (
        calculate_trade_statistics(fifo_result.matches)
        if not fifo_result.matches.empty
        else None
    )
    equity_curve = build_trade_equity_curve(
        market_data,
        trades,
        initial_cash,
    )

    return TradeAnalysis(
        trades=trades,
        fifo=fifo_result,
        pnl=pnl_summary,
        statistics=statistics,
        equity_curve=equity_curve,
    )
