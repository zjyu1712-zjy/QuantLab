from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantlab.trade_review import (
    TradeSignalReview,
    review_trades_against_signals,
)


TIMING_LABELS = {
    "ON_TIME": "按模型日期成交",
    "EARLY": "提前成交",
    "LATE": "延迟成交",
    "NO_MATCH": "模型外交易",
    "MISSED_SIGNAL": "未执行模型信号",
}


@dataclass(frozen=True)
class TradeReviewView:
    """交易复盘页面使用的汇总、明细和行动提示。"""

    symbol: str
    review: TradeSignalReview
    final_equity: float
    cumulative_return: float
    win_rate: float | None
    profit_loss_ratio: float | None
    timing_summary: pd.DataFrame
    top_winners: pd.DataFrame
    top_losers: pd.DataFrame
    recommendations: tuple[str, ...]


def _rank_contributors(
    matches: pd.DataFrame,
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if matches.empty:
        return matches.copy(), matches.copy()

    ranked = matches.copy()
    ranked["return_on_cost"] = (
        ranked["realized_pnl"] / ranked["cost_basis"]
    )
    winners = (
        ranked.loc[ranked["realized_pnl"] > 0]
        .nlargest(top_n, "realized_pnl")
        .reset_index(drop=True)
    )
    losers = (
        ranked.loc[ranked["realized_pnl"] < 0]
        .nsmallest(top_n, "realized_pnl")
        .reset_index(drop=True)
    )
    return winners, losers


def _build_recommendations(
    review: TradeSignalReview,
) -> tuple[str, ...]:
    counts = review.timing_counts
    recommendations: list[str] = []
    if counts["NO_MATCH"]:
        recommendations.append(
            f"复核 {counts['NO_MATCH']} 笔模型外交易的主观依据，"
            "确认是否属于计划内例外。"
        )
    if counts["MISSED_SIGNAL"]:
        recommendations.append(
            f"复核 {counts['MISSED_SIGNAL']} 个未执行模型信号，"
            "记录当时未执行的原因。"
        )
    if counts["EARLY"] or counts["LATE"]:
        recommendations.append(
            f"本区间有 {counts['EARLY']} 笔提前、"
            f"{counts['LATE']} 笔延迟成交，可结合当时信息检查择时偏差。"
        )
    if review.fee_to_initial_cash_rate > 0.005:
        recommendations.append(
            "累计手续费已超过初始资金的 0.5%，需要重点检查交易频率。"
        )
    if "note" not in review.analysis.trades.columns:
        recommendations.append(
            "可在交易 CSV 中增加 note 列，记录交易理由，便于后续行为复盘。"
        )
    if not recommendations:
        recommendations.append(
            "实际交易与模型时点基本一致；继续记录交易理由并观察长期稳定性。"
        )
    return tuple(recommendations)


def build_trade_review_view(
    market_data: pd.DataFrame,
    trade_data: pd.DataFrame,
    *,
    short_window: int,
    long_window: int,
    initial_cash: float,
    tolerance_trading_days: int = 3,
) -> TradeReviewView:
    """生成个人交易盈亏、费用和模型信号对齐视图。"""
    review = review_trades_against_signals(
        market_data,
        trade_data,
        short_window,
        long_window,
        initial_cash=initial_cash,
        tolerance_trading_days=tolerance_trading_days,
    )
    equity_curve = review.analysis.equity_curve
    final_equity = float(equity_curve["equity"].iloc[-1])
    cumulative_return = float(
        equity_curve["cumulative_return"].iloc[-1]
    )
    statistics = review.analysis.statistics
    win_rate = statistics.win_rate if statistics is not None else None
    profit_loss_ratio = (
        statistics.profit_loss_ratio
        if statistics is not None
        else None
    )
    timing_summary = pd.DataFrame(
        [
            {
                "status": status,
                "label": TIMING_LABELS[status],
                "count": count,
            }
            for status, count in review.timing_counts.items()
        ]
    )
    top_winners, top_losers = _rank_contributors(
        review.analysis.fifo.matches
    )

    return TradeReviewView(
        symbol=str(equity_curve["symbol"].iloc[0]),
        review=review,
        final_equity=final_equity,
        cumulative_return=cumulative_return,
        win_rate=win_rate,
        profit_loss_ratio=profit_loss_ratio,
        timing_summary=timing_summary,
        top_winners=top_winners,
        top_losers=top_losers,
        recommendations=_build_recommendations(review),
    )
