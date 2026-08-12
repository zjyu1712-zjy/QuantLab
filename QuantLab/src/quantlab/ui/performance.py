from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from quantlab.analytics.performance import (
    PerformanceAnalysis,
    analyze_performance,
)


@dataclass(frozen=True)
class MarketPerformanceView:
    """投资表现页面需要的指标、曲线和数据范围。"""

    symbol: str
    start_date: date
    end_date: date
    observation_count: int
    analysis: PerformanceAnalysis
    detail: pd.DataFrame


def build_market_performance_view(
    market_data: pd.DataFrame,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    annual_risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> MarketPerformanceView:
    """根据一份标准化行情生成页面所需的统一绩效结果。"""
    required_columns = {"date", "symbol", "close"}
    missing_columns = required_columns - set(market_data.columns)
    if missing_columns:
        missing_text = "、".join(sorted(missing_columns))
        raise ValueError(f"行情数据缺少必要列：{missing_text}")

    selected = market_data.loc[
        :,
        ["date", "symbol", "close"],
    ].copy()
    selected["date"] = pd.to_datetime(selected["date"])
    selected = selected.sort_values("date").reset_index(drop=True)

    symbols = selected["symbol"].astype("string").unique()
    if len(symbols) != 1:
        raise ValueError("投资表现页面一次只能分析一只 ETF")

    if start_date is not None:
        selected = selected[
            selected["date"].dt.date >= start_date
        ]
    if end_date is not None:
        selected = selected[
            selected["date"].dt.date <= end_date
        ]
    selected = selected.reset_index(drop=True)

    if selected.empty:
        raise ValueError("所选日期范围内没有行情数据")

    if len(selected) < 3:
        raise ValueError("计算完整收益和风险指标至少需要 3 个价格观测值")

    prices = pd.Series(
        selected["close"].astype(float).to_numpy(),
        index=pd.DatetimeIndex(selected["date"]),
        name="close",
    )
    analysis = analyze_performance(
        prices,
        annual_risk_free_rate=annual_risk_free_rate,
        periods_per_year=periods_per_year,
    )

    detail = selected.copy()
    detail["daily_return"] = analysis.daily_returns.reindex(
        pd.DatetimeIndex(detail["date"])
    ).to_numpy()
    detail["net_value"] = analysis.net_value.to_numpy()
    detail["drawdown"] = analysis.drawdown.to_numpy()

    return MarketPerformanceView(
        symbol=str(symbols[0]),
        start_date=selected["date"].iloc[0].date(),
        end_date=selected["date"].iloc[-1].date(),
        observation_count=len(selected),
        analysis=analysis,
        detail=detail,
    )
