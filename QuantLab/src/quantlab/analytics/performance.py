"""统一的收益与风险分析入口。"""

from dataclasses import dataclass

import pandas as pd

from quantlab.analytics.risk import (
    calculate_annualized_volatility,
    calculate_drawdown_curve,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)
from quantlab.analytics.returns import (
    calculate_annualized_return,
    calculate_daily_returns,
    calculate_net_value,
    calculate_total_return,
)
from quantlab.config.settings import DEFAULT_ANALYTICS_CONFIG


@dataclass(frozen=True)
class PerformanceAnalysis:
    """一次绩效分析产生的指标和曲线。"""

    daily_returns: pd.Series
    net_value: pd.Series
    drawdown: pd.Series
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    max_drawdown: float
    sharpe_ratio: float


def analyze_performance(
    prices: pd.Series,
    *,
    annual_risk_free_rate: float = (
        DEFAULT_ANALYTICS_CONFIG.annual_risk_free_rate
    ),
    periods_per_year: int = DEFAULT_ANALYTICS_CONFIG.periods_per_year,
) -> PerformanceAnalysis:
    """从价格序列生成统一的收益、风险和曲线结果。"""
    daily_returns = calculate_daily_returns(prices)
    net_value = calculate_net_value(prices)
    drawdown = calculate_drawdown_curve(net_value)

    return PerformanceAnalysis(
        daily_returns=daily_returns,
        net_value=net_value,
        drawdown=drawdown,
        cumulative_return=calculate_total_return(prices),
        annualized_return=calculate_annualized_return(
            prices,
            periods_per_year=periods_per_year,
        ),
        annualized_volatility=calculate_annualized_volatility(
            daily_returns,
            periods_per_year=periods_per_year,
        ),
        max_drawdown=calculate_max_drawdown(net_value),
        sharpe_ratio=calculate_sharpe_ratio(
            daily_returns,
            annual_risk_free_rate=annual_risk_free_rate,
            periods_per_year=periods_per_year,
        ),
    )
