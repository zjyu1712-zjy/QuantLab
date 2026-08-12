"""基础收益率指标。"""

import pandas as pd
from pandas.api.types import is_numeric_dtype

from quantlab.config.settings import DEFAULT_ANALYTICS_CONFIG


def _validate_prices(
    prices: pd.Series,
    *,
    minimum_observations: int = 2,
) -> None:
    """检查价格序列能否用于收益率计算。"""
    if not isinstance(prices, pd.Series):
        raise TypeError("价格必须是 pandas Series")

    if len(prices) < minimum_observations:
        raise ValueError(f"价格序列至少需要 {minimum_observations} 个观测值")

    if prices.isna().any():
        raise ValueError("价格序列不能包含空值")

    if not is_numeric_dtype(prices):
        raise ValueError("价格序列必须是数值类型")

    if (prices <= 0).any():
        raise ValueError("价格必须大于0")


def calculate_daily_returns(prices: pd.Series) -> pd.Series:
    """根据相邻收盘价计算简单日收益率。"""
    _validate_prices(prices)

    daily_returns = (
        prices.astype(float)
        .pct_change(fill_method=None)
        .dropna()
    )
    daily_returns.name = "daily_return"
    return daily_returns


def calculate_total_return(prices: pd.Series) -> float:
    """计算整个观察区间的累计收益率。"""
    _validate_prices(prices)

    return float(prices.iloc[-1] / prices.iloc[0] - 1)


def calculate_net_value(prices: pd.Series) -> pd.Series:
    """把价格序列标准化为从1开始的净值曲线。"""
    _validate_prices(
        prices,
        minimum_observations=1,
    )

    net_value = prices.astype(float) / float(prices.iloc[0])
    net_value.name = "net_value"
    return net_value


def calculate_annualized_return(
    prices: pd.Series,
    *,
    periods_per_year: int = DEFAULT_ANALYTICS_CONFIG.periods_per_year,
) -> float:
    """按观察期复合增长率计算年化收益率。"""
    _validate_prices(prices)

    if periods_per_year <= 0:
        raise ValueError("每年期数必须大于0")

    observed_periods = len(prices) - 1
    growth_multiple = float(prices.iloc[-1] / prices.iloc[0])

    return growth_multiple ** (periods_per_year / observed_periods) - 1
