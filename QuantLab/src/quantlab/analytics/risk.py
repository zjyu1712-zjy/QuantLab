"""基础风险指标。"""

from math import sqrt

import pandas as pd
from pandas.api.types import is_numeric_dtype

from quantlab.config.settings import DEFAULT_ANALYTICS_CONFIG


def _validate_daily_returns(
    daily_returns: pd.Series,
    *,
    minimum_observations: int = 2,
) -> None:
    """检查日收益率序列能否用于风险指标计算。"""
    if not isinstance(daily_returns, pd.Series):
        raise TypeError("日收益率必须是 pandas Series")

    if len(daily_returns) < minimum_observations:
        raise ValueError(
            f"日收益率至少需要 {minimum_observations} 个观测值"
        )

    if daily_returns.isna().any():
        raise ValueError("日收益率不能包含空值")

    if not is_numeric_dtype(daily_returns):
        raise ValueError("日收益率必须是数值类型")


def calculate_annualized_volatility(
    daily_returns: pd.Series,
    *,
    periods_per_year: int = DEFAULT_ANALYTICS_CONFIG.periods_per_year,
) -> float:
    """用日收益率样本标准差计算年化波动率。"""
    _validate_daily_returns(daily_returns)

    if periods_per_year <= 0:
        raise ValueError("每年期数必须大于0")

    daily_volatility = daily_returns.astype(float).std(ddof=1)

    return float(daily_volatility * sqrt(periods_per_year))


def calculate_sharpe_ratio(
    daily_returns: pd.Series,
    *,
    annual_risk_free_rate: float = (
        DEFAULT_ANALYTICS_CONFIG.annual_risk_free_rate
    ),
    periods_per_year: int = DEFAULT_ANALYTICS_CONFIG.periods_per_year,
) -> float:
    """计算使用复合日无风险利率的年化夏普比率。"""
    _validate_daily_returns(daily_returns)

    if periods_per_year <= 0:
        raise ValueError("每年期数必须大于0")

    if annual_risk_free_rate <= -1:
        raise ValueError("年化无风险利率必须大于-100%")

    numeric_returns = daily_returns.astype(float)
    daily_volatility = numeric_returns.std(ddof=1)

    if daily_volatility <= 1e-15:
        raise ValueError("收益率波动率为0，无法计算夏普比率")

    daily_risk_free_rate = (
        (1 + annual_risk_free_rate) ** (1 / periods_per_year)
        - 1
    )
    daily_excess_returns = numeric_returns - daily_risk_free_rate

    return float(
        daily_excess_returns.mean()
        / daily_volatility
        * sqrt(periods_per_year)
    )


def _validate_net_value(net_value: pd.Series) -> None:
    """检查净值序列能否用于回撤计算。"""
    if not isinstance(net_value, pd.Series):
        raise TypeError("净值必须是 pandas Series")

    if net_value.empty:
        raise ValueError("净值序列不能为空")

    if net_value.isna().any():
        raise ValueError("净值序列不能包含空值")

    if not is_numeric_dtype(net_value):
        raise ValueError("净值序列必须是数值类型")

    if (net_value <= 0).any():
        raise ValueError("净值必须大于0")


def calculate_drawdown_curve(net_value: pd.Series) -> pd.Series:
    """计算每个观测时点相对历史最高净值的回撤。"""
    _validate_net_value(net_value)

    running_maximum = net_value.astype(float).cummax()
    drawdown = net_value.astype(float) / running_maximum - 1
    drawdown.name = "drawdown"
    return drawdown


def calculate_max_drawdown(net_value: pd.Series) -> float:
    """返回净值曲线中最深的回撤，结果为0或负数。"""
    drawdown = calculate_drawdown_curve(net_value)

    return float(drawdown.min())
