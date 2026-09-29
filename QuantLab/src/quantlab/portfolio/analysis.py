"""完整组合曲线的指标适配；复用既有公式，区分未定义指标和运行错误。"""

from copy import deepcopy
from datetime import date
from math import isfinite
from numbers import Integral, Real

import numpy as np
import pandas as pd

from quantlab.analytics.returns import (
    calculate_annualized_return, calculate_daily_returns, calculate_net_value, calculate_total_return,
)
from quantlab.analytics.risk import (
    calculate_annualized_volatility, calculate_drawdown_curve, calculate_max_drawdown, calculate_sharpe_ratio,
)
from quantlab.backtest.provenance import dataframe_sha256
from quantlab.config.settings import AnalyticsConfig, DEFAULT_ANALYTICS_CONFIG
from quantlab.portfolio.types import (
    EQUITY_DTYPES, PortfolioAnalysis, PortfolioBacktestError, PortfolioBacktestResult, PortfolioPerformance,
)


def _fail(message: str, reason: str = "invalid_portfolio_result") -> None:
    raise PortfolioBacktestError(message, reason=reason)


def _checked_config(config: AnalyticsConfig) -> tuple[int, float]:
    """不改变旧配置类；组合边界额外拒绝布尔整数、非有限值等歧义。"""
    if not isinstance(config, AnalyticsConfig):
        _fail("指标配置必须为 AnalyticsConfig", "invalid_analytics_config")
    periods, rate = config.periods_per_year, config.annual_risk_free_rate
    if (isinstance(periods, bool) or not isinstance(periods, Integral) or periods <= 0
            or isinstance(rate, bool) or not isinstance(rate, Real)):
        _fail("每年期数必须为正整数，无风险利率必须为实数，不能为布尔值", "invalid_analytics_config")
    try:
        finite = isfinite(periods) and isfinite(rate)
    except OverflowError:
        finite = False
    if not finite or rate <= -1:
        _fail("指标参数必须有限，无风险年利率必须大于 -100%", "invalid_analytics_config")
    return int(periods), float(rate)


def _checked_equity(result: PortfolioBacktestResult) -> tuple[PortfolioBacktestResult, pd.Series]:
    """只接受完整研究结果，不补日期、不删除坏值，也不重放交易以认证结果。"""
    if not isinstance(result, PortfolioBacktestResult):
        _fail("分析输入必须是完整的 PortfolioBacktestResult")
    copied = deepcopy(result)
    meta = copied.metadata
    if (not isinstance(meta, dict) or meta.get("schema_version") != "portfolio-run-v1"
            or meta.get("status") != "complete"):
        _fail("组合结果缺少完整运行身份，不能分析内部引擎或失败结果")
    try:
        axis_values = meta["data"]["calendar"]["dates"]
        initial_cash = meta["strategy"]["config"]["backtest"]["initial_cash"]
    except (KeyError, TypeError) as error:
        raise PortfolioBacktestError("组合结果缺少日期轴或初始资金记录", reason="invalid_portfolio_result") from error
    if not isinstance(axis_values, list) or not axis_values:
        _fail("组合结果日期轴必须是非空 ISO 日期列表")
    try:
        if any(not isinstance(value, str) or date.fromisoformat(value).isoformat() != value for value in axis_values):
            _fail("组合结果日期轴必须使用规范 ISO 日期")
        dates = pd.DatetimeIndex(axis_values, dtype="datetime64[ns]", name="date")
    except (ValueError, TypeError, OverflowError) as error:
        raise PortfolioBacktestError("组合结果日期轴无效", reason="invalid_portfolio_result") from error
    if not dates.is_unique or not dates.is_monotonic_increasing or dates.hasnans:
        _fail("组合结果日期轴必须升序、无重复且无空值")
    if isinstance(initial_cash, bool) or not isinstance(initial_cash, Real):
        _fail("组合结果初始资金记录无效")
    try:
        finite_cash = isfinite(initial_cash)
    except OverflowError:
        finite_cash = False
    if not finite_cash or initial_cash <= 0:
        _fail("组合结果初始资金记录必须为有限正数")

    for name in ("decisions", "targets", "orders", "trades", "positions", "equity_curve", "input_data", "factor_data"):
        if not isinstance(getattr(copied, name), pd.DataFrame):
            _fail(f"完整组合结果缺少 {name} 数据表")
    for table, key in ((copied.input_data, "market_sha256"), (copied.factor_data, "factor_sha256")):
        if (table.empty or not table.columns.is_unique or "date" not in table
                or table.date.dtype != np.dtype("datetime64[ns]")):
            _fail("组合输入快照缺少规范日期或为空")
        if not pd.DatetimeIndex(table.date.drop_duplicates(), name="date").equals(dates):
            _fail("组合输入快照与原始完整日期轴不一致")
        try:
            digest = dataframe_sha256(table)
        except (ValueError, TypeError, OverflowError) as error:
            raise PortfolioBacktestError("组合输入快照不能计算内容指纹", reason="invalid_portfolio_result") from error
        if digest != meta["data"].get(key):
            _fail("组合输入快照内容指纹不一致，不自动修复身份")

    curve = copied.equity_curve
    if (curve.columns.tolist() != list(EQUITY_DTYPES)
            or any(str(curve[key].dtype) != dtype for key, dtype in EQUITY_DTYPES.items())):
        _fail("组合资金曲线必须保留规范字段、日期和数值类型")
    if not pd.DatetimeIndex(curve.date, name="date").equals(dates):
        _fail("组合资金曲线必须包含原始完整日期轴，不能缺失、重复或重排日期")
    values = curve.equity.to_numpy(dtype="float64", copy=True)
    if not np.isfinite(values).all() or (values <= 0).any():
        _fail("组合总资产必须为有限正数，不能包含空值、无穷或非正值")
    if values[0] != initial_cash:
        _fail("组合资金曲线必须从原始初始资金开始，不能丢失热身或首次成本")
    return copied, pd.Series(values, index=dates, name="equity", dtype="float64")


def _calculate(label, function, *args, positive=False, **kwargs):
    """只转译算术范围错误；不能把程序错误或非有限结果变成 None。"""
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            value = function(*args, **kwargs)
    except (OverflowError, FloatingPointError, ZeroDivisionError) as error:
        raise PortfolioBacktestError(f"{label}超出可靠数值范围", reason="numeric_range") from error
    numbers = value.to_numpy(dtype="float64") if isinstance(value, pd.Series) else np.asarray(value)
    if not np.isfinite(numbers).all() or (positive and (numbers <= 0).any()):
        _fail(f"{label}产生非有限或下溢的数值", "numeric_range")
    return value


def analyze_portfolio_result(
    result: PortfolioBacktestResult, *, analytics_config: AnalyticsConfig = DEFAULT_ANALYTICS_CONFIG,
) -> PortfolioAnalysis:
    """从全部研究日期计算指标，返回独立副本，不改原运行身份或已有公式。"""
    periods, risk_free_rate = _checked_config(analytics_config)
    backtest, equity = _checked_equity(result)
    net_value = _calculate("净值", calculate_net_value, equity, positive=True)
    drawdown = _calculate("回撤", calculate_drawdown_curve, net_value)
    max_drawdown = _calculate("最大回撤", calculate_max_drawdown, net_value)
    reasons = {}
    total = annualized = volatility = sharpe = None
    daily_returns = pd.Series(index=equity.index[:0], dtype="float64", name="daily_return")
    if len(equity) < 2:
        reasons.update({key: "至少需要 2 个资产观测值，才能形成一个收益区间"
                        for key in ("cumulative_return", "annualized_return")})
    else:
        daily_returns = _calculate("每日收益率", calculate_daily_returns, equity)
        total = _calculate("累计收益率", calculate_total_return, equity)
        annualized = _calculate("年化收益率", calculate_annualized_return, equity, periods_per_year=periods)
    if len(daily_returns) < 2:
        reasons.update({key: "至少需要 3 个资产观测值（2 个收益率），才能计算样本波动"
                        for key in ("annualized_volatility", "sharpe_ratio")})
    else:
        volatility = _calculate("年化波动率", calculate_annualized_volatility, daily_returns, periods_per_year=periods)
        daily_std = _calculate("日收益率样本标准差", daily_returns.std, ddof=1)
        if daily_std <= 1e-15:
            reasons["sharpe_ratio"] = "日收益率样本波动率不高于 1e-15，夏普比率未定义"
        else:
            sharpe = _calculate("夏普比率", calculate_sharpe_ratio, daily_returns,
                                periods_per_year=periods, annual_risk_free_rate=risk_free_rate)
    backtest.metadata["analytics_config"] = {
        "periods_per_year": periods, "annual_risk_free_rate": risk_free_rate,
    }
    performance = PortfolioPerformance(
        daily_returns=daily_returns, net_value=net_value, drawdown=drawdown,
        cumulative_return=total, annualized_return=annualized, annualized_volatility=volatility,
        max_drawdown=max_drawdown, sharpe_ratio=sharpe, unavailable_reasons=reasons,
    )
    return PortfolioAnalysis(backtest=backtest, performance=performance)
