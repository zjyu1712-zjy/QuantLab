"""组合研究的纯展示适配：不运行回测、不修改结果、不保存页面状态。"""

from __future__ import annotations

from datetime import date
from math import isfinite
import re

import altair as alt
import numpy as np
import pandas as pd

from quantlab.portfolio.types import PortfolioAnalysis, PortfolioBacktestError


_COLUMN_LABELS = {
    "date": "日期", "decision_id": "调仓编号", "order_id": "订单编号", "trade_id": "成交编号",
    "signal_date": "信号日期", "execution_date": "执行日期", "symbol": "ETF 代码",
    "valid_count": "有效因子数量", "top_n": "选择数量 N", "status": "状态", "reason": "原因",
    "cutoff_tie": "选取边界同值", "factor_value": "因子值", "target_weight": "目标权重（比例）",
    "side": "方向", "requested_quantity": "申请份额", "filled_quantity": "成交份额",
    "unfilled_quantity": "未成交份额", "quantity": "份额", "open_price": "开盘价（元）",
    "execution_price": "成交价（元）", "notional": "成交额（元）", "commission": "佣金（元）",
    "tax": "税费（元）", "slippage_cost": "滑点成本（元）", "cash_after": "成交后现金（元）",
    "position_after": "成交后持仓份额", "close": "收盘价（元）", "position_value": "持仓市值（元）",
    "weight": "持仓权重（比例）", "cash": "现金（元）", "equity": "总资产（元）",
    "net_value": "净值", "cumulative_return": "累计收益率（比例）",
    "cumulative_fees_taxes": "累计佣金及税费（元）", "cumulative_slippage": "累计滑点（元）",
    "drawdown": "回撤（比例）", "is_valid": "因子有效", "has_observation": "有行情观测",
    "open": "开盘价（元）", "high": "最高价（元）", "low": "最低价（元）", "volume": "成交量（份）",
}
_STATUS_LABELS = {
    "EXECUTED": "已执行", "EXPIRED": "已过期", "PENDING": "待执行",
    "SKIPPED_INSUFFICIENT_FACTORS": "有效因子不足，跳过调仓",
    "FILLED": "全部成交", "PARTIALLY_FILLED": "部分成交", "REJECTED": "未成交",
    "valid": "有效", "warmup": "热身期", "missing_window": "窗口数据缺失", "numeric_range": "数值超出范围",
}
_ERROR_MESSAGES = {
    "invalid_config": "组合参数无效，请检查选择数量、调仓间隔和交易成本设置。",
    "invalid_analytics_config": "绩效指标参数无效，请检查每年期数和无风险利率。",
    "invalid_research_input": "研究集输入未通过校验，请重新确认数据版本、日期范围和复权声明。",
    "invalid_factor": "所选因子无效或不适用于组合研究，请重新选择。",
    "invalid_factor_result": "因子结果未通过校验，无法生成可靠的调仓安排。",
    "invalid_run_identity": "组合运行身份未通过校验；若刚更新代码，请重启 QuantLab 后重试。",
    "invalid_portfolio_result": "组合结果不完整或未通过校验，无法展示绩效分析。",
    "invalid_comparison": "组合与基准的日期轴、数据版本或成本口径不一致，无法进行可靠比较；请重新运行。",
    "invalid_market_data": "行情数据未通过校验，请检查研究集的数据覆盖。",
    "invalid_schedule": "调仓安排未通过校验，本次组合研究已停止。",
    "invalid_execution_input": "模拟交易输入无效，本次组合研究已停止。",
    "missing_open": "调仓所需开盘价缺失，本次组合研究已停止。",
    "missing_close": "持仓所需收盘价缺失，无法完成估值，本次组合研究已停止。",
    "zero_volume": "拟成交资产当日成交量为零，不支持本次成交假设，组合研究已停止。",
    "negative_cash": "模拟交易出现负现金余额，不能继续无融资组合研究。",
    "negative_sell_proceeds": "卖出费用超过成交额，净收入为负，本次组合研究已停止。",
    "non_positive_equity": "组合总资产已耗尽，无法继续完整绩效研究。",
    "numeric_range": "计算超出可靠数值范围，请检查资金、价格和指标参数。",
}
_UNKNOWN_ERROR = "组合研究未能完成，请检查研究集与参数后重试；本次未生成可展示的结果。"


def format_metric(value: float | None, *, percent: bool = True) -> str:
    """未定义/非有限指标明确显示为破折号，保留真实零值。"""
    if value is None or not isfinite(value):
        return "—"
    return format(value, ".2%" if percent else ".2f")


def portfolio_curve_table(analysis: PortfolioAnalysis) -> pd.DataFrame:
    """保留结果的完整日期轴，用日期对齐已有回撤，不计算或补造收益。"""
    frame = analysis.backtest.equity_curve.loc[:, ["date", "equity"]].copy(deep=True).reset_index(drop=True)
    frame["drawdown"] = analysis.performance.drawdown.reindex(pd.DatetimeIndex(frame.date)).to_numpy(copy=True)
    return frame


def build_portfolio_chart(frame: pd.DataFrame, *, measure: str) -> alt.Chart:
    """完整序列折线；总资产用元，负向回撤以百分比显示，缺失值断线。"""
    if measure not in {"equity", "drawdown"}:
        raise ValueError("组合曲线仅支持总资产或回撤")
    title, number_format = (("组合总资产（元）", ",.2f") if measure == "equity" else ("回撤（%）", ".2%"))
    # 显式记录绕过 DataFrame 的 5000 行默认阈值，不截断、不抽样、不改变全局设置。
    rows = [{"date": pd.Timestamp(day).isoformat(), measure: float(value) if pd.notna(value) and isfinite(value) else None}
            for day, value in frame.loc[:, ["date", measure]].itertuples(index=False, name=None)]
    single_point = sum(row[measure] is not None for row in rows) == 1
    # 三个真实日期刻度给窄屏留出空间，避免中间标签挤掉最新日期。
    tick_indices = sorted({0, (len(rows) - 1) // 2, len(rows) - 1}) if rows else []
    tick_dates = [pd.Timestamp(frame.date.iloc[index]) for index in tick_indices]
    ticks = [alt.DateTime(year=day.year, month=day.month, date=day.day) for day in tick_dates]
    return alt.Chart(alt.Data(values=rows)).mark_line(
        point=single_point, invalid="break-paths-filter-domains",
    ).encode(
        x=alt.X("date:T", title="日期", axis=alt.Axis(
            format="%Y-%m-%d", labelAngle=0, labelOverlap="greedy", values=ticks,
        )),
        y=alt.Y(f"{measure}:Q", title=title, axis=alt.Axis(format=number_format),
                scale=alt.Scale(zero=measure == "drawdown")),
        tooltip=[alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"),
                 alt.Tooltip(f"{measure}:Q", title=title, format=number_format)],
    ).properties(width="container", height=280)


def portfolio_detail_table(frame: pd.DataFrame) -> pd.DataFrame:
    """复制并转译规范明细，数值/日期/空值保持原样；未知枚举不猜测。"""
    result = frame.copy(deep=True)
    for column, labels in (("status", _STATUS_LABELS), ("side", {"BUY": "买入", "SELL": "卖出"}),
                           ("reason", _ERROR_MESSAGES)):
        if column in result:
            result[column] = result[column].replace(labels)
    return result.rename(columns=_COLUMN_LABELS)


def portfolio_error_message(error: Exception) -> str:
    """只展示白名单原因以及严格验证的日期/代码，绝不使用异常原文。"""
    if not isinstance(error, PortfolioBacktestError):
        return _UNKNOWN_ERROR
    reason = error.reason
    if not isinstance(reason, str) or reason not in _ERROR_MESSAGES:
        return _UNKNOWN_ERROR
    context = []
    if type(error.date) is date:
        context.append(error.date.isoformat())
    if isinstance(error.symbols, (tuple, list)):
        symbols = [symbol for symbol in error.symbols if isinstance(symbol, str) and re.fullmatch(r"[0-9]{6}", symbol)]
        if symbols:
            context.append("ETF " + "、".join(symbols))
    return _ERROR_MESSAGES[reason] + ("（" + "；".join(context) + "）" if context else "")


def safe_portfolio_error(error: Exception) -> PortfolioBacktestError:
    """只保留可存档的失败上下文，不保留底层异常原文或回溯。"""
    reason, day, symbols = "unavailable", None, ()
    if isinstance(error, PortfolioBacktestError):
        if isinstance(error.reason, str) and error.reason in _ERROR_MESSAGES:
            reason = error.reason
        if type(error.date) is date:
            day = error.date
        if isinstance(error.symbols, (tuple, list)):
            symbols = tuple(symbol for symbol in error.symbols
                            if isinstance(symbol, str) and re.fullmatch(r"[0-9]{6}", symbol))
    return PortfolioBacktestError("基准未完成", reason=reason, date=day, symbols=symbols)


def _checked_comparison(strategy: PortfolioAnalysis, benchmark: PortfolioAnalysis) -> pd.DatetimeIndex:
    """比较必须基于同一完整研究轴和成本口径；不对齐、修补或认证回测。"""
    def fail():
        raise PortfolioBacktestError("组合与基准的完整日期轴或研究口径不一致", reason="invalid_comparison")

    if not isinstance(strategy, PortfolioAnalysis) or not isinstance(benchmark, PortfolioAnalysis):
        fail()
    try:
        left, right = strategy.backtest.metadata, benchmark.backtest.metadata
        if (left["schema_version"] != "portfolio-run-v1"
                or right["schema_version"] != "portfolio-benchmark-v1"
                or left["status"] != "complete" or right["status"] != "complete"):
            fail()
        for key in ("preview_sha256", "market_sha256", "calendar"):
            if not left["data"][key] or left["data"][key] != right["data"][key]:
                fail()
        if (left["strategy"]["config"]["backtest"] != right["strategy"]["config"]["backtest"]
                or left["analytics_config"] != right["analytics_config"]
                or left["strategy"]["signal_anchor_position"] != right["strategy"]["signal_anchor_position"]):
            fail()
        axis_values = left["data"]["calendar"]["dates"]
        if (not isinstance(axis_values, list) or not axis_values
                or any(not isinstance(day, str) or date.fromisoformat(day).isoformat() != day
                       for day in axis_values)):
            fail()
        axis = pd.DatetimeIndex(axis_values, dtype="datetime64[ns]", name="date")
        if axis.hasnans or not axis.is_unique or not axis.is_monotonic_increasing:
            fail()
        for analysis in (strategy, benchmark):
            curve, net = analysis.backtest.equity_curve, analysis.performance.net_value
            if (not isinstance(curve, pd.DataFrame) or not isinstance(net, pd.Series)
                    or not isinstance(net.index, pd.DatetimeIndex)
                    or len(curve) != len(axis) or len(net) != len(axis)
                    or curve.date.dtype != np.dtype("datetime64[ns]")
                    or not pd.DatetimeIndex(curve.date).equals(axis) or not net.index.equals(axis)):
                fail()
            values = net.to_numpy(dtype="float64", copy=True)
            if not np.isfinite(values).all() or (values <= 0).any() or values[0] != 1.:
                fail()
        if not strategy.performance.net_value.index.equals(benchmark.performance.net_value.index):
            fail()
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as error:
        if isinstance(error, PortfolioBacktestError):
            raise
        raise PortfolioBacktestError("比较输入缺少有效的完整研究口径", reason="invalid_comparison") from error
    return axis


def portfolio_comparison_table(strategy: PortfolioAnalysis, benchmark: PortfolioAnalysis) -> pd.DataFrame:
    """直接复制已分析净值；保留每个原始日期，不合并、不补值。"""
    axis = _checked_comparison(strategy, benchmark)
    return pd.DataFrame({
        "date": axis.copy(),
        "strategy_net_value": strategy.performance.net_value.to_numpy(dtype="float64", copy=True),
        "benchmark_net_value": benchmark.performance.net_value.to_numpy(dtype="float64", copy=True),
    })


def build_portfolio_comparison_chart(table: pd.DataFrame) -> alt.Chart:
    """完整双线净值图；颜色和虚线同时区分基准，单日期显示点。"""
    labels = ["组合策略", "全池等权买入持有"]
    rows = [{"date": pd.Timestamp(day).isoformat(), "series": label, "net_value": float(value)}
            for label, column in zip(labels, ("strategy_net_value", "benchmark_net_value"))
            for day, value in table.loc[:, ["date", column]].itertuples(index=False, name=None)]
    indices = sorted({0, (len(table) - 1) // 2, len(table) - 1}) if len(table) else []
    ticks = [alt.DateTime(year=day.year, month=day.month, date=day.day)
             for day in (pd.Timestamp(table.date.iloc[index]) for index in indices)]
    return alt.Chart(alt.Data(values=rows)).mark_line(point=len(table) == 1).encode(
        x=alt.X("date:T", title="日期", axis=alt.Axis(
            format="%Y-%m-%d", labelAngle=0, labelOverlap="greedy", values=ticks)),
        y=alt.Y("net_value:Q", title="净值（起点=1）", axis=alt.Axis(format=".2f"), scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", title=None,
                        scale=alt.Scale(domain=labels, range=["#2563eb", "#64748b"]),
                        legend=alt.Legend(orient="top", direction="horizontal", labelLimit=220)),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=labels, range=[[1, 0], [6, 4]]), legend=None),
        tooltip=[alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"),
                 alt.Tooltip("series:N", title="研究对象"), alt.Tooltip("net_value:Q", title="净值", format=".4f")],
    ).properties(width="container", height=280)


def portfolio_comparison_metrics(strategy: PortfolioAnalysis, benchmark: PortfolioAnalysis) -> pd.DataFrame:
    """并列呈现既有指标；未定义值保持破折号，不重新计算。"""
    _checked_comparison(strategy, benchmark)
    rows = [
        ("期末总资产（元）", None, False), ("累计收益率", "cumulative_return", True),
        ("年化收益率", "annualized_return", True), ("年化波动率", "annualized_volatility", True),
        ("最大回撤", "max_drawdown", True), ("夏普比率", "sharpe_ratio", False),
    ]
    return pd.DataFrame([
        [label] + [format_metric(analysis.backtest.equity_curve.equity.iloc[-1] if field is None
                                else getattr(analysis.performance, field), percent=percent)
                   for analysis in (strategy, benchmark)]
        for label, field, percent in rows
    ], columns=["指标", "组合策略", "全池等权买入持有"])


def cumulative_return_difference_pp(strategy: PortfolioAnalysis, benchmark: PortfolioAnalysis) -> float | None:
    """累计收益率差（百分点），不是风险调整 alpha。"""
    _checked_comparison(strategy, benchmark)
    left, right = strategy.performance.cumulative_return, benchmark.performance.cumulative_return
    if left is None or right is None or not isfinite(left) or not isfinite(right):
        return None
    difference = (left - right) * 100.
    if not isfinite(difference):
        raise PortfolioBacktestError("累计收益率差超出可靠范围", reason="invalid_comparison")
    return difference
