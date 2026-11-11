"""年度片段表的只读适配；原全区间绩效、存档及实验包均不变。"""

from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from quantlab.portfolio.types import PortfolioAnalysis, PortfolioBacktestError
from quantlab.portfolio.yearly import yearly_equity_statistics
from quantlab.ui.portfolio_display import _checked_comparison


_LABELS = {
    "year": "年份", "base_date": "收益起算日", "first_date": "年内首条观测",
    "last_date": "年内末条观测", "observation_count": "年内观测数", "return_count": "收益区间数",
    "period_return": "策略区间收益", "max_drawdown": "年内最大回撤", "status": "覆盖说明",
}
_STATUSES = {
    "observed_segment": "仅为已观测片段，未认证全年覆盖",
    "no_return_interval": "仅一个起始资产点，没有收益区间",
    "no_observations": "该年没有观测，不能视为零收益",
    "gap_spanning_years": "跨整年缺口：含缺失年份区间，不是单年收益",
}


def _annual_analysis(analysis):
    """保留原结果声明的完整日期轴，不借用来源行情、不补齐数据。"""
    try:
        if not isinstance(analysis, PortfolioAnalysis):
            raise ValueError()
        result = analysis.backtest
        meta, curve = result.metadata, result.equity_curve
        if (meta["schema_version"] not in ("portfolio-run-v1", "portfolio-benchmark-v1")
                or meta["status"] != "complete"):
            raise ValueError()
        dates = meta["data"]["calendar"]["dates"]
        if (not isinstance(dates, list) or not dates
                or any(type(day) is not str or date.fromisoformat(day).isoformat() != day for day in dates)):
            raise ValueError()
        axis = pd.DatetimeIndex(dates, dtype="datetime64[ns]")
        if (not pd.DatetimeIndex(curve.date).equals(axis)
                or curve.equity.iloc[0] != meta["strategy"]["config"]["backtest"]["initial_cash"]):
            raise ValueError()
        return yearly_equity_statistics(curve)
    except PortfolioBacktestError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError) as error:
        raise PortfolioBacktestError("年度统计输入不完整", reason="invalid_portfolio_result") from error


def portfolio_yearly_table(analysis: PortfolioAnalysis, benchmark: PortfolioAnalysis | None = None) -> pd.DataFrame:
    """派生年度片段及同口径基准差；百分比列仍为比例，差值列使用百分点。"""
    table = _annual_analysis(analysis)
    table["基准区间收益"] = np.nan
    table["收益差（百分点）"] = np.nan
    if benchmark is not None:
        _checked_comparison(analysis, benchmark)
        baseline = _annual_analysis(benchmark)
        table["基准区间收益"] = baseline.period_return.to_numpy(copy=True)
        with np.errstate(over="ignore", invalid="ignore"):
            difference = (table.period_return - baseline.period_return) * 100
        if not np.isfinite(difference.loc[table.period_return.notna()]).all():
            raise PortfolioBacktestError("年度收益差超出可靠数值范围", reason="numeric_range")
        table["收益差（百分点）"] = difference
    table["status"] = table.status.map(_STATUSES)
    return table.rename(columns=_LABELS)


def render_portfolio_yearly(analysis, *, benchmark=None, benchmark_status="unavailable",
                            historical=False, namespace="portfolio_live"):
    """每次只用显式传入的一对结果，无独立会话缓存；异常不隐藏原回测。"""
    st.subheader("分年份表现检查")
    if historical:
        st.caption("年度统计由当前版本依据已保存资金曲线临时计算，不重跑策略，不改变原保存指标、档案或 ZIP。")
    else:
        st.caption("年度统计基于本次完整资金曲线临时计算，不重跑策略，不改变原全区间指标。")
    st.caption(
        "按收益区间结束年份归属：首年从研究起点起算，后续年份连接此前最后一条资产记录；"
        "保留热身与现金期，不年化，也不把各年收益率直接相加。年份标签不代表完整全年。"
    )
    try:
        table = portfolio_yearly_table(analysis)
    except (ValueError, TypeError, OverflowError):
        st.warning("年度统计暂不可用：资金曲线不完整或数值超出可靠范围。原回测结果仍保留。")
        return
    if benchmark is not None:
        try:
            table = portfolio_yearly_table(analysis, benchmark)
        except (ValueError, TypeError, OverflowError):
            st.warning("基准年度统计不可用：日期轴、口径或数值校验未通过。仅保留策略年度表，基准及收益差留空。")
    else:
        state = "未运行" if benchmark_status == "not_run" else "不可用"
        st.info(f"基准{state}；年度表仅展示策略，基准收益和收益差留空，不视为零。")
    if table["覆盖说明"].str.contains("跨整年").any():
        st.warning("资金曲线存在跨整年缺口；对应行含跨年观测跨度，不能解释为该单一自然年的收益。请展开覆盖日期核对。")
    st.dataframe(
        table[["年份", "策略区间收益", "年内最大回撤", "基准区间收益", "收益差（百分点）", "覆盖说明"]],
        hide_index=True, width="stretch", key=f"{namespace}_yearly",
        column_config={
            "年份": st.column_config.NumberColumn(format="%d"),
            **{name: st.column_config.NumberColumn(format="percent")
               for name in ("策略区间收益", "年内最大回撤", "基准区间收益")},
            "收益差（百分点）": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        "年内最大回撤包含收益起算资产点，并在每个年度片段重新计算峰值，不等于全程回撤的年度切片。"
        "收益差＝策略区间收益−基准区间收益，单位为百分点；不是风险调整 alpha。空值表示不可计算，不是零。"
    )
    with st.expander("年度覆盖日期与计算口径"):
        st.dataframe(
            table[["年份", "收益起算日", "年内首条观测", "年内末条观测", "年内观测数", "收益区间数", "覆盖说明"]],
            hide_index=True, width="stretch", key=f"{namespace}_yearly_coverage",
            column_config={name: st.column_config.DateColumn(format="YYYY-MM-DD")
                           for name in ("收益起算日", "年内首条观测", "年内末条观测")},
        )
        st.caption("完整性以原研究日期轴为边界，未核验权威交易日历；缺失年份不补值。此检查描述已有样本，不是样本外验证或盈利证明。")
