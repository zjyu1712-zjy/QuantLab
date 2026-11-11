"""当前/历史共用的只读集中度展示，无会话缓存、无档案写入。"""

from datetime import date
from math import isfinite
from numbers import Real

import pandas as pd
import streamlit as st

from quantlab.portfolio.concentration import GainConcentration, equity_gain_concentration
from quantlab.portfolio.types import PortfolioAnalysis, PortfolioBacktestError
from quantlab.ui.portfolio_display import format_metric


def portfolio_gain_concentration(analysis: PortfolioAnalysis) -> GainConcentration:
    """只接受声明完整的策略曲线，不把当前研究数据混入历史实验。"""
    try:
        if not isinstance(analysis, PortfolioAnalysis):
            raise ValueError()
        result = analysis.backtest
        meta, curve = result.metadata, result.equity_curve
        if meta["schema_version"] != "portfolio-run-v1" or meta["status"] != "complete":
            raise ValueError()
        dates = meta["data"]["calendar"]["dates"]
        cash = meta["strategy"]["config"]["backtest"]["initial_cash"]
        if (not isinstance(dates, list) or not dates
                or any(type(day) is not str or date.fromisoformat(day).isoformat() != day for day in dates)
                or isinstance(cash, bool) or not isinstance(cash, Real) or not isfinite(cash) or cash <= 0):
            raise ValueError()
        if (not pd.DatetimeIndex(curve.date).equals(pd.DatetimeIndex(dates, dtype="datetime64[ns]"))
                or curve.equity.iloc[0] != cash):
            raise ValueError()
        return equity_gain_concentration(curve)
    except PortfolioBacktestError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError) as error:
        raise PortfolioBacktestError("集中度输入不完整", reason="invalid_portfolio_result") from error


def render_portfolio_concentration(analysis, *, historical=False, namespace="portfolio_live"):
    st.subheader("收益集中度检查")
    if historical:
        st.caption("集中度由当前版本依据本实验已保存资金曲线临时计算，不重跑策略，不改变原指标、档案或 ZIP。")
    else:
        st.caption("集中度基于本次完整资金曲线临时计算，不重跑策略，不改变原全区间指标。")
    st.caption(
        "口径：相邻观测总资产之差取正值，最高至多 5 个正增量之和 ÷ 全部正增量之和。"
        "分母不是最终净利润；包含持仓估值变化及曲线已计入的成本，不是已实现交易盈利。"
    )
    try:
        result = portfolio_gain_concentration(analysis)
    except (ValueError, TypeError, OverflowError):
        st.warning("集中度暂不可用：资金曲线不完整或数值超出可靠范围。原回测结果仍保留。")
        return
    st.caption(f"完整曲线共 {result.interval_count} 个观测区间，其中 {result.positive_count} 个正向增量区间；"
               "保留热身与现金期，不补日期。观测区间可能跨越多个自然日，并非逐笔交易。")
    if result.status != "available":
        reason = ("仅一个资产观测点，没有收益区间。" if result.status == "no_return_interval"
                  else "没有正向资产增量，分母为零，不能计算占比。")
        st.info("集中度不可用：" + reason)
        return
    st.write(f"前 {result.selected_count} 个盈利观测区间占比：**{format_metric(result.top5_share)}**")
    st.caption(f"所选正增量合计 {result.top_gain:,.2f} 元 / 全部正增量合计 {result.gross_positive_gain:,.2f} 元。")
    if result.positive_count < 5:
        st.info(f"正向增量区间不足 5 个，实际纳入 {result.selected_count} 个；此时占比为 100%，不能据此判定风险高。")
    table = result.intervals.rename(columns={"start_date": "区间起始日", "end_date": "区间结束日",
                                            "asset_gain": "资产增量（元）", "positive_gain_share": "占全部正向增量比例"})
    st.dataframe(table, hide_index=True, width="stretch", key=f"{namespace}_concentration_intervals",
        column_config={
            "区间起始日": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "区间结束日": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "资产增量（元）": st.column_config.NumberColumn(format="%.2f"),
            "占全部正向增量比例": st.column_config.NumberColumn(format="percent"),
        })
    st.caption("按资产增量从高到低排列，并列取较早区间；窄屏可在表格内横向滚动。"
               "这是样本内的金额集中程度，不设好坏阈值，不删除这些区间重算收益，也不证明未来稳定性。")
