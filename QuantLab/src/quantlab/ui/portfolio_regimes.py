"""当前/历史各自策略和基准的只读涨跌分组，无会话缓存和存档写入。"""

from math import isfinite
from numbers import Real

import pandas as pd
import streamlit as st

from quantlab.portfolio.regimes import benchmark_regime_statistics
from quantlab.portfolio.types import PortfolioAnalysis, PortfolioBacktestError
from quantlab.ui.portfolio_display import _checked_comparison


_GROUPS = {"up": "基准上涨区间", "down": "基准下跌区间", "flat": "基准零变动区间",
           "cash": "基准现金区间（不参与涨跌分组）", "entry": "基准首次建仓区间（不参与涨跌分组）"}
_COLUMNS = {"group": "基准区间分类", "interval_count": "区间数量",
            "strategy_mean_return": "策略平均区间收益", "benchmark_mean_return": "基准平均区间收益",
            "mean_difference_pp": "平均收益差（百分点）"}


def portfolio_regime_table(strategy: PortfolioAnalysis, benchmark: PortfolioAnalysis) -> pd.DataFrame:
    """校验同一研究身份和完整日期轴；不做宽松日期对齐或借用其他基准。"""
    _checked_comparison(strategy, benchmark)
    try:
        baseline_rule = benchmark.backtest.metadata["strategy"]
        if (baseline_rule["name"] != "equal_weight_buy_and_hold"
                or baseline_rule["execution_rule"] != "L+1_open_once"):
            raise ValueError()
        for item in (strategy, benchmark):
            cash = item.backtest.metadata["strategy"]["config"]["backtest"]["initial_cash"]
            if (isinstance(cash, bool) or not isinstance(cash, Real) or not isfinite(cash)
                    or cash <= 0 or item.backtest.equity_curve.equity.iloc[0] != cash):
                raise ValueError()
        table = benchmark_regime_statistics(strategy.backtest.equity_curve, benchmark.backtest.equity_curve)
    except PortfolioBacktestError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError) as error:
        raise PortfolioBacktestError("涨跌分组输入不完整或不是一次买入持有基准", reason="invalid_comparison") from error
    table["group"] = table.group.map(_GROUPS)
    return table.rename(columns=_COLUMNS)


def render_portfolio_regimes(analysis, *, benchmark=None, benchmark_status="unavailable",
                             historical=False, namespace="portfolio_live"):
    st.subheader("按研究池基准涨跌检查表现")
    st.caption("涨跌分组由当前版本依据本实验已保存的策略与基准曲线临时计算，不重跑，不改变原指标、档案或 ZIP。"
               if historical else "涨跌分组基于本次完整策略与基准曲线临时计算，不重跑，不改变原全区间指标。")
    st.caption("这是按同期基准已发生涨跌做的事后分组，不是牛熊市识别、预测或交易信号。"
               "研究池基准不是市场指数：仅初始目标等权，之后权重漂移，并包含交易成本和余留现金。")
    if benchmark is None or benchmark_status != "complete":
        state = "未运行" if benchmark_status == "not_run" else "不可用"
        st.info(f"涨跌分组不可用：基准{state}，不能用策略自身走势或其他实验基准替代。原策略结果仍保留。")
        return
    try:
        table = portfolio_regime_table(analysis, benchmark)
    except (ValueError, TypeError, OverflowError):
        st.warning("涨跌分组暂不可用：策略/基准口径、完整日期轴、持仓或数值校验未通过。原回测结果仍保留。")
        return
    total = int(table["区间数量"].sum())
    grouped = int(table["区间数量"].iloc[:3].sum())
    st.caption(f"完整曲线共 {total} 个观测区间：{grouped} 个参与涨跌分组，"
               f"{int(table['区间数量'].iloc[3])} 个基准现金区间，{int(table['区间数量'].iloc[4])} 个首次建仓区间。"
               "不补日期，观测区间可能跨多个自然日，不是逐笔交易。")
    st.caption("仅当基准在区间两端均有持仓时，按其区间收益的正、负、零分类。"
               "现金及首次建仓只列数量，不参与组内均值；持仓零变动也不等同于市场震荡。")
    if total == 0:
        st.info("涨跌分组不可用：仅一个资产观测点，没有收益区间。")
    elif grouped == 0:
        st.info("涨跌分组不可用：没有可分组的持仓区间，现金或首次建仓不能用于判断环境表现。")
    st.dataframe(table, hide_index=True, width="stretch", key=f"{namespace}_regimes",
        column_config={
            "区间数量": st.column_config.NumberColumn(format="%d"),
            "策略平均区间收益": st.column_config.NumberColumn(format="percent"),
            "基准平均区间收益": st.column_config.NumberColumn(format="percent"),
            "平均收益差（百分点）": st.column_config.NumberColumn(format="%.4f"),
        })
    st.caption("收益按相邻总资产之比减一计算；均值是组内各观测区间收益的算术平均，"
               "平均收益差＝策略与基准区间收益之差的均值，单位为百分点，不是 alpha。"
               "空组及未参与分组行的收益留空，不代表零；窄屏可在表格内横向滚动。")
    st.caption("同组区间可能不连续，不能把均值当累计或年化收益；不拼接资金曲线，"
               "不计算组内回撤、夏普或交易胜率。请结合样本数量阅读，这不是样本外验证或未来稳定性证明。")
