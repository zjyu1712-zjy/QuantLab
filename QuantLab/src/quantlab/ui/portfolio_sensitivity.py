"""参数对比的独立按钮与会话状态，不触碰单次结果、基准和实验档案。"""

from dataclasses import asdict

import pandas as pd
import streamlit as st

from quantlab.portfolio.sensitivity import (
    INTERVAL_CANDIDATES, TOP_N_CANDIDATES, run_parameter_sensitivity,
)
from quantlab.portfolio.types import PortfolioBacktestError
from quantlab.ui.portfolio_display import format_metric, portfolio_error_message
from quantlab.ui.portfolio_heatmap import METRIC_LABELS, build_parameter_heatmap


_METRICS = (("cumulative_return", "累计收益率"), ("annualized_return", "年化收益率"),
            ("annualized_volatility", "年化波动率"), ("max_drawdown", "最大回撤"), ("sharpe_ratio", "夏普比率"))


def clear_portfolio_sensitivity(*, reset_metric=True):
    for key in ("portfolio_sensitivity_result", "portfolio_sensitivity_identity"):
        st.session_state.pop(key, None)
    if reset_metric:
        st.session_state.pop("portfolio_sensitivity_metric", None)


def sensitivity_display_table(result):
    """将安全原因码翻译为说明；缺失值不替换成 0。"""
    rows = []
    for row in result.itertuples(index=False):
        failed = row.status == "failed"
        if failed:
            explanation = portfolio_error_message(PortfolioBacktestError("参数组未完成", reason=row.failure_reason))
        else:
            explanation = "；".join(f"{label}：{row.unavailable_reasons[field]}"
                                    for field, label in _METRICS if field in row.unavailable_reasons)
            if row.trade_count == 0:
                explanation = "未发生成交；可能历史不足、有效因子不足或资金不足。" + explanation
        rows.append({
            "选择数量 N": row.top_n, "调仓间隔（期）": row.rebalance_interval,
            "状态": "失败" if failed else ("完成（零成交）" if row.trade_count == 0 else "完成"),
            **{label: format_metric(getattr(row, field), percent=field != "sharpe_ratio") for field, label in _METRICS},
            "成交笔数": "—" if pd.isna(row.trade_count) else str(row.trade_count), "说明": explanation or "—",
        })
    return pd.DataFrame(rows)


def render_portfolio_sensitivity(preview, definition, identity, *, config, analytics_config):
    """只在点击时运行；N/K 是遍历维度，不进入固定输入身份。"""
    active_identity = (identity, config.ascending, asdict(config.backtest), asdict(analytics_config),
                       TOP_N_CANDIDATES, INTERVAL_CANDIDATES)
    if st.session_state.get("portfolio_sensitivity_identity") != active_identity:
        had_result = "portfolio_sensitivity_result" in st.session_state
        clear_portfolio_sensitivity()
        if had_result:
            st.info("参数对比的固定输入已变化，旧对比已清除；请重新点击“运行参数对比”。")
    st.subheader("小型参数敏感性对比")
    candidates = [n for n in TOP_N_CANDIDATES if n <= len(preview.sources)]
    st.caption(
        f"固定当前快照、因子、排名方向、资金、费用与指标口径，仅比较 N = {' / '.join(map(str, candidates))}，"
        f"调仓间隔 = 5 / 10 / 20 期，共 {len(candidates) * 3} 组。单次回测上方的 N 和调仓间隔不限制此网格。"
    )
    st.caption(
        f"{preview.start} 至 {preview.end} · {len(preview.alignment.calendar.dates)} 个研究日期位置 · "
        f"{definition.description} · {'从低到高' if config.ascending else '从高到低'} · "
        f"初始资金 {config.backtest.initial_cash:,.2f} 元。使用完整区间，含热身与空仓期；期数不是自然日。"
    )
    st.caption("这是同一样本内的敏感性观察，不自动挑选最优参数，不代表未来收益；"
               "不运行基准、不替换单次回测、不批量保存实验。结果仅保留在当前会话。")
    with st.expander("本轮固定成本与指标口径"):
        costs = config.backtest
        st.write(f"佣金率 {costs.commission_rate:.4%}；最低佣金 {costs.minimum_commission:.2f} 元；"
                 f"滑点率 {costs.slippage_rate:.4%}；卖出税率 {costs.sell_tax_rate:.4%}；每手 {costs.lot_size} 份。")
        st.write(f"年化每年 {analytics_config.periods_per_year} 期；无风险年利率 {analytics_config.annual_risk_free_rate:.2%}。"
                 "最大回撤用负数表示下跌。费用与成交假设沿用单次组合回测。")
    if st.button("运行参数对比", key="portfolio_sensitivity_run"):
        # 同输入重跑只替换结果；删除仍在前端显示的控件状态会使选项与图表分裂。
        clear_portfolio_sensitivity(reset_metric=False)
        try:
            with st.spinner(f"正在逐组运行 {len(candidates) * 3} 个参数组合……"):
                result = run_parameter_sensitivity(
                    preview, factor_name=definition.name, ascending=config.ascending,
                    backtest_config=config.backtest, analytics_config=analytics_config,
                )
            st.session_state["portfolio_sensitivity_result"] = result
            st.session_state["portfolio_sensitivity_identity"] = active_identity
        except Exception as error:
            st.error("参数对比未完成：" + portfolio_error_message(error))
    result = st.session_state.get("portfolio_sensitivity_result")
    if result is None:
        st.caption("点击按钮后才会计算；调整输入不会自动批量运行。")
        return
    failed = int(result.status.eq("failed").sum())
    st.caption(f"本轮 {len(result)} 组：完成 {len(result) - failed} 组，失败 {failed} 组。按 N、调仓间隔排列，不按收益排序。")
    metric = st.selectbox("热力图指标", list(METRIC_LABELS), format_func=METRIC_LABELS.__getitem__,
                          key="portfolio_sensitivity_metric", help="只重绘现有汇总，不重新回测。")
    try:
        chart = build_parameter_heatmap(result, metric=metric)
        st.altair_chart(chart, width="stretch", key="portfolio_sensitivity_heatmap")
    except Exception:
        st.warning("热力图暂时无法展示，原参数汇总与回测结果未改变；请查看下方表格。")
    st.caption("蓝色越深只表示数值越大，不代表策略更好。色阶按本轮所选指标含零的范围缩放，"
               "不同指标或不同轮次不能直接比较颜色。灰格分别标注“失败”或“不可用”；真实零值正常着色。")
    if metric == "max_drawdown":
        st.caption("最大回撤保留负号，越接近 0 表示该区间回撤越小。")
    elif metric == "annualized_volatility":
        st.caption("波动率数值越大表示历史波动越大，深色不表示风险更低。")
    table = sensitivity_display_table(result)
    st.dataframe(table, hide_index=True, width="stretch", key="portfolio_sensitivity_table")
    st.caption("窄屏可在表格内横向滚动。— 表示不可用；成交笔数统计实际买卖成交记录，不是完整买卖往返次数。")
    with st.expander("逐组状态与不可用原因"):
        for row in table.to_dict("records"):
            st.write(f"N={row['选择数量 N']} / {row['调仓间隔（期）']}期 · {row['状态']} · {row['说明']}")
