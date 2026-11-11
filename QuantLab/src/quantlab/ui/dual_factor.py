"""双因子预览的独立控件与展示；不修改单因子或组合会话结果。"""

import pandas as pd
import streamlit as st

from quantlab import factors
from quantlab.data.research_sets import ResearchSetPreview
from quantlab.ui.factors import STATUS_LABELS


def clear_dual_factor_preview() -> None:
    """研究集合失效时，同时清除双因子旧结果和日期控件。"""
    for key in list(st.session_state):
        if key.startswith("factor_pair_"):
            del st.session_state[key]


def dual_factor_table(result: factors.DualFactorPreview) -> pd.DataFrame:
    """保留全部 ETF 与各因子原因；只格式化副本，不按显示分数重新排序。"""
    rows = result.data
    labels = {**STATUS_LABELS, "warmup": "历史不足（前 20 个日期位置）"}
    reasons = []
    for row in rows.itertuples(index=False):
        if row.is_common:
            reasons.append("共同有效，参与评分" if result.common_count >= 2 else "共同有效，但数量不足两只，暂不评分")
        else:
            missing = []
            if row.momentum_status != "valid":
                missing.append("动量：" + labels[row.momentum_status])
            if row.volatility_status != "valid":
                missing.append("波动率：" + labels[row.volatility_status])
            reasons.append("未进入共同样本；" + "；".join(missing))
    return pd.DataFrame({
        "综合名次": rows["rank"].map(lambda value: "—" if pd.isna(value) else str(int(value))),
        "ETF 代码": rows.symbol,
        "20 期动量": rows.momentum_value.map(lambda value: "—" if pd.isna(value) else f"{value:.2%}"),
        "动量状态": rows.momentum_status.map(labels),
        "20 期年化波动率": rows.volatility_value.map(lambda value: "—" if pd.isna(value) else f"{value:.2%}"),
        "波动率状态": rows.volatility_status.map(labels),
        "动量评分": rows.momentum_score.map(lambda value: "—" if pd.isna(value) else f"{value:.2f}"),
        "低波动评分": rows.volatility_score.map(lambda value: "—" if pd.isna(value) else f"{value:.2f}"),
        "综合评分": rows.combined_score.map(lambda value: "—" if pd.isna(value) else f"{value:.2f}"),
        "共同样本 / 原因": reasons,
    }).reset_index(drop=True)


def render_dual_factor_preview(preview: ResearchSetPreview, identity: tuple) -> None:
    """日期与计算按钮独立；集合/日期变化或计算失败后不展示旧成功结果。"""
    if st.session_state.get("factor_pair_scope") != identity:
        clear_dual_factor_preview()
        st.session_state["factor_pair_scope"] = identity
    st.subheader("双因子等权评分预览")
    st.caption(
        "固定使用 20 期动量（从高到低）与 20 期年化波动率（从低到高），"
        "独立于上方单因子选择及其排名方向。两项均需 21 条完整价格，前 20 条为热身期。"
    )
    st.caption(
        "先选两项因子都有效的共同 ETF 样本，再分别计算 0–100 相对评分；"
        "综合评分 = 动量评分 × 50% + 低波动评分 × 50%。这是因子权重，不是 ETF 仓位。"
    )
    selected_date = st.selectbox(
        "双因子查看日期", preview.alignment.calendar.dates,
        index=len(preview.alignment.calendar.dates) - 1, key="factor_pair_date",
        format_func=lambda value: value.isoformat(),
        help="仅使用这个日期；改变日期会清除旧预览，需要重新计算，不影响单因子日期。",
    )
    current_identity = (identity, selected_date)
    if st.session_state.get("factor_pair_identity") != current_identity:
        st.session_state.pop("factor_pair_result", None)
        st.session_state.pop("factor_pair_identity", None)
    if st.button("计算双因子评分", key="factor_pair_calculate"):
        st.session_state.pop("factor_pair_result", None)
        st.session_state.pop("factor_pair_identity", None)
        try:
            with st.spinner("正在校验快照并计算共同样本评分……"):
                result = factors.run_dual_factor_preview(preview, selected_date)
            st.session_state["factor_pair_result"] = result
            st.session_state["factor_pair_identity"] = current_identity
        except Exception:
            st.error("双因子计算未完成，旧预览已清除。请核对快照和日期后重试；重复失败时请联系开发者排查。")
    result = st.session_state.get("factor_pair_result")
    if result is None:
        st.caption("选择日期后点击计算；日期或研究集合变化后需重新计算，结果仅保留在当前会话。")
        return
    st.caption(f"{result.selected_date.isoformat()} · 共同有效 / 集合总数：{result.common_count} / {len(result.data)}（只 ETF）")
    st.dataframe(dual_factor_table(result), hide_index=True, width="stretch")
    if result.common_count < 2:
        st.info("共同有效 ETF 不足两只，所有评分及综合名次留空；原始因子仍可单独有效，不能用剩余一项代替两项。")
    st.caption(
        "同值按平均并列名次评分，某一项全部相同时该项均为 50 分；综合同分保留并列名次（如 1、1、3），"
        "仅按 ETF 代码排列。显示保留两位小数，不参与重新排序。"
    )
    st.info("这是固定双因子的研究示例，未验证预测能力；评分不是收益率或上涨概率，不生成 Top N、调仓或交易，不改写档案。")
    with st.expander("双因子口径与研究限制"):
        st.write("共同样本可能少于单因子的有效样本，因此分项分数可能与单因子页不同；不是直接平均两个旧分数。不同日期或研究池的同分不能直接等同。")
        st.code(result.input_preview_sha256, language=None)
        for warning in result.warnings:
            st.caption(warning)
