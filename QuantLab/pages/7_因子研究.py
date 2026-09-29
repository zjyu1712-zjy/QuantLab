from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantlab.data.research_storage import list_research_sets, load_research_set  # noqa: E402
from quantlab.factors import FactorRunResult, create_default_registry, run_research_factor  # noqa: E402
from quantlab.ui.factors import (  # noqa: E402
    build_factor_chart, daily_factor_table, factor_history, history_factor_table,
)
from quantlab.ui.runtime import get_research_storage_dir, is_public_mode, require_access  # noqa: E402


ADJUSTMENTS = {"none": "不复权", "qfq": "前复权", "hfq": "后复权"}
MODES = {"observed_union": "行情日期并集", "supplied": "用户提供日历"}
SOURCE_LABELS = {
    "symbol": "ETF 代码", "version_id": "数据版本", "provider": "来源",
    "imported_at": "导入时间", "adjustment": "价格口径", "adjustment_basis": "口径依据",
    "file_sha256": "文件指纹", "metadata_sha256": "元数据指纹",
    "input_content_sha256": "完整历史内容指纹",
}


def _clear_result():
    for key in ("factor_result", "factor_result_identity", "factor_view_date", "factor_view_symbol"):
        st.session_state.pop(key, None)


def _show_provenance(result: FactorRunResult):
    with st.expander("公式、来源与研究限制"):
        definition = result.definition
        st.caption(f"因子：{definition.name} · 版本：{definition.version} · {definition.description}")
        st.code(definition.formula, language=None)
        st.write(
            f"回看 {definition.lookback_periods} 期，含当前期共 {definition.window_observations} 条观测；"
            "整个窗口必须完整，收盘数据可用后计算。数值越大只表示过去涨幅更高。"
        )
        sources = pd.DataFrame(result.sources).rename(columns=SOURCE_LABELS)
        sources["价格口径"] = sources["价格口径"].map(ADJUSTMENTS)
        sources["口径依据"] = sources["口径依据"].map({"provider": "数据源记录", "user_declared": "用户声明"})
        st.dataframe(sources, hide_index=True, width="stretch")
        st.caption("研究输入指纹")
        st.code(result.input_preview_sha256, language=None)
        st.caption("对齐内容指纹")
        st.code(result.alignment_content_sha256, language=None)
        st.caption("完整日期轴声明（含模式、来源、覆盖范围和日历指纹）")
        st.json(asdict(result.calendar), expanded=False)
        st.caption("完整因子定义")
        st.json(asdict(definition), expanded=False)
        for warning in result.warnings:
            st.warning(warning)


def _show_result(result: FactorRunResult):
    axis = result.calendar
    st.caption(
        f"快照区间：{result.start} 至 {result.end} · {len(result.sources)} 只 ETF · "
        f"{len(axis.dates)} 个日期位置 · {ADJUSTMENTS[result.adjustment]} · {MODES[axis.mode]}"
    )
    if result.adjustment == "none":
        st.caption("当前使用未复权价格，可能受分红、拆并份额影响；价格动量不是总回报。")
    st.subheader("当日结果")
    selected_date = st.selectbox(
        "查看日期", axis.dates, index=len(axis.dates) - 1, key="factor_view_date",
        format_func=lambda value: value.isoformat(),
        help="只显示这个日期，不自动回退到更早的有效值。",
    )
    table = daily_factor_table(result, selected_date)
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("按 ETF 代码排列，不是排名。— 表示不可计算，不是 0%；原因只描述当前输入。")

    st.subheader(f"单只 ETF {result.definition.lookback_periods} 期动量变化")
    symbol = st.selectbox("查看 ETF", [s["symbol"] for s in result.sources], key="factor_view_symbol")
    history = factor_history(result, symbol)
    st.caption("曲线展示该 ETF 的整个快照区间；上方日期只影响当日表。空值保留断点，不补零。")
    if history.is_valid.any():
        st.altair_chart(build_factor_chart(history, result.definition), width="stretch")
    else:
        st.info("这只 ETF 在当前区间没有可用因子值；请在明细中查看历史不足或窗口缺失等原因。")
    with st.expander("查看该 ETF 的逐日数值与原因"):
        st.dataframe(history_factor_table(history, result.definition), hide_index=True, width="stretch")
    _show_provenance(result)


def render_factor_page():
    require_access()
    research_dir = get_research_storage_dir(PROJECT_ROOT)
    scope = str(research_dir)
    if st.session_state.get("factor_storage_scope") != scope:
        for key in list(st.session_state):
            if key.startswith("factor_"):
                del st.session_state[key]
        st.session_state["factor_storage_scope"] = scope

    st.title("因子研究")
    st.caption("比较不同回看周期的历史价格变化；期数按研究日期轴计算，不是自然日。")
    st.info("仅描述历史价格变化，不是策略收益或买卖信号；本页不排名、不做组合回测、不自动下单。")
    if is_public_mode():
        st.warning("当前为临时会话，仅显示本会话的研究集合；结果不会自动保存。")

    try:
        scan = list_research_sets(research_dir)
    except (ValueError, OSError):
        _clear_result()
        st.error("无法读取研究集合目录，请检查本地目录权限后重试。")
        return
    if scan.issues:
        st.warning(f"有 {len(scan.issues)} 个集合未通过校验，已从可选列表排除；请到‘多 ETF 数据’检查。")
    records = {record.set_id: record for record in scan.records}
    if not records:
        _clear_result()
        st.session_state.pop("factor_set_id", None)
        st.info("还没有可用的已保存集合。请先到‘多 ETF 数据’生成并保存研究集合。")
        return
    if st.session_state.get("factor_set_id") not in records:
        _clear_result()
        st.session_state.pop("factor_set_id", None)
    selected = st.selectbox(
        "选择已保存集合", list(records), key="factor_set_id",
        format_func=lambda value: f"{records[value].name} · {records[value].created_at[:19]} UTC · {value[:8]}",
    )
    try:
        # 每次页面重跑均重验快照，不让损坏/被替换的文件沿用之前的成功结果。
        loaded = load_research_set(research_dir, selected)
    except (ValueError, OSError):
        _clear_result()
        st.error("所选集合无法载入或校验未通过。旧结果已清除，请返回‘多 ETF 数据’检查后重试。")
        return
    registry = create_default_registry()
    factor_names = sorted(registry.names(), key=lambda name: registry.get(name).definition.lookback_periods)
    if st.session_state.get("factor_name") not in factor_names:
        st.session_state.pop("factor_name", None)
    factor_name = st.selectbox(
        "选择因子", factor_names, key="factor_name",
        format_func=lambda name: registry.get(name).definition.description,
    )
    definition = registry.get(factor_name).definition
    periods = definition.lookback_periods
    st.caption(f"{periods} 期价格动量 = 当前收盘价 ÷ {periods} 期前收盘价 − 1；需要 {definition.window_observations} 条完整观测。")
    identity = (scope, selected, loaded.preview.preview_sha256, definition.name, definition.version)
    if st.session_state.get("factor_result_identity") != identity:
        _clear_result()
    if st.button(f"计算 {periods} 期动量", key="factor_calculate", type="primary"):
        _clear_result()
        try:
            with st.spinner("正在核验快照并逐只计算……"):
                result = run_research_factor(loaded.preview, factor_name, registry=registry)
            st.session_state["factor_result"] = result
            st.session_state["factor_result_identity"] = identity
        except Exception:
            # 不向页面复制可能含私人路径或凭证的底层异常，也不能保留旧结果。
            _clear_result()
            st.error("因子计算未完成，旧结果已清除。请检查快照和历史区间，重新计算；重复失败时请联系开发者排查。")
    result = st.session_state.get("factor_result")
    if result is None:
        st.caption("选择集合后点击计算；本页只读取快照，计算结果仅保留在当前会话内。")
        return
    _show_result(result)


render_factor_page()
