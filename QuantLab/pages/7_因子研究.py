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
    factor_label, factor_explanation, ranked_factor_table, top_n_factor_preview,
)
from quantlab.ui.runtime import (  # noqa: E402
    get_research_storage_dir, get_portfolio_storage_dir, is_public_mode, require_access,
)
from quantlab.ui.portfolio import clear_portfolio_result, render_portfolio, render_portfolio_result  # noqa: E402
from quantlab.ui.portfolio_records import render_portfolio_history, historical_benchmark_message  # noqa: E402
from quantlab.ui.portfolio_sensitivity import clear_portfolio_sensitivity  # noqa: E402
from quantlab.ui.dual_factor import clear_dual_factor_preview, render_dual_factor_preview  # noqa: E402


ADJUSTMENTS = {"none": "不复权", "qfq": "前复权", "hfq": "后复权"}
MODES = {"observed_union": "行情日期并集", "supplied": "用户提供日历"}
SOURCE_LABELS = {
    "symbol": "ETF 代码", "version_id": "数据版本", "provider": "来源",
    "imported_at": "导入时间", "adjustment": "价格口径", "adjustment_basis": "口径依据",
    "file_sha256": "文件指纹", "metadata_sha256": "元数据指纹",
    "input_content_sha256": "完整历史内容指纹",
}


def _clear_result():
    for key in ("factor_result", "factor_result_identity", "factor_view_date", "factor_view_symbol",
                "factor_daily_order", "factor_rank_direction", "factor_selection_n"):
        st.session_state.pop(key, None)


def _show_provenance(result: FactorRunResult):
    with st.expander("公式、来源与研究限制"):
        definition = result.definition
        st.caption(f"因子：{definition.name} · 版本：{definition.version} · {definition.description}")
        st.code(definition.formula, language=None)
        st.write(
            f"回看 {definition.lookback_periods} 期，含当前期共 {definition.window_observations} 条观测；"
            "整个窗口必须完整，收盘数据可用后计算。" + factor_explanation(definition)
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


def _show_selection_preview(result: FactorRunResult, selected_date, *, ascending: bool):
    """筛选仅作用于这一张目标配置表，不改变原始结果与历史图。"""
    st.subheader("前 N 只 · 等权配置预览")
    n = st.number_input(
        "选择数量 N", min_value=1, max_value=len(result.sources),
        value=min(3, len(result.sources)), step=1, key="factor_selection_n",
        help="恰好选择 N 只。沿用上方日期、因子和排名方向；有效数量不足时不生成配置。",
    )
    st.caption("仅为当前日期的目标配置，不是实际持仓；不模拟成交、不计算组合收益、不自动下单。")
    preview = top_n_factor_preview(result, selected_date, n, ascending=ascending)
    if preview.table.empty:
        st.info(
            f"当日仅有 {preview.valid_count} 只 ETF 可参与筛选，需要 {n} 只，未生成配置。"
            "请减小 N 或更换日期。"
        )
        return
    if preview.cutoff_tie:
        st.warning(
            f"第 {n} 只的筛选边界存在原始因子值相同的 ETF；为保持恰好 {n} 只，"
            "按 ETF 代码升序截取。这不表示入选 ETF 更优。"
        )
    st.dataframe(
        preview.table, hide_index=True, width="stretch",
        column_config={"目标权重": st.column_config.NumberColumn(format="percent")},
    )
    st.caption(
        f"已选 {n} 只，每只目标权重 = 1/{n}；未舍入权重合计 100%，显示值合计可能有舍入差异。"
        "有效的 0 和负值仍参与筛选，不额外过滤。"
    )


def _show_daily_comparison(result: FactorRunResult):
    """完整当日表保留全部 ETF；筛选预览与历史曲线彼此独立。"""
    st.subheader("当日结果")
    selected_date = st.selectbox(
        "查看日期", result.calendar.dates, index=len(result.calendar.dates) - 1, key="factor_view_date",
        format_func=lambda value: value.isoformat(),
        help="只显示这个日期，不自动回退到更早的有效值。",
    )
    order = st.radio("当日表显示方式", ["ETF 代码", "因子排名"], horizontal=True, key="factor_daily_order")
    if order == "因子排名":
        directions = {"descending": "从高到低", "ascending": "从低到高"}
        direction = st.selectbox(
            "排名方向", list(directions), key="factor_rank_direction",
            index=int(result.definition.direction == "higher_is_more_volatile"),
            format_func=directions.__getitem__,
        )
        table = ranked_factor_table(result, selected_date, ascending=direction == "ascending")
        valid_ranks = table.loc[table["名次"].ne("—"), "名次"]
        st.caption(
            f"{selected_date.isoformat()} · {factor_label(result.definition)} · {directions[direction]} · "
            f"可参与排名 / 集合总数：{len(valid_ranks)} / {len(table)}（只 ETF）"
        )
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(
            "仅在当前集合、日期和因子周期内比较。按未四舍五入数值排名，同值并列（如 1、1、3）；"
            "并列项按 ETF 代码排列。显示值相同可能仍有细微差异，名次不一定相同。"
        )
        st.caption("名次只是所选方向的排列，不是投资优劣或买卖建议；缺失项保留在表尾，不占名次。")
        st.caption(
            "相对评分（0–100）= 100 ×（有效数量 − 平均并列名次）÷（有效数量 − 1）。"
            "按所选方向越靠前分数越高；真实同值同分，名次列仍保留 1、1、3 这样的并列名次。"
        )
        st.caption(
            "评分只表示本集合、所选日期的相对位置，不保留原值差距，不是收益率或上涨概率；"
            "不同日期或研究池的同分不能直接等同。评分仅供观察，不改变下方 Top N 或组合回测。"
        )
        if len(valid_ranks) < 2:
            st.info("当日可参与排名的 ETF 不足两只，相对评分留空，无法进行有效的横向比较；原始因子状态与名次保留。")
        elif valid_ranks.nunique() == 1:
            st.info("当日可用因子值全部相同，均为并列第 1 名、相对评分 50.00，无法据此区分 ETF。")
    else:
        st.dataframe(daily_factor_table(result, selected_date), hide_index=True, width="stretch")
        st.caption("按 ETF 代码排列，不是排名。")
    zero_label = "0.00" if result.definition.direction == "higher_is_more_upward_efficient" else "0%"
    st.caption(f"因子值列的 — 表示不可计算，不是 {zero_label}；原因只描述当前输入。")
    if order == "因子排名":
        _show_selection_preview(result, selected_date, ascending=direction == "ascending")


def _show_result(result: FactorRunResult):
    axis = result.calendar
    st.caption(
        f"快照区间：{result.start} 至 {result.end} · {len(result.sources)} 只 ETF · "
        f"{len(axis.dates)} 个日期位置 · {ADJUSTMENTS[result.adjustment]} · {MODES[axis.mode]}"
    )
    if result.adjustment == "none":
        st.caption("当前使用未复权价格，分红、拆并份额可能影响价格变化及其波动；不是总回报口径。")
    _show_daily_comparison(result)

    st.subheader(f"单只 ETF {factor_label(result.definition)}变化")
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


def _render_current_research(research_dir, archive_dir):
    """集合不可用只结束当前研究区域，不阻断独立的历史档案。"""
    scope = str(research_dir)
    try:
        scan = list_research_sets(research_dir)
    except (ValueError, OSError):
        _clear_result()
        clear_portfolio_result()
        clear_portfolio_sensitivity()
        clear_dual_factor_preview()
        st.error("无法读取研究集合目录，请检查本地目录权限后重试。")
        return
    if scan.issues:
        st.warning(f"有 {len(scan.issues)} 个集合未通过校验，已从可选列表排除；请到‘多 ETF 数据’检查。")
    records = {record.set_id: record for record in scan.records}
    if not records:
        _clear_result()
        clear_portfolio_result()
        clear_portfolio_sensitivity()
        clear_dual_factor_preview()
        st.session_state.pop("factor_set_id", None)
        st.info("还没有可用的已保存集合。请先到‘多 ETF 数据’生成并保存研究集合。")
        return
    if st.session_state.get("factor_set_id") not in records:
        _clear_result()
        clear_portfolio_result()
        clear_portfolio_sensitivity()
        clear_dual_factor_preview()
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
        clear_portfolio_result()
        clear_portfolio_sensitivity()
        clear_dual_factor_preview()
        st.error("所选集合无法载入或校验未通过。旧结果已清除，请返回‘多 ETF 数据’检查后重试。")
        return
    registry = create_default_registry()
    factor_names = sorted(
        registry.names(), key=lambda name: (name.partition("_")[0], registry.get(name).definition.lookback_periods),
    )
    if st.session_state.get("factor_name") not in factor_names:
        st.session_state.pop("factor_name", None)
    factor_name = st.selectbox(
        "选择因子", factor_names, key="factor_name",
        index=factor_names.index("momentum_20"),
        format_func=lambda name: registry.get(name).definition.description,
    )
    definition = registry.get(factor_name).definition
    periods = definition.lookback_periods
    if definition.direction == "higher_is_more_volatile":
        st.caption(
            f"{definition.description} = 最近 {periods} 个简单收益率的样本标准差 × √252；"
            f"需要 {definition.window_observations} 条完整收盘价观测。252 为固定年化假设。"
        )
        st.caption(factor_explanation(definition))
    elif definition.direction == "higher_is_more_above_ma":
        st.caption(
            f"{definition.description} = 当前收盘价 ÷ 最近 {definition.window_observations} 条收盘价的简单均值 − 1；"
            f"需要含当前期的 {definition.window_observations} 条完整观测，前 {periods} 个日期位置为热身期。"
        )
        st.caption(factor_explanation(definition))
    elif definition.direction == "higher_is_less_drawdown":
        st.caption(
            f"{definition.description}：在含当前期的 {definition.window_observations} 条收盘价窗口内，"
            "逐期计算‘价格 ÷ 截至该期的窗口内最高价 − 1’，取最小值。"
            f"前 {periods} 个日期位置为热身期，不使用窗口外高点。"
        )
        st.caption(factor_explanation(definition))
    elif definition.direction == "higher_is_more_volume":
        st.caption(
            f"{definition.description} = 当日成交量 ÷ 此前 {periods} 期平均成交量 − 1；"
            f"分母不含当日，需要 {definition.window_observations} 条完整观测，前 {periods} 个日期位置为热身期。"
        )
        st.caption("分母为 0 时不可用；历史均量大于 0 时，当日真实零成交量为有效的 -100%，不等于缺失行情。")
        st.caption(factor_explanation(definition))
    elif definition.direction == "higher_is_more_upward_efficient":
        st.caption(
            f"{definition.description} = 首尾净价格变化 ÷ 最近 {periods} 期逐期价格变化绝对值之和；"
            f"需要 {definition.window_observations} 条完整收盘价，前 {periods} 个日期位置为热身期。"
        )
        st.caption("恒价窗口按定义为 0.00，首尾相同的震荡路径也为 0.00；缺失数据仍不可用，不填成零。")
        st.caption(factor_explanation(definition))
    else:
        st.caption(f"{periods} 期价格动量 = 当前收盘价 ÷ {periods} 期前收盘价 − 1；需要 {definition.window_observations} 条完整观测。")
    identity = (scope, selected, loaded.preview.preview_sha256, definition.name, definition.version)
    if st.session_state.get("factor_result_identity") != identity:
        _clear_result()
    observation, portfolio, pair = st.tabs(["因子观察", "组合回测", "双因子预览"])
    with observation:
        if st.button(f"计算 {factor_label(definition)}", key="factor_calculate", type="primary"):
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
        else:
            _show_result(result)
    with portfolio:
        render_portfolio(loaded.preview, definition, identity, archive_dir=archive_dir)
    with pair:
        render_dual_factor_preview(loaded.preview, (scope, selected, loaded.preview.preview_sha256))


def render_factor_page():
    # 访问校验必须早于任何本地研究集合或档案读取。
    require_access()
    research_dir = get_research_storage_dir(PROJECT_ROOT)
    archive_dir = get_portfolio_storage_dir(PROJECT_ROOT)
    scope = str(research_dir)
    if st.session_state.get("factor_storage_scope") != scope:
        for key in list(st.session_state):
            if key.startswith(("factor_", "portfolio_")):
                del st.session_state[key]
        st.session_state["factor_storage_scope"] = scope
    st.title("因子研究")
    st.caption("查看历史动量、波动率、均线偏离、滚动价格回撤、成交量变化与趋势效率；期数按研究日期轴计算，不是自然日。")
    st.info("因子观察描述历史价格与成交量变化；组合回测按指定规则模拟交易。排名不等于策略收益，本页不自动下单。")
    if is_public_mode():
        st.warning("当前为临时会话，仅显示本会话的研究集合和实验档案；结果不会自动保存。")
    current, history = st.tabs(["当前研究", "历史实验"])
    with current:
        _render_current_research(research_dir, archive_dir)
    with history:
        loaded = render_portfolio_history(archive_dir)
        if loaded is not None:
            render_portfolio_result(
                loaded.strategy, benchmark=loaded.benchmark, benchmark_status=loaded.benchmark_status,
                benchmark_message=historical_benchmark_message(loaded), historical=True,
                namespace="portfolio_history",
            )


render_factor_page()
