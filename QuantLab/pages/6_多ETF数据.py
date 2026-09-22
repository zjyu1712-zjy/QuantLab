from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantlab.data.calendar import parse_calendar_csv  # noqa: E402
from quantlab.data.catalog import scan_market_catalog  # noqa: E402
from quantlab.data.research_sets import DatasetSelection, build_research_set  # noqa: E402
from quantlab.data.research_storage import list_research_sets, load_research_set, save_research_set  # noqa: E402
from quantlab.ui.multi_asset import can_save_preview, friendly_research_error, research_request_sha256  # noqa: E402
from quantlab.ui.runtime import get_market_storage_dirs, get_research_storage_dir, is_public_mode, require_access  # noqa: E402


ADJUSTMENTS = {None: "请确认口径", "none": "不复权", "qfq": "前复权", "hfq": "后复权"}
MODES = {"observed_union": "行情日期并集", "supplied": "上传交易日历"}
REPORT_LABELS = {
    "symbol": "ETF 代码", "first_observed": "首条观测", "last_observed": "末条观测",
    "calendar_count": "日期数", "observation_count": "有行情", "missing_count": "缺失数",
    "missing_ratio": "缺失比例", "longest_missing_run": "最长连续缺失", "zero_volume_count": "零成交量",
    "version_id": "数据版本", "provider": "来源", "adjustment": "复权口径", "adjustment_basis": "口径依据",
}


def _show_error(error):
    message, detail = friendly_research_error(error)
    st.error(message)
    if detail:
        with st.expander("高级错误信息（可能包含本地路径）"):
            st.code(detail, language=None)


def _show_issues(issues):
    if issues:
        st.warning(f"有 {len(issues)} 项数据或集合不可用，已从可选列表排除。")
        with st.expander("查看不可用项目的详细原因"):
            for issue in issues:
                st.text(issue)


def _clear_preview():
    st.session_state.pop("research_preview_data", None)
    st.session_state.pop("research_preview_request_sha256", None)


def _show_preview(preview):
    axis = preview.alignment.calendar
    columns = st.columns(3)
    columns[0].metric("ETF 数量", len(preview.sources))
    columns[1].metric("日期轴长度", len(axis.dates))
    columns[2].metric("共同日期数", len(preview.alignment.common_dates))
    st.caption(f"研究区间：{preview.start} 至 {preview.end}（含端点） · 价格口径：{ADJUSTMENTS[preview.adjustment]}")
    st.caption(f"日期轴：{MODES[axis.mode]} · 来源：{axis.source} · 声明覆盖：{axis.coverage_start} 至 {axis.coverage_end}")
    for warning in preview.warnings:
        st.warning(warning)
    report = preview.coverage.copy()
    report["adjustment"] = report.adjustment.map(ADJUSTMENTS)
    report["adjustment_basis"] = report.adjustment_basis.map({"provider": "数据源记录", "user_declared": "用户声明"})
    st.dataframe(report.rename(columns=REPORT_LABELS), hide_index=True, width="stretch")
    with st.expander("对齐明细（前 200 行）与共同日期"):
        st.caption("has_observation 只表示有有效记录，不代表可以成交；空白行情没有被填充。")
        st.dataframe(preview.alignment.data.head(200), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame({"共同日期": preview.alignment.common_dates}), hide_index=True)
    with st.expander("高级：固定版本、来源路径与内容指纹"):
        st.dataframe(pd.DataFrame(preview.sources), hide_index=True, width="stretch")
        st.code(preview.preview_sha256, language=None)


def _calendar_inputs(start, end):
    mode = st.radio("日期轴方式", list(MODES), format_func=MODES.get, key="research_calendar_mode", horizontal=True)
    if mode == "observed_union":
        st.caption("默认使用行情中出现过的日期；如果某天所有 ETF 都缺记录，并集无法发现它。")
        return None, None
    upload = st.file_uploader("交易日历 CSV（UTF-8，仅 date 一列）", type=["csv"], key="research_calendar_file")
    source = st.text_input("日历来源", key="research_calendar_source", placeholder="例如：交易所日历导出；请自行确认来源")
    columns = st.columns(2)
    coverage_start = columns[0].date_input("日历声明覆盖开始", value=start, key="research_calendar_start",
                                         min_value=date(1900, 1, 1), max_value=date(2100, 12, 31))
    coverage_end = columns[1].date_input("日历声明覆盖结束", value=end, key="research_calendar_end",
                                       min_value=date(1900, 1, 1), max_value=date(2100, 12, 31))
    if upload is None:
        return None, ValueError("请上传交易日历，填写来源和覆盖范围；上传不代表官方认证。")
    try:
        return parse_calendar_csv(upload.getvalue(), source=source, coverage_start=coverage_start, coverage_end=coverage_end), None
    except ValueError as error:
        return None, error


def _render_create(raw_dir, processed_dir, research_dir):
    try:
        scan = scan_market_catalog(processed_dir, raw_dir)
    except (ValueError, OSError) as error:
        _clear_preview()
        _show_error(error)
        return
    _show_issues(scan.issues)
    if len({record.symbol for record in scan.records}) < 2:
        _clear_preview()
        st.info("请先到‘添加资产数据’导入至少两只 ETF 的历史行情；这里不会自动下载或拼接版本。")
        return
    name = st.text_input("研究集合名称", value="我的 ETF 研究集合", key="research_name")
    symbols = st.multiselect("选择 ETF（至少两只）", sorted({record.symbol for record in scan.records}), key="research_symbols")
    selections = []
    for symbol in sorted(symbols):
        versions = {record.version_id: record for record in scan.records if record.symbol == symbol}
        columns = st.columns([3, 2])
        selected = columns[0].selectbox(
            f"{symbol} 数据版本", list(versions), key=f"research_version_{symbol}",
            format_func=lambda version, records=versions: f"{version} · {records[version].provider} · {records[version].row_count} 条",
        )
        record = versions[selected]
        known = record.adjustment is not None
        adjustment = columns[1].selectbox(
            f"{symbol} 价格口径", [record.adjustment] if known else list(ADJUSTMENTS),
            format_func=ADJUSTMENTS.get, key=f"research_adjustment_{symbol}", disabled=known,
            help="数据源已有口径时不可覆盖；CSV 未记录口径时，请查明后明确声明。",
        )
        selections.append(DatasetSelection(record, None if known else adjustment))
    columns = st.columns(2)
    start = columns[0].date_input("研究开始日期", value=date(2000, 1, 1), key="research_start",
                                   min_value=date(1900, 1, 1), max_value=date(2100, 12, 31))
    end = columns[1].date_input("研究结束日期", value=date.today(), key="research_end",
                                 min_value=date(1900, 1, 1), max_value=date(2100, 12, 31))
    st.caption("默认区间为 2000 年至今，可按研究需要修改；不会为区间内未出现的自然日补行情。")
    calendar, calendar_error = _calendar_inputs(start, end)
    selected_tuple = tuple(selections)
    current = None if calendar_error else research_request_sha256(
        name=name, selections=selected_tuple, start=start, end=end, calendar=calendar,
    )
    previous = st.session_state.get("research_preview_request_sha256")
    if previous is not None and not can_save_preview(current, previous):
        _clear_preview()
        st.info("输入已变化，旧预览已失效。请重新生成后再保存。")
    if st.button("生成预览", key="research_preview", type="primary"):
        _clear_preview()
        try:
            if calendar_error:
                raise calendar_error
            with st.spinner("正在读取固定版本并检查日期与口径……"):
                preview = build_research_set(selected_tuple, start=start, end=end, calendar=calendar)
            st.session_state["research_preview_data"] = preview
            st.session_state["research_preview_request_sha256"] = current
        except (ValueError, OSError) as error:
            _show_error(error)
    preview = st.session_state.get("research_preview_data")
    valid = preview is not None and can_save_preview(current, st.session_state.get("research_preview_request_sha256"))
    if valid:
        st.subheader("当前预览")
        _show_preview(preview)
    st.caption("保存的是本次预览快照，不会重新选择最新行情；再次点击保存会生成独立集合。")
    if st.button("保存本次预览", key="research_save", disabled=not valid):
        # 不依赖浏览器 disabled；保存事件始终重查请求与完整状态。
        current_preview = st.session_state.get("research_preview_data")
        if current_preview is None or not can_save_preview(current, st.session_state.get("research_preview_request_sha256")):
            st.error("当前配置没有匹配的预览，请先重新生成。")
        else:
            try:
                saved = save_research_set(current_preview, research_dir, name=name)
                st.success(f"已保存：{saved.name} · 集合 ID：{saved.set_id}")
            except (ValueError, OSError) as error:
                _show_error(error)


def _render_saved(research_dir):
    try:
        scan = list_research_sets(research_dir)
    except (ValueError, OSError) as error:
        st.session_state.pop("research_loaded_set", None)
        _show_error(error)
        return
    _show_issues(scan.issues)
    if not scan.records:
        st.session_state.pop("research_loaded_set", None)
        st.info("暂无可载入的已保存集合。保存成功后可在这里重新查看。")
        return
    records = {record.set_id: record for record in scan.records}
    selected = st.selectbox("选择已保存集合", list(records), key="research_saved_id",
                            format_func=lambda value: f"{records[value].name} · {records[value].created_at[:19]} UTC · {value[:8]}")
    loaded = st.session_state.get("research_loaded_set")
    if loaded is not None and loaded.record.set_id != selected:
        st.session_state.pop("research_loaded_set", None)
    if st.button("载入集合", key="research_load"):
        st.session_state.pop("research_loaded_set", None)
        try:
            st.session_state["research_loaded_set"] = load_research_set(research_dir, selected)
            st.success("集合已载入，文件与内容校验通过。")
        except (ValueError, OSError) as error:
            _show_error(error)
    loaded = st.session_state.get("research_loaded_set")
    if loaded is not None:
        st.subheader(f"已载入：{loaded.record.name}")
        _show_preview(loaded.preview)


require_access()
raw_dir, processed_dir = get_market_storage_dirs(PROJECT_ROOT)
research_dir = get_research_storage_dir(PROJECT_ROOT)
scope = (str(raw_dir), str(processed_dir), str(research_dir))
if st.session_state.get("research_storage_scope") != scope:
    for key in list(st.session_state):
        if key.startswith("research_"):
            del st.session_state[key]
    st.session_state["research_storage_scope"] = scope

st.title("多 ETF 数据")
st.caption("固定版本 · 日期对齐 · 缺失检查 · 研究快照")
st.info("研究集合不等于历史可交易股票池。本页不计算因子、不做组合回测，也不会自动下单。")
if is_public_mode():
    st.warning("当前为临时会话：行情与研究集合仅对本会话可见，结束后不保证保留，请勿作为长期存档。")
else:
    st.caption("本地模式：集合保存在项目数据目录中，刷新页面后仍可重新载入。")
create_tab, saved_tab = st.tabs(["新建研究集合", "已保存集合"])
with create_tab:
    _render_create(raw_dir, processed_dir, research_dir)
with saved_tab:
    _render_saved(research_dir)
