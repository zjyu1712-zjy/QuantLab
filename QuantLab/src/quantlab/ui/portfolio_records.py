"""组合实验的显式保存和历史选择；不运行回测，不持有历史分析缓存。"""

from datetime import date
from hashlib import sha256
from pathlib import Path

import streamlit as st

from quantlab.portfolio import (
    PortfolioBacktestError, list_portfolio_runs, load_portfolio_run, save_portfolio_run, build_portfolio_export,
)
from quantlab.ui.portfolio_display import portfolio_error_message
from quantlab.ui.runtime import is_public_mode


def render_portfolio_save(analysis, output_dir, *, benchmark=None, benchmark_error=None):
    """只有明确点击按钮才写盘；失败和重复保存均保留当前分析。"""
    context = (str(output_dir), analysis.backtest.metadata["run_id"])
    if st.session_state.get("portfolio_archive_draft_context") != context:
        st.session_state["portfolio_archive_draft_context"] = context
        meta = analysis.backtest.metadata
        st.session_state["portfolio_archive_name"] = (
            f"{meta['factor_definition']['description']} · Top {meta['strategy']['config']['top_n']}"
        )
        st.session_state.pop("portfolio_archive_notice", None)
    with st.expander("保存本次实验", expanded=True):
        if is_public_mode():
            st.warning("当前为临时公开会话：保存仅在本会话有效，会话结束后可能清除，不写入本地个人档案。")
        st.caption("保存完整参数、数据快照、交易明细及策略和基准结果。仅点击下方按钮才会写入档案。")
        with st.form("portfolio_archive_form"):
            name = st.text_input("实验名称", key="portfolio_archive_name", max_chars=200)
            submitted = st.form_submit_button("保存实验", key="portfolio_archive_save")
        if submitted:
            if not name.strip():
                st.session_state["portfolio_archive_notice"] = ("error", "请填写实验名称；尚未保存。")
            else:
                try:
                    record = save_portfolio_run(
                        analysis, output_dir, name=name, benchmark=benchmark, benchmark_error=benchmark_error,
                    )
                except FileExistsError:
                    st.session_state["portfolio_archive_notice"] = (
                        "info", "档案已存在或保存位置有同名路径冲突，未覆盖。请到“历史实验”核对；"
                        "若找不到本次运行，请检查档案目录。更换名称不会覆盖原文件。",
                    )
                except (ValueError, OSError, OverflowError):
                    st.session_state["portfolio_archive_notice"] = (
                        "error", "实验未能保存，当前回测结果仍保留。请检查目录权限；若程序刚更新，请重启后重新运行再保存。",
                    )
                else:
                    st.session_state["portfolio_archive_notice"] = (
                        "success", f"已保存实验：{record.name}（运行编号 {record.run_id[:8]}）。可在“历史实验”中查看。",
                    )
        notice = st.session_state.get("portfolio_archive_notice")
        if notice:
            getattr(st, notice[0])(notice[1])


def _clear_selection():
    st.session_state.pop("portfolio_history_id", None)


def render_portfolio_history(output_dir):
    """仅返回本次校验后读到的历史档案；调用方用显式参数展示，不写入当前运行状态。"""
    scope = str(Path(output_dir))
    if st.session_state.get("portfolio_history_scope") != scope:
        _clear_selection()
        st.session_state["portfolio_history_scope"] = scope
    st.subheader("已保存的组合实验")
    st.caption("原回测和全区间指标按保存值展示，不重新计算因子或重跑策略；年度统计、收益集中度与基准涨跌分组是基于存档曲线的派生分析。无需原研究集合和行情文件。")
    if is_public_mode():
        st.warning("仅显示本临时会话保存的实验，不会读取本地个人档案或其他会话。")
    try:
        scan = list_portfolio_runs(output_dir)
    except (ValueError, OSError):
        _clear_selection()
        st.error("无法读取组合档案目录，请检查目录权限后重试；当前回测结果不受影响。")
        return None
    if scan.issues:
        st.warning(f"有 {len(scan.issues)} 份档案无法读取或未通过校验，已从列表排除；原文件未修改。")
    records = {record.run_id: record for record in scan.records}
    if not records:
        _clear_selection()
        st.info("还没有可读取的历史实验。完成组合回测后，可在“保存本次实验”中主动保存。")
        return None
    if st.session_state.get("portfolio_history_id") not in records:
        _clear_selection()
    selected = st.selectbox(
        "选择历史实验", list(records), index=None, placeholder="选择一份已保存的实验",
        key="portfolio_history_id",
        format_func=lambda key: f"{records[key].name} · {records[key].saved_at[:19]} UTC · {key[:8]}",
    )
    if selected is None:
        return None
    try:
        loaded = load_portfolio_run(output_dir, selected)
    except (ValueError, OSError):
        st.error("所选档案无法读取或校验未通过，未展示旧历史结果；当前回测结果不受影响。")
        return None
    st.caption(f"实验名称：{loaded.record.name} · 保存时间：{loaded.record.saved_at[:19]} UTC")
    st.caption(f"原运行编号：{loaded.record.run_id} · 原运行时间：{loaded.strategy.backtest.metadata['run_at']}")
    for warning in loaded.warnings:
        st.warning(warning)
    render_portfolio_export(loaded)
    return loaded


def render_portfolio_export(loaded):
    """下载与当前选中的历史快照绑定；打包失败不清除历史展示或当前回测。"""
    with st.expander("导出本次历史实验", expanded=True):
        st.caption("实验包含完整 JSON 备份、策略与基准 CSV 表格及中文说明。只导出保存值，不重新回测。")
        st.warning("包含完整行情、因子快照和已保存元数据，可能含私人信息；分享前请检查。")
        try:
            result = build_portfolio_export(loaded)
        except (ValueError, OSError, OverflowError):
            st.error("实验包生成失败，未提供下载；已保存档案、历史展示和当前回测结果均未修改。请重新选择档案后重试。")
            return
        st.download_button(
            "下载实验包（ZIP）", data=result.content, file_name=result.file_name, mime="application/zip",
            key=f"portfolio_export_{loaded.record.run_id}_{sha256(loaded.archive_bytes).hexdigest()}",
            on_click="ignore",
        )
        st.caption("CSV 收益率使用比例值（0.05 = 5%），缺失项留空；精确备份以 JSON 为准。下载不会修改或另存本地档案。")


def historical_benchmark_message(loaded):
    """把档案里的有限失败上下文交给既有中文错误翻译器，不展示异常全文。"""
    if loaded.benchmark_failure is None:
        return ""
    failure = loaded.benchmark_failure
    error = PortfolioBacktestError(
        "基准未完成", reason=failure["reason"],
        date=date.fromisoformat(failure["date"]) if failure["date"] else None,
        symbols=tuple(failure["symbols"]),
    )
    return portfolio_error_message(error)
