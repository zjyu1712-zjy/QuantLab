"""页面共享的回测保存与下载入口。"""

from pathlib import Path

import streamlit as st

from quantlab.analytics.performance import PerformanceAnalysis
from quantlab.backtest import BacktestResult, save_backtest_run, serialize_backtest_run


def render_run_record(
    result: BacktestResult,
    performance: PerformanceAnalysis,
    output_dir: Path,
    *,
    label: str = "本次回测",
) -> None:
    """页面每次显式运行后保存；即使磁盘保存失败也允许用户下载。"""
    serialized = serialize_backtest_run(result, performance=performance)
    with st.expander(f"{label}：实验记录与下载"):
        try:
            saved_path = save_backtest_run(result, output_dir, performance=performance)
            st.caption(f"已保存实验记录：{saved_path.name}")
        except OSError:
            st.warning("计算已完成，但本地记录未能保存。请下载本次记录并检查目录权限。")
        metadata = result.metadata
        st.write(f"运行时间（UTC）：{metadata.run_at}")
        st.write(f"数据版本：{metadata.data['version']}")
        st.caption("记录包含本次行情、信号、参数、成交、资金曲线和代码指纹，便于日后复核。")
        st.download_button(
            f"下载{label}记录（JSON）",
            data=serialized,
            file_name=f"quantlab-{metadata.run_id}.json",
            mime="application/json",
            key=f"download-{metadata.run_id}",
            on_click="ignore",
        )
