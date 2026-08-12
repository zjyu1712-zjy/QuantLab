from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantlab.ui import (  # noqa: E402
    build_market_performance_view,
    get_market_storage_dirs,
    is_public_mode,
    list_market_datasets,
    load_market_dataset,
    require_access,
)


RAW_MARKET_DIR, PROCESSED_MARKET_DIR = get_market_storage_dirs(PROJECT_ROOT)


@st.cache_data(show_spinner=False)
def _load_cached_market_dataset(
    processed_path: str,
    modified_time_ns: int,
) -> pd.DataFrame:
    del modified_time_ns
    return load_market_dataset(processed_path)


def _format_percent(value: float) -> str:
    return f"{value:.2%}"


def _render_growth_path(has_asset: bool) -> None:
    with st.container(border=True):
        steps = st.columns(5)
        labels = (
            ("选择资产", "添加并管理数据"),
            ("自动检查", "校验与清洗完成"),
            ("开始分析", "了解表现与风险"),
            ("策略实验", "验证不同投资方法"),
            ("投资复盘", "总结并持续改进"),
        )
        for index, (title, caption) in enumerate(labels, start=1):
            state = "已完成" if has_asset and index <= 2 else "下一步"
            if not has_asset and index > 1:
                state = "待开始"
            steps[index - 1].markdown(f"**{index}. {title}**")
            steps[index - 1].caption(f"{caption} · {state}")


def _render_empty_home() -> None:
    st.subheader("我的投资")
    st.write(
        "添加一只 ETF 的历史行情后，QuantLab 会先自动检查数据，"
        "再为你生成收益、风险和投资实验入口。"
    )
    st.page_link(
        "pages/1_数据管理.py",
        label="添加第一只 ETF",
        icon=":material/add_chart:",
    )
    st.info(
        "你不需要先理解数据格式。可以上传常见行情文件，"
        "也可以下载模板后再填写。",
        icon=":material/lightbulb:",
    )


def _render_investment_snapshot(record) -> None:
    processed_path = Path(record.processed_path)
    market_data = _load_cached_market_dataset(
        str(processed_path),
        processed_path.stat().st_mtime_ns,
    )
    view = build_market_performance_view(market_data)
    latest_close = float(view.detail["close"].iloc[-1])

    title_col, action_col = st.columns([3, 1])
    with title_col:
        st.subheader("我的投资现在怎么样")
        st.caption(
            f"当前关注：{view.symbol} · {view.start_date} 至 {view.end_date}"
        )
    with action_col:
        st.page_link(
            "pages/2_投资表现分析.py",
            label="继续分析",
            icon=":material/arrow_forward:",
        )

    metrics = st.columns(3)
    metrics[0].metric("最新收盘价", f"{latest_close:.4f}")
    metrics[1].metric(
        "累计收益率",
        _format_percent(view.analysis.cumulative_return),
    )
    metrics[2].metric(
        "最大回撤",
        _format_percent(view.analysis.max_drawdown),
    )

    st.markdown("#### 净值曲线")
    st.line_chart(
        view.detail.set_index("date")[["net_value"]],
        x_label="日期",
        y_label="净值（起点 = 1）",
        width="stretch",
    )

    lower = st.columns(2)
    with lower[0]:
        st.markdown("#### 最近交易")
        st.caption(
            "上传个人交易记录后，这里会显示最近成交、方向和实际费用。"
        )
        st.page_link(
            "pages/5_交易复盘.py",
            label="导入交易记录",
            icon=":material/receipt_long:",
        )
    with lower[1]:
        st.markdown("#### 最近一次复盘")
        st.caption(
            "完成一次交易复盘后，这里会保留最值得关注的行为发现。"
        )
        st.page_link(
            "pages/5_交易复盘.py",
            label="开始复盘",
            icon=":material/fact_check:",
        )


def _render_next_areas() -> None:
    st.divider()
    st.subheader("从分析到改进")
    areas = st.columns(3)
    with areas[0]:
        st.markdown("### 分析")
        st.write("看清收益、风险、净值和回撤，回答投资表现如何。")
        st.page_link(
            "pages/2_投资表现分析.py",
            label="查看投资表现",
            icon=":material/analytics:",
        )
    with areas[1]:
        st.markdown("### 实验")
        st.write("用历史数据验证不同方法，观察结果会有什么变化。")
        st.page_link(
            "pages/3_双均线回测.py",
            label="创建策略实验",
            icon=":material/science:",
        )
    with areas[2]:
        st.markdown("### 改进")
        st.write("对比真实交易与规则信号，找到下一次可以改进的地方。")
        st.page_link(
            "pages/5_交易复盘.py",
            label="进行交易复盘",
            icon=":material/fact_check:",
        )


def render_home_page() -> None:
    st.markdown('<div class="ql-eyebrow">QuantLab · 个人投资量化成长实验室</div>', unsafe_allow_html=True)
    st.title("继续建立你的科学投资体系")
    st.caption(
        "用数据记录投资，用分析发现问题，用实验验证方法，用复盘持续改进。"
    )
    if is_public_mode():
        st.caption("当前为临时分享会话；上传内容不会与其他访客共享。")

    try:
        records = list_market_datasets(PROCESSED_MARKET_DIR, RAW_MARKET_DIR)
    except Exception:
        records = []

    _render_growth_path(bool(records))
    st.write("")
    if not records:
        _render_empty_home()
    else:
        try:
            _render_investment_snapshot(records[0])
        except Exception as error:
            st.warning("这份资产数据暂时无法生成总览，请先检查资产数据。")
            with st.expander("查看详细原因"):
                st.code(str(error), language=None)
            st.page_link(
                "pages/1_数据管理.py",
                label="检查资产数据",
                icon=":material/rule:",
            )
    _render_next_areas()
    st.caption(
        f"截至 {date.today().isoformat()} · QuantLab 不连接券商，也不会自动下单。"
    )


require_access()
render_home_page()
