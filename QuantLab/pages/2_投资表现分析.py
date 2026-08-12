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
    """文件没有变化时复用已读取的标准化行情。"""
    del modified_time_ns
    return load_market_dataset(processed_path)


def _format_percent(value: float) -> str:
    return f"{value:.2%}"


def render_performance_page() -> None:
    """渲染单 ETF 投资表现分析页面。"""
    st.markdown('<div class="ql-eyebrow">分析</div>', unsafe_allow_html=True)
    st.title("我的投资表现")
    st.caption(
        "看清这只 ETF 的收益、风险和波动过程，回答“我的投资表现如何”。"
    )

    try:
        records = list_market_datasets(
            PROCESSED_MARKET_DIR,
            RAW_MARKET_DIR,
        )
    except Exception as error:
        st.error(f"读取行情目录失败：{error}")
        return

    if not records:
        st.warning(
            "还没有可分析的资产，请先前往“添加资产数据”。",
            icon=":material/add_chart:",
        )
        return

    selected_index = st.selectbox(
        "选择资产",
        options=range(len(records)),
        format_func=lambda index: (
            f"{records[index].symbol} · "
            f"{records[index].row_count or '未知'} 条行情 · "
            f"添加于 {records[index].imported_at[:10]}"
        ),
    )
    record = records[selected_index]
    processed_path = Path(record.processed_path)

    try:
        market_data = _load_cached_market_dataset(
            str(processed_path),
            processed_path.stat().st_mtime_ns,
        )
    except Exception as error:
        st.error("暂时无法读取这份资产数据，请返回“添加资产数据”重新检查。")
        with st.expander("查看详细原因"):
            st.code(str(error), language=None)
        return

    earliest_date = pd.to_datetime(market_data["date"]).min().date()
    latest_date = pd.to_datetime(market_data["date"]).max().date()

    control_columns = st.columns([1, 1, 1])
    start_date = control_columns[0].date_input(
        "开始日期",
        value=earliest_date,
        min_value=earliest_date,
        max_value=latest_date,
    )
    end_date = control_columns[1].date_input(
        "结束日期",
        value=latest_date,
        min_value=earliest_date,
        max_value=latest_date,
    )
    risk_free_rate_percent = control_columns[2].number_input(
        "年化无风险利率（%）",
        min_value=-99.0,
        max_value=100.0,
        value=0.0,
        step=0.1,
        help="只影响夏普比率，默认按 0% 计算。",
    )

    if not isinstance(start_date, date) or not isinstance(end_date, date):
        st.error("请选择有效的开始和结束日期")
        return
    if start_date > end_date:
        st.error("开始日期不能晚于结束日期")
        return

    try:
        with st.spinner("正在计算收益和风险指标……"):
            view = build_market_performance_view(
                market_data,
                start_date=start_date,
                end_date=end_date,
                annual_risk_free_rate=(
                    risk_free_rate_percent / 100
                ),
            )
    except Exception as error:
        st.error(f"无法完成投资表现分析：{error}")
        return

    st.caption(
        f"ETF {view.symbol} · {view.start_date} 至 {view.end_date} · "
        f"{view.observation_count} 个交易日"
    )

    st.subheader("核心指标")
    metrics = st.columns(5)
    metrics[0].metric(
        "累计收益率",
        _format_percent(view.analysis.cumulative_return),
    )
    metrics[1].metric(
        "年化收益率",
        _format_percent(view.analysis.annualized_return),
    )
    metrics[2].metric(
        "年化波动率",
        _format_percent(view.analysis.annualized_volatility),
    )
    metrics[3].metric(
        "最大回撤",
        _format_percent(view.analysis.max_drawdown),
    )
    metrics[4].metric(
        "夏普比率",
        f"{view.analysis.sharpe_ratio:.2f}",
    )

    st.subheader("净值曲线")
    st.line_chart(
        view.detail.set_index("date")[["net_value"]],
        x_label="日期",
        y_label="净值（起点 = 1）",
        width="stretch",
    )

    st.subheader("回撤曲线")
    st.area_chart(
        view.detail.set_index("date")[["drawdown"]],
        x_label="日期",
        y_label="回撤",
        width="stretch",
    )

    st.subheader("每日明细")
    display_data = view.detail.copy()
    display_data["date"] = display_data["date"].dt.date
    st.dataframe(
        display_data,
        hide_index=True,
        width="stretch",
        column_config={
            "date": "日期",
            "symbol": "ETF",
            "close": st.column_config.NumberColumn(
                "收盘价",
                format="%.4f",
            ),
            "daily_return": st.column_config.NumberColumn(
                "日收益率",
                format="percent",
            ),
            "net_value": st.column_config.NumberColumn(
                "净值",
                format="%.4f",
            ),
            "drawdown": st.column_config.NumberColumn(
                "回撤",
                format="percent",
            ),
        },
    )

    with st.expander("这些指标如何理解？"):
        st.markdown(
            """
- **累计收益率**：观察期末相对期初的总涨跌幅。
- **年化收益率**：把观察期复合收益换算成一年的增长率。
- **年化波动率**：日收益率样本标准差乘以 `√252`。
- **最大回撤**：净值相对历史高点出现过的最大跌幅。
- **夏普比率**：每承担一单位波动所获得的年化超额收益。

短样本的年化结果可能非常极端，应结合数据区间谨慎解释。
            """
        )

    with st.expander("高级信息：数据来源"):
        st.write(f"数据版本：{record.version_id}")
        st.write(f"添加时间：{record.imported_at[:19]} UTC")


require_access()
render_performance_page()
