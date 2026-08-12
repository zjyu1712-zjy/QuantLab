from __future__ import annotations

from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantlab.ui import (  # noqa: E402
    TIMING_LABELS,
    build_trade_review_view,
    get_market_storage_dirs,
    list_market_datasets,
    load_market_dataset,
    load_uploaded_trade_csv,
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


@st.cache_data(show_spinner=False)
def _load_cached_trade_upload(
    filename: str,
    content: bytes,
) -> pd.DataFrame:
    return load_uploaded_trade_csv(filename, content)


@st.cache_data(show_spinner=False)
def _run_cached_review(
    market_data: pd.DataFrame,
    trade_data: pd.DataFrame,
    short_window: int,
    long_window: int,
    initial_cash: float,
    tolerance_trading_days: int,
):
    return build_trade_review_view(
        market_data,
        trade_data,
        short_window=short_window,
        long_window=long_window,
        initial_cash=initial_cash,
        tolerance_trading_days=tolerance_trading_days,
    )


def _format_percent(value: float) -> str:
    return f"{value:.2%}"


def _format_optional_percent(value: float | None) -> str:
    return "暂无" if value is None else _format_percent(value)


def _format_optional_ratio(value: float | None) -> str:
    return "暂无" if value is None else f"{value:.2f}"


def _render_timing_chart(summary: pd.DataFrame) -> None:
    chart = (
        alt.Chart(summary)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X("count:Q", title="数量", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y(
                "label:N",
                title=None,
                sort="-x",
            ),
            color=alt.Color(
                "status:N",
                legend=None,
                scale=alt.Scale(
                    domain=list(TIMING_LABELS),
                    range=[
                        "#2ca02c",
                        "#5470c6",
                        "#ff9f40",
                        "#d62728",
                        "#9467bd",
                    ],
                ),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="复盘分类"),
                alt.Tooltip("count:Q", title="数量"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")


def _render_contributor_table(data: pd.DataFrame) -> None:
    display_columns = [
        "buy_trade_id",
        "sell_trade_id",
        "quantity",
        "buy_price",
        "sell_price",
        "realized_pnl",
        "return_on_cost",
    ]
    st.dataframe(
        data.loc[:, display_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "buy_trade_id": "买入编号",
            "sell_trade_id": "卖出编号",
            "quantity": "匹配数量",
            "buy_price": st.column_config.NumberColumn(
                "买入价",
                format="%.4f",
            ),
            "sell_price": st.column_config.NumberColumn(
                "卖出价",
                format="%.4f",
            ),
            "realized_pnl": st.column_config.NumberColumn(
                "已实现盈亏",
                format="¥%.2f",
            ),
            "return_on_cost": st.column_config.NumberColumn(
                "成本收益率",
                format="percent",
            ),
        },
    )


def render_trade_review_page() -> None:
    """渲染个人真实交易复盘页面。"""
    st.markdown('<div class="ql-eyebrow">改进</div>', unsafe_allow_html=True)
    st.title("交易复盘")
    st.caption(
        "用 FIFO 计算真实盈亏，并将每笔实际成交与双均线信号的"
        "预期次日成交日进行对照。"
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
            "还没有可用于复盘的资产，请先前往“添加资产数据”。",
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

    observation_count = len(market_data)
    if observation_count < 3:
        st.error("交易复盘至少需要 3 个交易日的行情")
        return

    uploaded_trade = st.file_uploader(
        "上传个人交易记录",
        type=["csv"],
        help=(
            "必需列：trade_id、trade_datetime、symbol、side、"
            "quantity、price、fee；可增加 note、strategy_tag、account。"
        ),
    )
    st.caption(
        "交易文件只在本机用于本次分析。若文件包含 note 列，"
        "复盘明细会同时展示当时记录的交易理由。"
    )

    default_short = 20 if observation_count > 60 else 1
    default_long = 60 if observation_count > 60 else 2
    with st.form("trade_review_form"):
        st.subheader("复盘参数")
        columns = st.columns(4)
        short_window = columns[0].number_input(
            "短期均线（交易日）",
            min_value=1,
            max_value=max(observation_count - 1, 1),
            value=default_short,
            step=1,
        )
        long_window = columns[1].number_input(
            "长期均线（交易日）",
            min_value=2,
            max_value=observation_count,
            value=default_long,
            step=1,
        )
        initial_cash = columns[2].number_input(
            "期初可用资金（元）",
            min_value=1.0,
            value=100_000.0,
            step=10_000.0,
        )
        tolerance = columns[3].number_input(
            "信号匹配容差（交易日）",
            min_value=0,
            value=3,
            step=1,
            help=(
                "实际成交与模型预期成交日相差不超过该交易日数，"
                "视为可匹配。"
            ),
        )
        submitted = st.form_submit_button(
            "生成交易复盘",
            type="primary",
            width="stretch",
        )

    if not submitted:
        st.info("上传个人交易 CSV、确认参数后，点击“生成交易复盘”。")
        return

    if uploaded_trade is None:
        st.error("请先上传个人交易记录 CSV")
        return

    if int(short_window) >= int(long_window):
        st.error("短期均线周期必须小于长期均线周期")
        return

    try:
        trade_data = _load_cached_trade_upload(
            uploaded_trade.name,
            uploaded_trade.getvalue(),
        )
        with st.spinner("正在匹配成交、计算盈亏并生成复盘……"):
            view = _run_cached_review(
                market_data,
                trade_data,
                int(short_window),
                int(long_window),
                float(initial_cash),
                int(tolerance),
            )
    except Exception as error:
        st.error(f"交易复盘无法完成：{error}")
        st.info(
            "请确认交易代码与所选 ETF 一致、成交日具有对应行情，"
            "且买入资金充足、卖出数量不超过持仓。"
        )
        return

    analysis = view.review.analysis
    st.caption(
        f"ETF {view.symbol} · {observation_count} 个交易日 · "
        f"{len(trade_data)} 条真实成交"
    )
    with st.expander("高级信息：本次复盘数据"):
        st.write(f"数据版本：{record.version_id}")

    st.subheader("真实交易结果")
    result_columns = st.columns(7)
    result_columns[0].metric(
        "累计收益率",
        _format_percent(view.cumulative_return),
    )
    result_columns[1].metric(
        "期末总资产",
        f"¥{view.final_equity:,.2f}",
    )
    result_columns[2].metric(
        "已实现盈亏",
        f"¥{analysis.pnl.realized_pnl:,.2f}",
    )
    result_columns[3].metric(
        "未实现盈亏",
        f"¥{analysis.pnl.unrealized_pnl:,.2f}",
    )
    result_columns[4].metric(
        "胜率",
        _format_optional_percent(view.win_rate),
    )
    result_columns[5].metric(
        "盈亏比",
        _format_optional_ratio(view.profit_loss_ratio),
    )
    result_columns[6].metric(
        "累计手续费",
        f"¥{view.review.total_fees:,.2f}",
    )

    st.subheader("个人资金曲线")
    st.line_chart(
        analysis.equity_curve.set_index("date")[["net_value"]],
        x_label="日期",
        y_label="净值（期初资金 = 1）",
        width="stretch",
    )

    st.subheader("真实成交与模型信号")
    timing_columns = st.columns(5)
    for column, status in zip(
        timing_columns,
        TIMING_LABELS,
        strict=True,
    ):
        column.metric(
            TIMING_LABELS[status],
            view.review.timing_counts[status],
        )
    _render_timing_chart(view.timing_summary)

    alignment = view.review.trade_alignment.copy()
    alignment["timing_status"] = alignment["timing_status"].map(
        TIMING_LABELS
    )
    st.dataframe(
        alignment,
        hide_index=True,
        width="stretch",
        column_config={
            "trade_id": "交易编号",
            "trade_datetime": "实际成交时间",
            "side": "方向",
            "quantity": "数量",
            "price": st.column_config.NumberColumn(
                "成交价",
                format="%.4f",
            ),
            "fee": st.column_config.NumberColumn(
                "手续费",
                format="¥%.2f",
            ),
            "signal_id": "模型信号编号",
            "signal_date": "信号日期",
            "expected_execution_date": "模型预期成交日",
            "offset_trading_days": "相差交易日",
            "timing_status": "复盘分类",
            "note": "交易理由",
            "strategy_tag": "策略标签",
            "account": "账户",
        },
    )

    if not view.review.missed_signals.empty:
        with st.expander("查看未执行的模型信号"):
            st.dataframe(
                view.review.missed_signals,
                hide_index=True,
                width="stretch",
            )

    st.subheader("费用与交易频率")
    fee_columns = st.columns(4)
    fee_columns[0].metric("真实成交笔数", len(trade_data))
    fee_columns[1].metric(
        "成交金额合计",
        f"¥{view.review.turnover:,.2f}",
    )
    fee_columns[2].metric(
        "手续费 / 成交金额",
        _format_percent(view.review.fee_to_turnover_rate),
    )
    fee_columns[3].metric(
        "手续费 / 期初资金",
        _format_percent(view.review.fee_to_initial_cash_rate),
    )

    st.subheader("主要盈亏贡献")
    winner_column, loser_column = st.columns(2)
    with winner_column:
        st.markdown("**主要盈利交易**")
        if view.top_winners.empty:
            st.info("所选区间内没有已实现盈利交易。")
        else:
            _render_contributor_table(view.top_winners)
    with loser_column:
        st.markdown("**主要亏损交易**")
        if view.top_losers.empty:
            st.info("所选区间内没有已实现亏损交易。")
        else:
            _render_contributor_table(view.top_losers)

    st.subheader("本次复盘提示")
    for recommendation in view.recommendations:
        st.markdown(f"- {recommendation}")

    with st.expander("查看持仓、FIFO 匹配和原始成交明细"):
        st.markdown("**当前未平仓批次**")
        if analysis.pnl.open_positions.empty:
            st.info("当前没有未平仓持仓。")
        else:
            st.dataframe(
                analysis.pnl.open_positions,
                hide_index=True,
                width="stretch",
            )
        st.markdown("**FIFO 已平仓匹配**")
        if analysis.fifo.matches.empty:
            st.info("当前没有已平仓 FIFO 匹配。")
        else:
            st.dataframe(
                analysis.fifo.matches,
                hide_index=True,
                width="stretch",
            )
        st.markdown("**原始真实成交**")
        st.dataframe(
            analysis.trades,
            hide_index=True,
            width="stretch",
        )

    st.caption(
        "模型信号对齐用于复盘，不代表真实交易必须服从双均线；"
        "提前、延迟或模型外交易需要结合当时的信息和交易备注判断。"
    )


require_access()
render_trade_review_page()
