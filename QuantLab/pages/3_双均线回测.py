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
    build_moving_average_backtest_view,
    get_market_storage_dirs,
    list_market_datasets,
    load_market_dataset,
    require_access,
)
from quantlab.ui.run_records import render_run_record  # noqa: E402


RAW_MARKET_DIR, PROCESSED_MARKET_DIR = get_market_storage_dirs(PROJECT_ROOT)


@st.cache_data(show_spinner=False)
def _load_cached_market_dataset(
    processed_path: str,
    modified_time_ns: int,
) -> pd.DataFrame:
    """文件未变化时复用标准化行情。"""
    del modified_time_ns
    return load_market_dataset(processed_path)


def _run_backtest(
    market_data: pd.DataFrame,
    short_window: int,
    long_window: int,
    initial_cash: float,
    commission_rate: float,
    minimum_commission: float,
    slippage_rate: float,
    sell_tax_rate: float,
    lot_size: int,
    annual_risk_free_rate: float,
    data_source: str,
    data_version: str,
):
    """每次主动运行生成新的实验身份，避免复用旧运行时间和代码版本。"""
    return build_moving_average_backtest_view(
        market_data,
        short_window=short_window,
        long_window=long_window,
        initial_cash=initial_cash,
        commission_rate=commission_rate,
        minimum_commission=minimum_commission,
        slippage_rate=slippage_rate,
        sell_tax_rate=sell_tax_rate,
        lot_size=lot_size,
        annual_risk_free_rate=annual_risk_free_rate,
        data_source=data_source,
        data_version=data_version,
    )


def _format_percent(value: float) -> str:
    return f"{value:.2%}"


def _render_price_chart(
    chart_data: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> None:
    short_column = f"sma_{short_window}"
    long_column = f"sma_{long_window}"
    line_data = chart_data[
        ["date", "close", short_column, long_column]
    ].melt(
        id_vars="date",
        var_name="series",
        value_name="price",
    )
    line_labels = {
        "close": "收盘价",
        short_column: f"短期均线（{short_window}日）",
        long_column: f"长期均线（{long_window}日）",
    }
    line_data["series"] = line_data["series"].map(line_labels)

    lines = (
        alt.Chart(line_data)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="日期"),
            y=alt.Y(
                "price:Q",
                title="价格",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(
                    domain=list(line_labels.values()),
                    range=["#5470c6", "#2ca02c", "#d62728"],
                ),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="日期"),
                alt.Tooltip("series:N", title="序列"),
                alt.Tooltip("price:Q", title="数值", format=".4f"),
            ],
        )
    )

    signals = chart_data[chart_data["signal"].notna()].copy()
    signal_points = (
        alt.Chart(signals)
        .mark_point(filled=True, size=110)
        .encode(
            x=alt.X("date:T", title="日期"),
            y=alt.Y("close:Q", title="价格"),
            color=alt.Color(
                "signal:N",
                title="信号",
                scale=alt.Scale(
                    domain=["BUY", "SELL"],
                    range=["#1b9e77", "#d95f02"],
                ),
            ),
            shape=alt.Shape(
                "signal:N",
                title="信号",
                scale=alt.Scale(
                    domain=["BUY", "SELL"],
                    range=["triangle-up", "triangle-down"],
                ),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="信号日期"),
                alt.Tooltip("signal:N", title="信号"),
                alt.Tooltip("close:Q", title="收盘价", format=".4f"),
            ],
        )
    )

    st.altair_chart(
        (lines + signal_points).properties(height=430),
        width="stretch",
    )


def render_backtest_page() -> None:
    """渲染双均线策略与回测页面。"""
    st.markdown('<div class="ql-eyebrow">实验</div>', unsafe_allow_html=True)
    st.title("双均线策略实验")
    st.caption(
        "短期均线上穿长期均线产生买入信号，下穿产生卖出信号；"
        "信号在收盘后确认，并于下一交易日开盘模拟成交。"
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
            "还没有可用于实验的资产，请先前往“添加资产数据”。",
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
        st.error("双均线回测至少需要 3 个交易日的行情")
        return

    default_short = 20 if observation_count > 60 else 1
    default_long = 60 if observation_count > 60 else 2

    with st.form("moving_average_backtest_form"):
        st.subheader("策略与成本参数")
        strategy_columns = st.columns(4)
        short_window = strategy_columns[0].number_input(
            "短期均线（交易日）",
            min_value=1,
            max_value=max(observation_count - 1, 1),
            value=default_short,
            step=1,
        )
        long_window = strategy_columns[1].number_input(
            "长期均线（交易日）",
            min_value=2,
            max_value=observation_count,
            value=default_long,
            step=1,
        )
        initial_cash = strategy_columns[2].number_input(
            "初始资金（元）",
            min_value=1.0,
            value=100_000.0,
            step=10_000.0,
        )
        lot_size = strategy_columns[3].number_input(
            "每手数量",
            min_value=1,
            value=100,
            step=1,
        )

        cost_columns = st.columns(5)
        commission_percent = cost_columns[0].number_input(
            "手续费率（%）",
            min_value=0.0,
            value=0.03,
            step=0.01,
            format="%.4f",
        )
        minimum_commission = cost_columns[1].number_input(
            "最低手续费（元）",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )
        slippage_percent = cost_columns[2].number_input(
            "单边滑点（%）",
            min_value=0.0,
            value=0.01,
            step=0.01,
            format="%.4f",
        )
        sell_tax_percent = cost_columns[3].number_input(
            "卖出税率（%）",
            min_value=0.0,
            value=0.0,
            step=0.01,
            format="%.4f",
        )
        risk_free_rate_percent = cost_columns[4].number_input(
            "年化无风险利率（%）",
            min_value=-99.0,
            max_value=100.0,
            value=0.0,
            step=0.1,
        )
        submitted = st.form_submit_button(
            "运行回测",
            type="primary",
            width="stretch",
        )

    if not submitted:
        st.info("设置参数后点击“运行回测”查看结果。")
        return

    if int(short_window) >= int(long_window):
        st.error("短期均线周期必须小于长期均线周期")
        return

    try:
        with st.spinner("正在生成信号并模拟成交……"):
            view = _run_backtest(
                market_data,
                int(short_window),
                int(long_window),
                float(initial_cash),
                float(commission_percent) / 100,
                float(minimum_commission),
                float(slippage_percent) / 100,
                float(sell_tax_percent) / 100,
                int(lot_size),
                float(risk_free_rate_percent) / 100,
                record.processed_path,
                record.version_id,
            )
    except Exception as error:
        st.error(f"回测无法完成：{error}")
        st.info(
            "如果资金曲线完全不变，通常表示所选参数未产生可成交信号；"
            "可尝试缩短均线周期或换用更长的行情区间。"
        )
        return

    st.caption(f"ETF {view.symbol} · {observation_count} 个交易日")
    render_run_record(
        view.result.backtest, view.result.performance,
        RAW_MARKET_DIR.parent.parent / "backtests",
    )

    st.subheader("回测结果")
    metrics = st.columns(6)
    metrics[0].metric(
        "累计收益率",
        _format_percent(view.result.performance.cumulative_return),
    )
    metrics[1].metric(
        "年化收益率",
        _format_percent(view.result.performance.annualized_return),
    )
    metrics[2].metric(
        "最大回撤",
        _format_percent(view.result.performance.max_drawdown),
    )
    metrics[3].metric(
        "夏普比率",
        f"{view.result.performance.sharpe_ratio:.2f}",
    )
    metrics[4].metric(
        "期末总资产",
        f"¥{view.final_equity:,.2f}",
    )
    metrics[5].metric(
        "累计交易成本",
        f"¥{view.total_costs:,.2f}",
    )

    st.subheader("行情、均线与信号")
    _render_price_chart(
        view.chart_data,
        view.short_window,
        view.long_window,
    )

    st.subheader("策略资金曲线")
    st.line_chart(
        view.chart_data.set_index("date")[["net_value"]],
        x_label="日期",
        y_label="净值（初始资金 = 1）",
        width="stretch",
    )

    status_columns = st.columns(3)
    status_columns[0].metric("信号数量", view.signal_count)
    status_columns[1].metric("成交笔数", view.filled_trade_count)
    status_columns[2].metric(
        "未成交订单",
        int(
            (
                view.result.backtest.orders["status"] != "FILLED"
            ).sum()
        ),
    )

    st.subheader("成交记录")
    trades = view.result.backtest.trades.copy()
    if trades.empty:
        st.info("本次回测没有成交记录。")
    else:
        st.dataframe(
            trades,
            hide_index=True,
            width="stretch",
            column_config={
                "trade_id": "成交编号",
                "signal_date": "信号日期",
                "execution_date": "成交日期",
                "side": "方向",
                "quantity": "数量",
                "open_price": st.column_config.NumberColumn(
                    "次日开盘价",
                    format="%.4f",
                ),
                "execution_price": st.column_config.NumberColumn(
                    "含滑点成交价",
                    format="%.4f",
                ),
                "notional": st.column_config.NumberColumn(
                    "成交金额",
                    format="¥%.2f",
                ),
                "commission": st.column_config.NumberColumn(
                    "手续费",
                    format="¥%.2f",
                ),
                "tax": st.column_config.NumberColumn(
                    "税费",
                    format="¥%.2f",
                ),
            },
        )

    with st.expander("查看订单审计记录和规则说明"):
        st.dataframe(
            view.result.backtest.orders,
            hide_index=True,
            width="stretch",
        )
        st.markdown(
            """
- `FILLED`：信号已在下一交易日开盘成交。
- `REJECTED`：因已有持仓、当前空仓或资金不足而未成交。
- `EXPIRED`：最后一个交易日的信号没有下一交易日可供成交。

回测采用单只 ETF、只做多、全仓买入和全部卖出。期末未强制平仓，
持仓按最后收盘价计入总资产。
            """
        )


require_access()
render_backtest_page()
