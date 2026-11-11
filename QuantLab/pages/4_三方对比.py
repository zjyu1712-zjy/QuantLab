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
    APPROACH_LABELS,
    build_three_way_comparison_view,
    get_market_storage_dirs,
    list_market_datasets,
    load_market_dataset,
    load_uploaded_trade_csv,
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


@st.cache_data(show_spinner=False)
def _load_cached_trade_upload(
    filename: str,
    content: bytes,
) -> pd.DataFrame:
    """上传内容未变化时复用交易记录转换与校验结果。"""
    return load_uploaded_trade_csv(filename, content)


def _run_comparison(
    market_data: pd.DataFrame,
    trade_data: pd.DataFrame,
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
    """每次比较都记录两个模拟回测的实际运行身份。"""
    return build_three_way_comparison_view(
        market_data,
        trade_data,
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


def _render_equity_chart(curves: pd.DataFrame) -> None:
    chart = (
        alt.Chart(curves)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("date:T", title="日期"),
            y=alt.Y(
                "net_value:Q",
                title="净值（初始资金 = 1）",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "approach:N",
                title=None,
                scale=alt.Scale(
                    domain=[
                        "买入持有",
                        "双均线策略",
                        "个人真实交易",
                    ],
                    range=["#5470c6", "#2ca02c", "#d95f02"],
                ),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="日期"),
                alt.Tooltip("approach:N", title="投资路径"),
                alt.Tooltip(
                    "net_value:Q",
                    title="净值",
                    format=".4f",
                ),
            ],
        )
        .properties(height=430)
    )
    st.altair_chart(chart, width="stretch")


def _render_summary(view) -> None:
    metrics = view.display_metrics.set_index("approach")
    st.subheader("三种路径的结果")
    columns = st.columns(3)
    for column, approach in zip(
        columns,
        ["买入持有", "双均线策略", "个人真实交易"],
        strict=True,
    ):
        row = metrics.loc[approach]
        with column:
            st.markdown(f"**{approach}**")
            st.metric(
                "累计收益率",
                _format_percent(float(row["cumulative_return"])),
            )
            st.caption(
                f"期末资产 ¥{float(row['final_equity']):,.2f} · "
                f"累计成本 ¥{float(row['total_costs']):,.2f}"
            )

    gaps = view.comparison.return_gaps
    conclusion_columns = st.columns(3)
    conclusion_columns[0].metric(
        "策略相对持有",
        _format_percent(gaps["strategy_minus_buy_and_hold"]),
        help="双均线累计收益率减去买入持有累计收益率",
    )
    conclusion_columns[1].metric(
        "真实交易相对策略",
        _format_percent(gaps["personal_minus_strategy"]),
        help="个人真实交易累计收益率减去双均线累计收益率",
    )
    conclusion_columns[2].metric(
        "真实交易相对持有",
        _format_percent(gaps["personal_minus_buy_and_hold"]),
        help="个人真实交易累计收益率减去买入持有累计收益率",
    )


def _render_metrics_table(metrics: pd.DataFrame) -> None:
    table = metrics.rename(
        columns={
            "approach": "投资路径",
            "final_equity": "期末资产",
            "cumulative_return": "累计收益率",
            "annualized_return": "年化收益率",
            "annualized_volatility": "年化波动率",
            "max_drawdown": "最大回撤",
            "sharpe_ratio": "夏普比率",
            "trade_count": "成交笔数",
            "total_costs": "累计成本",
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "期末资产": st.column_config.NumberColumn(format="¥%.2f"),
            "累计收益率": st.column_config.NumberColumn(format="percent"),
            "年化收益率": st.column_config.NumberColumn(format="percent"),
            "年化波动率": st.column_config.NumberColumn(format="percent"),
            "最大回撤": st.column_config.NumberColumn(format="percent"),
            "夏普比率": st.column_config.NumberColumn(format="%.2f"),
            "累计成本": st.column_config.NumberColumn(format="¥%.2f"),
        },
    )


def _render_review(view) -> None:
    conclusions = view.review.conclusions
    strategy_verb = (
        "跑赢" if conclusions["strategy_beats_buy_and_hold"] else "未跑赢"
    )
    personal_verb = (
        "跑赢" if conclusions["personal_beats_strategy"] else "未跑赢"
    )
    lowest_cost = APPROACH_LABELS[
        str(conclusions["lowest_cost_approach"])
    ]

    st.subheader("结果解读")
    st.markdown(
        f"""
- 双均线策略**{strategy_verb}**买入持有，收益差为
  **{_format_percent(float(conclusions["strategy_return_advantage"]))}**。
- 个人真实交易**{personal_verb}**双均线策略，收益差为
  **{_format_percent(float(conclusions["personal_return_advantage"]))}**。
- 本次累计成本最低的是**{lowest_cost}**。
        """
    )
    st.caption(str(conclusions["interpretation"]))

    attribution = view.review.cost_attribution.rename(
        columns={
            "comparison": "比较项",
            "observed_return_gap": "观察到的收益差",
            "extra_cost_rate": "额外成本率",
            "estimated_cost_effect": "估算成本影响",
            "remaining_gap": "成本外剩余差异",
        }
    )
    attribution["比较项"] = attribution["比较项"].map(
        {
            "strategy_vs_buy_and_hold": "双均线 vs 买入持有",
            "personal_vs_strategy": "真实交易 vs 双均线",
        }
    )
    with st.expander("查看成本近似拆解"):
        st.dataframe(
            attribution,
            hide_index=True,
            width="stretch",
            column_config={
                column: st.column_config.NumberColumn(format="percent")
                for column in [
                    "观察到的收益差",
                    "额外成本率",
                    "估算成本影响",
                    "成本外剩余差异",
                ]
            },
        )

    st.subheader("个人交易的主要盈亏贡献")
    winners, losers = st.columns(2)
    with winners:
        st.markdown("**主要盈利交易**")
        if view.review.top_winners.empty:
            st.info("所选区间内没有已实现盈利交易。")
        else:
            st.dataframe(
                view.review.top_winners,
                hide_index=True,
                width="stretch",
            )
    with losers:
        st.markdown("**主要亏损交易**")
        if view.review.top_losers.empty:
            st.info("所选区间内没有已实现亏损交易。")
        else:
            st.dataframe(
                view.review.top_losers,
                hide_index=True,
                width="stretch",
            )


def render_comparison_page() -> None:
    """渲染买入持有、双均线和个人真实交易的比较页面。"""
    st.markdown('<div class="ql-eyebrow">实验</div>', unsafe_allow_html=True)
    st.title("三种投资方法对比")
    st.caption(
        "在同一只 ETF、同一行情区间和相同初始资金下，比较买入持有、"
        "双均线策略与个人真实交易。"
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
            "还没有可用于比较的资产，请先前往“添加资产数据”。",
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
        st.error("三方对比至少需要 3 个交易日的行情")
        return

    uploaded_trade = st.file_uploader(
        "上传个人交易记录",
        type=["csv"],
        help=(
            "必需列：trade_id、trade_datetime、symbol、side、"
            "quantity、price、fee"
        ),
    )
    st.caption(
        "交易文件仅用于本次分析；个人交易采用文件中的实际成交价、"
        "数量和手续费。"
    )

    default_short = 20 if observation_count > 60 else 1
    default_long = 60 if observation_count > 60 else 2

    with st.form("three_way_comparison_form"):
        st.subheader("统一比较参数")
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
            "统一初始资金（元）",
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
            "模拟手续费率（%）",
            min_value=0.0,
            value=0.03,
            step=0.01,
            format="%.4f",
        )
        minimum_commission = cost_columns[1].number_input(
            "模拟最低手续费（元）",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )
        slippage_percent = cost_columns[2].number_input(
            "模拟单边滑点（%）",
            min_value=0.0,
            value=0.01,
            step=0.01,
            format="%.4f",
        )
        sell_tax_percent = cost_columns[3].number_input(
            "模拟卖出税率（%）",
            min_value=0.0,
            value=0.0,
            step=0.01,
            format="%.4f",
        )
        risk_free_percent = cost_columns[4].number_input(
            "年化无风险利率（%）",
            min_value=-99.0,
            max_value=100.0,
            value=0.0,
            step=0.1,
        )
        submitted = st.form_submit_button(
            "生成三方对比",
            type="primary",
            width="stretch",
        )

    if not submitted:
        st.info("上传个人交易 CSV、确认参数后，点击“生成三方对比”。")
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
        with st.spinner("正在统一口径并生成三方对比……"):
            view = _run_comparison(
                market_data,
                trade_data,
                int(short_window),
                int(long_window),
                float(initial_cash),
                float(commission_percent) / 100,
                float(minimum_commission),
                float(slippage_percent) / 100,
                float(sell_tax_percent) / 100,
                int(lot_size),
                float(risk_free_percent) / 100,
                record.processed_path,
                record.version_id,
            )
    except Exception as error:
        st.error(f"三方对比无法完成：{error}")
        st.info(
            "请确认交易代码与所选 ETF 一致、成交日期位于行情区间内，"
            "并且交易顺序不会出现资金不足或超额卖出。"
        )
        return

    st.caption(
        f"ETF {view.symbol} · {observation_count} 个交易日 · "
        f"{len(trade_data)} 条个人交易"
    )
    with st.expander("高级信息：本次比较数据"):
        st.write(f"数据版本：{record.version_id}")
    for label, analysis in (
        ("买入持有", view.comparison.buy_and_hold),
        ("双均线策略", view.comparison.strategy),
    ):
        render_run_record(
            analysis.backtest, analysis.performance,
            RAW_MARKET_DIR.parent.parent / "backtests", label=label,
        )
    _render_summary(view)

    st.subheader("三方资金曲线")
    _render_equity_chart(view.normalized_curves)

    st.subheader("完整绩效指标")
    _render_metrics_table(view.display_metrics)
    _render_review(view)

    with st.expander("查看统一口径与限制"):
        st.markdown(
            """
- 三种路径使用同一只 ETF、同一段日线行情和同一初始资金。
- 买入持有与双均线使用页面设置的手续费、滑点、卖出税和整手规则。
- 个人真实交易使用 CSV 中记录的成交价格、数量和手续费。
- 收益差不能单独证明差异来自择时或主观操作；成本拆解只是近似估算。
- 第一阶段仅支持单只 ETF、只做多，不包含自动下单。
            """
        )


require_access()
render_comparison_page()
