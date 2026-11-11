"""组合回测页面组件：输入、会话状态与展示，不重新实现交易或指标公式。"""

from dataclasses import asdict

import streamlit as st

from quantlab.config.settings import AnalyticsConfig, BacktestConfig
from quantlab.portfolio import (
    PortfolioConfig, analyze_portfolio_result, run_factor_portfolio, run_equal_weight_benchmark,
)
from quantlab.ui.portfolio_display import (
    build_portfolio_chart, format_metric, portfolio_curve_table,
    portfolio_detail_table, portfolio_error_message,
    build_portfolio_comparison_chart, portfolio_comparison_table,
    portfolio_comparison_metrics, cumulative_return_difference_pp,
    safe_portfolio_error,
)
from quantlab.ui.portfolio_records import render_portfolio_save
from quantlab.ui.portfolio_yearly import render_portfolio_yearly
from quantlab.ui.portfolio_concentration import render_portfolio_concentration
from quantlab.ui.portfolio_regimes import render_portfolio_regimes
from quantlab.ui.portfolio_sensitivity import clear_portfolio_sensitivity, render_portfolio_sensitivity


DIRECTIONS = {"descending": "从高到低", "ascending": "从低到高"}
METRICS = (
    ("cumulative_return", "累计收益率"), ("annualized_return", "年化收益率"),
    ("annualized_volatility", "年化波动率"), ("max_drawdown", "最大回撤"),
    ("sharpe_ratio", "夏普比率"),
)


def clear_portfolio_result():
    """只移除运行结果；不影响原来的因子观察与用户已填写的参数。"""
    for key in ("portfolio_analysis", "portfolio_result_identity", "portfolio_benchmark", "portfolio_benchmark_error",
                "portfolio_benchmark_failure"):
        st.session_state.pop(key, None)


def _parameters(preview, definition, identity):
    previous = st.session_state.get("portfolio_input_identity")
    if previous != identity:
        clear_portfolio_result()
        # 因子切换时恢复其默认方向；只换集合仍可保留用户的方向选择。
        if previous is None or previous[-2:] != identity[-2:]:
            st.session_state.pop("portfolio_direction", None)
        st.session_state["portfolio_input_identity"] = identity
    count = len(preview.sources)
    if not 1 <= st.session_state.get("portfolio_top_n", min(3, count)) <= count:
        st.session_state.pop("portfolio_top_n", None)

    left, right = st.columns(2)
    top_n = left.number_input(
        "组合选择数量 N", min_value=1, max_value=count, value=min(3, count), step=1,
        key="portfolio_top_n", help="选中 N 只后设定等权目标；实际权重受费用、整手份额和余留现金影响。",
    )
    direction = right.selectbox(
        "组合排名方向", list(DIRECTIONS), format_func=DIRECTIONS.__getitem__,
        index=int(definition.direction == "higher_is_more_volatile"), key="portfolio_direction",
        help="动量、均线偏离、价格最大回撤、成交量变化和趋势效率默认从高到低；回撤越接近 0 越靠前，趋势效率越接近 1 表示向上越连贯。波动率默认从低到高；方向不是收益保证。",
    )
    left, right = st.columns(2)
    interval = left.number_input(
        "调仓间隔（期）", min_value=1, value=20, step=1, key="portfolio_rebalance_interval",
        help="按完整研究日期轴的位置计数，不是自然日。首个信号在因子热身完成时产生。",
    )
    cash = right.number_input(
        "初始资金（元）", min_value=0.01, value=100000.0, step=10000.0,
        format="%.2f", key="portfolio_initial_cash",
    )
    with st.expander("高级交易设置"):
        left, right = st.columns(2)
        commission = left.number_input(
            "佣金率（%）", min_value=0.0, max_value=100.0, value=.03, step=.01,
            format="%.4f", key="portfolio_commission_pct", help="例如 0.03 表示万分之三；按每笔实际成交计算。",
        )
        minimum = right.number_input(
            "每笔最低佣金（元）", min_value=0.0, value=0.0, step=1.0,
            key="portfolio_minimum_commission",
        )
        left, right = st.columns(2)
        slippage = left.number_input(
            "滑点率（%）", min_value=0.0, max_value=99.99, value=.01, step=.01,
            format="%.4f", key="portfolio_slippage_pct",
            help="买入价上浮、卖出价下浮；已计入成交价格，不再作为费用重复扣除。",
        )
        sell_tax = right.number_input(
            "卖出税率（%）", min_value=0.0, max_value=100.0, value=0.0, step=.01,
            format="%.4f", key="portfolio_sell_tax_pct", help="这是可配置的模拟假设，默认 0，不是税率查询。",
        )
        lot_size = st.number_input(
            "每手份额", min_value=1, value=100, step=1, key="portfolio_lot_size",
        )
    return PortfolioConfig(
        top_n=top_n, rebalance_interval=interval, ascending=direction == "ascending",
        backtest=BacktestConfig(
            initial_cash=cash, commission_rate=commission / 100, minimum_commission=minimum,
            slippage_rate=slippage / 100, sell_tax_rate=sell_tax / 100, lot_size=lot_size,
        ),
    )


def _show_benchmark(analysis, benchmark, *, status, message, historical, namespace):
    """基准与策略独立失败；不可用时不展示差值、部分曲线或上次基准。"""
    st.subheader("策略与基准：净值对照")
    st.caption(
        "全池等权买入持有：与策略使用相同初始资金、交易成本、整手设置和完整日期轴。"
        "在所选因子首个计划执行日一次建仓，之后不再调仓；等权是初始目标，不是每日实际权重。"
    )
    if benchmark is None:
        if status == "not_run":
            st.info("这份实验未运行基准；下方仅展示组合策略结果。")
        else:
            st.warning("基准不可用；下方仅展示组合策略结果。" + message)
        return
    entry = benchmark.backtest.metadata["strategy"]["planned_execution_date"]
    if entry is None:
        st.info("区间未到计划建仓日（L+1），基准保持现金；没有借用更早数据或顺延建仓。")
    else:
        st.caption(f"基准计划建仓日：{entry}。此前持有现金，不跟随策略第一次实际成交日。")
        if benchmark.backtest.trades.empty:
            if benchmark.backtest.orders.empty:
                st.info("基准在计划建仓日按全池等权预算分配后，各 ETF 的目标份额均不足一手，未生成买入委托；此后保持现金，不顺延重试。")
            else:
                st.info("基准在计划建仓日未发生成交，此后保持现金，不顺延重试；原因见基准委托明细。")
    difference = cumulative_return_difference_pp(analysis, benchmark)
    st.metric("累计收益差（策略 − 基准）", "—" if difference is None else f"{difference:+.2f} 个百分点")
    st.caption("差值 = 策略累计收益率 − 基准累计收益率；不是收益率相除，也不是市场指数超额收益或风险调整 alpha。")
    st.altair_chart(build_portfolio_comparison_chart(portfolio_comparison_table(analysis, benchmark)), width="stretch",
                   key=f"{namespace}_comparison_chart")
    st.dataframe(portfolio_comparison_metrics(analysis, benchmark), hide_index=True, width="stretch",
                 key=f"{namespace}_comparison_metrics")
    for field, label in METRICS:
        if field in benchmark.performance.unavailable_reasons:
            st.caption(f"基准{label} — {benchmark.performance.unavailable_reasons[field]}。")
    with st.expander("基准资金、持仓、委托与成交明细"):
        for title, table in (("资金曲线", benchmark.backtest.equity_curve),
                             ("每日持仓", benchmark.backtest.positions),
                             ("委托明细", benchmark.backtest.orders),
                             ("成交明细", benchmark.backtest.trades)):
            st.caption(title)
            st.dataframe(portfolio_detail_table(table), hide_index=True, width="stretch", key=f"{namespace}_benchmark_{title}")
        st.caption("以下为存档中基准的来源、规则和原运行身份。" if historical else "以下为本次基准的来源、规则和运行身份；存档需主动点击“保存实验”。")
        st.json(benchmark.backtest.metadata, expanded=False)


def render_portfolio_result(analysis, *, benchmark=None, benchmark_status="unavailable", benchmark_message="",
                            historical=False, namespace="portfolio_live"):
    """显式展示一份分析及其自己的基准，不从会话状态借用另一份结果。"""
    result, performance = analysis.backtest, analysis.performance
    curve = portfolio_curve_table(analysis)
    st.subheader("历史实验结果" if historical else "组合回测结果")
    st.caption(
        f"{curve.date.iloc[0]:%Y-%m-%d} 至 {curve.date.iloc[-1]:%Y-%m-%d} · "
        f"{len(curve)} 个日期位置（含热身和空仓期） · "
        f"{len(result.metadata['data']['symbols'])} 只候选 ETF"
    )
    cells = st.columns(3) + st.columns(3)
    cells[0].metric("期末总资产（元）", f"{curve.equity.iloc[-1]:,.2f}")
    for cell, (field, label) in zip(cells[1:], METRICS):
        cell.metric(label, format_metric(getattr(performance, field), percent=field != "sharpe_ratio"))
    analytics = result.metadata["analytics_config"]
    st.caption(f"指标使用完整资金曲线；年化按每年 {analytics['periods_per_year']} 期，"
               f"无风险年利率为 {analytics['annual_risk_free_rate']:.2%}。最大回撤以负数表示下跌幅度。")
    for field, label in METRICS:
        if field in performance.unavailable_reasons:
            st.caption(f"{label} — {performance.unavailable_reasons[field]}。")
    if result.trades.empty:
        st.info("本次未发生成交，资金保持现金。请查看调仓决策和委托明细，区分历史不足、有效数量不足或资金不足等情况。")
    skipped = int(result.decisions.status.eq("SKIPPED_INSUFFICIENT_FACTORS").sum())
    if skipped:
        st.info(f"有 {skipped} 次调仓因有效因子不足 N 只而跳过，保留原持仓；未补入更早的因子值。")
    if result.decisions.cutoff_tie.any():
        st.caption("部分筛选边界存在同值，按 ETF 代码升序截取到 N 只；入选不表示更优。")
    _show_benchmark(analysis, benchmark, status=benchmark_status, message=benchmark_message,
                    historical=historical, namespace=namespace)
    render_portfolio_yearly(analysis, benchmark=benchmark, benchmark_status=benchmark_status,
                            historical=historical, namespace=namespace)
    render_portfolio_concentration(analysis, historical=historical, namespace=namespace)
    render_portfolio_regimes(analysis, benchmark=benchmark, benchmark_status=benchmark_status,
                             historical=historical, namespace=namespace)
    st.subheader("资金曲线")
    st.altair_chart(build_portfolio_chart(curve, measure="equity"), width="stretch", key=f"{namespace}_equity_chart")
    st.subheader("回撤曲线")
    st.altair_chart(build_portfolio_chart(curve, measure="drawdown"), width="stretch", key=f"{namespace}_drawdown_chart")
    last = result.equity_curve.iloc[-1]
    st.caption(
        f"期末现金 {last.cash:,.2f} 元 · 持仓市值 {last.position_value:,.2f} 元 · "
        f"累计佣金及税费 {last.cumulative_fees_taxes:,.2f} 元 · "
        f"累计滑点影响 {last.cumulative_slippage:,.2f} 元（已体现在成交价）"
    )
    for title, table in (
        ("资金曲线逐日明细", result.equity_curve), ("每日实际持仓", result.positions),
        ("调仓决策（含跳过和过期）", result.decisions), ("每次调仓的等权目标", result.targets),
        ("委托明细（含未成交部分）", result.orders), ("实际成交明细", result.trades),
    ):
        with st.expander(title):
            if table.empty:
                st.info("本次没有对应记录。")
            else:
                st.dataframe(portfolio_detail_table(table), hide_index=True, width="stretch", key=f"{namespace}_{title}")
    with st.expander("本次参数、数据来源与研究限制"):
        st.caption("此处为档案中保存的原始参数、数据来源和研究限制；未重新计算或修改历史。" if historical else
                   "本次结果不会自动保存；请主动点击“保存实验”。输入是已保存快照，不重新读取原行情文件。")
        for warning in result.metadata["warnings"]:
            st.warning(warning)
        st.json(result.metadata, expanded=False)


def render_portfolio(preview, definition, identity, *, archive_dir=None):
    """结果与完整研究输入/策略参数绑定，查看日期或 ETF 不参与运行身份。"""
    st.subheader("单因子 Top N 等权组合")
    st.caption(
        "收盘后产生信号，在下一研究日期开盘模拟成交；先卖后买，不借款、不做空。"
        "这里只做历史模拟，不自动下单。"
    )
    st.caption(
        f"完整快照：{preview.start} 至 {preview.end} · {len(preview.sources)} 只 ETF。"
        "“因子观察”中的查看日期、查看 ETF 和配置预览，不会截取这里的回测区间。"
    )
    if preview.adjustment == "none":
        st.caption("当前快照为未复权价格，分红和拆并份额可能影响模拟结果；不是总回报口径。")
    else:
        st.caption("当前快照使用复权价格作为模拟价格；结果并非对真实历史成交金额的还原，也不单独计入分红现金。")
    try:
        config = _parameters(preview, definition, identity)
    except (ValueError, OverflowError) as error:
        clear_portfolio_result()
        clear_portfolio_sensitivity()
        st.error(portfolio_error_message(error))
        return
    analytics_config = AnalyticsConfig()
    result_identity = (identity, asdict(config), asdict(analytics_config))
    if st.session_state.get("portfolio_result_identity") != result_identity:
        had_result = "portfolio_analysis" in st.session_state
        clear_portfolio_result()
        if had_result:
            st.info("回测参数已变化，旧结果已清除。点击“运行组合回测”以生成新结果。")
    if st.button("运行组合回测", key="portfolio_run", type="primary"):
        clear_portfolio_result()
        try:
            with st.spinner("正在验证完整快照、模拟调仓并计算指标……"):
                result = run_factor_portfolio(preview, factor_name=definition.name, config=config)
                analysis = analyze_portfolio_result(result, analytics_config=analytics_config)
            # 两步均成功才发布结果；失败时不留下半次运行。
            st.session_state["portfolio_analysis"] = analysis
            st.session_state["portfolio_result_identity"] = result_identity
            try:
                with st.spinner("正在运行全池等权买入持有基准……"):
                    benchmark = run_equal_weight_benchmark(
                        preview, factor_name=definition.name, backtest_config=config.backtest,
                        analytics_config=analytics_config,
                    )
                    # 严格核对可比性后才发布；不做日期交集或填补缺失。
                    portfolio_comparison_table(analysis, benchmark)
                    cumulative_return_difference_pp(analysis, benchmark)
                st.session_state["portfolio_benchmark"] = benchmark
            except Exception as error:
                st.session_state["portfolio_benchmark_error"] = portfolio_error_message(error)
                st.session_state["portfolio_benchmark_failure"] = safe_portfolio_error(error)
        except Exception as error:
            clear_portfolio_result()
            st.error(portfolio_error_message(error))
    render_portfolio_sensitivity(preview, definition, identity, config=config, analytics_config=analytics_config)
    analysis = st.session_state.get("portfolio_analysis")
    if analysis is None:
        st.caption("设置参数后点击运行；不会随参数调整自动计算，也不会自动保存。")
    else:
        benchmark = st.session_state.get("portfolio_benchmark")
        if archive_dir is not None:
            render_portfolio_save(analysis, archive_dir, benchmark=benchmark,
                                  benchmark_error=st.session_state.get("portfolio_benchmark_failure"))
        render_portfolio_result(
            analysis, benchmark=benchmark, benchmark_status="complete" if benchmark else "unavailable",
            benchmark_message=st.session_state.get("portfolio_benchmark_error", ""),
        )
