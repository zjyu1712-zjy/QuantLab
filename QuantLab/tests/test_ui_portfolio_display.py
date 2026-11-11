"""组合展示只转译呈现，不缩短研究轴、不改写数值、不泄漏错误细节。"""

from copy import deepcopy
from dataclasses import replace
from datetime import date
import importlib
import json

import numpy as np
import pandas as pd
import pytest

from quantlab.config.settings import BacktestConfig
from quantlab.portfolio import (
    PortfolioBacktestError, PortfolioConfig, analyze_portfolio_result, run_factor_portfolio,
)


def display():
    assert importlib.util.find_spec("quantlab.ui.portfolio_display") is not None, "组合展示适配层尚未实现"
    return importlib.import_module("quantlab.ui.portfolio_display")


def chart_rows(spec):
    data = spec["data"]
    return data["values"] if "values" in data else spec["datasets"][data["name"]]


@pytest.mark.parametrize("value,percent,want", [
    (None, True, "—"), (np.nan, True, "—"), (np.inf, False, "—"),
    (-np.inf, True, "—"), (0.0, True, "0.00%"),
    (0.12345, True, "12.35%"), (-0.125, True, "-12.50%"),
    (1.2345, False, "1.23"),
])
def test_metric_format_distinguishes_missing_from_zero(value, percent, want):
    assert display().format_metric(value, percent=percent) == want


def test_curve_preserves_warmup_first_cost_and_all_dates_without_mutation(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=23)
    result = run_factor_portfolio(preview, factor_name="momentum_20", config=PortfolioConfig(
        top_n=1, backtest=BacktestConfig(lot_size=100, commission_rate=0,
                                        minimum_commission=10, slippage_rate=0)))
    analysis = analyze_portfolio_result(result)
    before = deepcopy(analysis)
    # Row labels are not the date axis and must not affect alignment.
    analysis.backtest.equity_curve.index = pd.Index(range(23, 46))
    frame = display().portfolio_curve_table(analysis)
    assert frame.columns.tolist() == ["date", "equity", "drawdown"]
    assert frame.date.tolist() == pd.date_range("2025-01-02", periods=23, freq="B").tolist()
    assert frame.equity.tolist() == [100000.0] * 21 + [99990.0, 100440.0]
    assert frame.drawdown.tolist() == pytest.approx([0.0] * 21 + [-0.0001, 0.0])
    assert analysis.backtest.equity_curve.index.tolist() == list(range(23, 46))
    frame.loc[:, "equity"] = -10
    frame.loc[:, "drawdown"] = -1
    frame.loc[:, "date"] = pd.Timestamp("2000-01-01")
    pd.testing.assert_frame_equal(analysis.backtest.equity_curve.reset_index(drop=True), before.backtest.equity_curve)
    pd.testing.assert_series_equal(analysis.performance.drawdown, before.performance.drawdown)
    assert analysis.backtest.metadata == before.backtest.metadata


@pytest.mark.parametrize("periods", [1, 10])
def test_cash_and_one_observation_keep_curve_and_undefined_metrics(make_portfolio_preview, periods):
    preview, _ = make_portfolio_preview(periods=periods)
    analysis = analyze_portfolio_result(run_factor_portfolio(
        preview, factor_name="momentum_20", config=PortfolioConfig(top_n=2)))
    ui = display()
    frame = ui.portfolio_curve_table(analysis)
    assert frame.equity.tolist() == [100000.0] * periods
    assert frame.drawdown.tolist() == [0.0] * periods
    assert ui.format_metric(analysis.performance.sharpe_ratio, percent=False) == "—"
    assert ui.format_metric(analysis.performance.max_drawdown) == "0.00%"
    if periods == 1:
        assert ui.format_metric(analysis.performance.cumulative_return) == "—"
        assert ui.build_portfolio_chart(frame, measure="equity").to_dict()["mark"]["point"] is True


@pytest.mark.parametrize("measure,title,number_format", [
    ("equity", "组合总资产（元）", ",.2f"),
    ("drawdown", "回撤（%）", ".2%"),
])
def test_chart_preserves_negative_drawdown_real_dates_and_unit_tooltips(measure, title, number_format):
    frame = pd.DataFrame({"date": pd.to_datetime(["2025-01-02", "2025-01-06"]),
                          "equity": [100000.0, 95000.0], "drawdown": [0.0, -0.05]})
    before = frame.copy(deep=True)
    spec = display().build_portfolio_chart(frame, measure=measure).to_dict()
    assert spec["width"] == "container"
    assert spec["encoding"]["x"]["type"] == "temporal"
    assert "timeUnit" not in spec["encoding"]["x"]
    assert spec["encoding"]["y"]["field"] == measure
    assert spec["encoding"]["y"]["title"] == title
    assert spec["encoding"]["y"]["axis"]["format"] == number_format
    assert spec["encoding"]["tooltip"][1]["format"] == number_format
    assert spec["encoding"]["tooltip"][1]["title"] == title
    assert [row[measure] for row in chart_rows(spec)] == frame[measure].tolist()
    pd.testing.assert_frame_equal(frame, before)


def test_chart_over_5000_rows_serializes_every_observation():
    frame = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=6001),
                          "equity": np.arange(6001, dtype=float) + 100000,
                          "drawdown": np.zeros(6001)})
    spec = json.loads(display().build_portfolio_chart(frame, measure="equity").to_json())
    rows = chart_rows(spec)
    assert len(rows) == 6001
    assert rows[0]["equity"] == 100000.0
    assert rows[-1]["equity"] == 106000.0
    assert "transform" not in spec


@pytest.mark.parametrize("periods", [1, 6001])
def test_chart_date_labels_are_sparse_horizontal_and_include_endpoints(periods):
    frame = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=periods), "equity": 100000.0})
    spec = display().build_portfolio_chart(frame, measure="equity").to_dict()
    axis = spec["encoding"]["x"]["axis"]
    assert axis.get("labelAngle") == 0
    assert axis.get("labelOverlap") == "greedy"
    assert 1 <= len(axis["values"]) <= 3
    assert axis["values"][0] == {"year": 2000, "month": 1, "date": 1}
    last = frame.date.iloc[-1]
    assert axis["values"][-1] == {"year": last.year, "month": last.month, "date": last.day}


def test_chart_nonfinite_values_are_gaps_not_zero_or_removed_dates():
    frame = pd.DataFrame({"date": pd.date_range("2025-01-02", periods=5),
                          "equity": [100.0, np.nan, np.inf, -np.inf, 90.0],
                          "drawdown": [0.0, np.nan, np.inf, -np.inf, -0.1]})
    before = frame.copy(deep=True)
    spec = display().build_portfolio_chart(frame, measure="drawdown").to_dict()
    assert [row["drawdown"] for row in chart_rows(spec)] == [0.0, None, None, None, -0.1]
    assert len(chart_rows(spec)) == 5
    assert spec["mark"]["invalid"] == "break-paths-filter-domains"
    pd.testing.assert_frame_equal(frame, before)


def test_chart_rejects_unknown_measure_instead_of_guessing():
    with pytest.raises(ValueError):
        display().build_portfolio_chart(pd.DataFrame(), measure="return")


def test_detail_tables_translate_all_result_columns_and_leave_source_untouched(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=45)
    result = run_factor_portfolio(preview, factor_name="momentum_20", config=PortfolioConfig(top_n=1))
    for name in ("decisions", "targets", "orders", "trades", "positions", "equity_curve"):
        source = getattr(result, name)
        before = source.copy(deep=True)
        translated = display().portfolio_detail_table(source)
        assert len(translated) == len(source)
        assert len(translated.columns) == len(source.columns)
        assert not set(translated.columns) & set(source.columns)
        for index, column in enumerate(source.columns):
            if column not in {"status", "side", "reason"}:
                pd.testing.assert_series_equal(translated.iloc[:, index], source[column], check_names=False)
        if not translated.empty:
            translated.iloc[0, 0] = translated.iloc[-1, 0]
        pd.testing.assert_frame_equal(source, before)


def test_detail_translates_status_direction_and_structured_reasons_without_faking_values():
    frame = pd.DataFrame({
        "status": ["EXECUTED", "EXPIRED", "PENDING", "SKIPPED_INSUFFICIENT_FACTORS", "FILLED",
                   "PARTIALLY_FILLED", "REJECTED", "valid", "warmup", "missing_window", "numeric_range", "NEW_STATUS"],
        "side": ["BUY", "SELL"] * 6,
        "reason": ["missing_open", "有效因子 1 只，不足 N=2；跳过本次调仓"] * 6,
    })
    before = frame.copy(deep=True)
    table = display().portfolio_detail_table(frame)
    assert table["状态"].tolist() == ["已执行", "已过期", "待执行", "有效因子不足，跳过调仓", "全部成交",
                                     "部分成交", "未成交", "有效", "热身期", "窗口数据缺失", "数值超出范围", "NEW_STATUS"]
    assert table["方向"].tolist() == ["买入", "卖出"] * 6
    assert "开盘价" in table["原因"].iloc[0]
    assert table["原因"].iloc[1] == frame.reason.iloc[1]
    pd.testing.assert_frame_equal(frame, before)


def test_identity_error_explains_restart_after_source_changes():
    error = PortfolioBacktestError("SECRET", reason="invalid_run_identity")
    message = display().portfolio_error_message(error)
    assert "若刚更新代码" in message
    assert "重启 QuantLab" in message
    assert "SECRET" not in message


@pytest.mark.parametrize("reason,keyword", [
    ("invalid_config", "参数"), ("invalid_analytics_config", "指标"),
    ("invalid_research_input", "研究集"), ("invalid_factor", "因子"),
    ("invalid_factor_result", "因子"), ("invalid_run_identity", "身份"),
    ("invalid_portfolio_result", "结果"), ("invalid_market_data", "行情"),
    ("invalid_schedule", "调仓"), ("invalid_execution_input", "交易"),
    ("missing_open", "开盘价"), ("missing_close", "收盘价"),
    ("zero_volume", "成交量"), ("negative_cash", "现金"),
    ("negative_sell_proceeds", "卖出"), ("non_positive_equity", "资产"),
    ("numeric_range", "数值"),
])
def test_known_errors_use_safe_reason_and_validated_context_only(reason, keyword):
    error = PortfolioBacktestError("SECRET /Users/private/project.csv token=secret", reason=reason,
                                   date=date(2025, 1, 6), symbols=("510300", "../../secret", "159915"))
    message = display().portfolio_error_message(error)
    assert keyword in message
    assert "2025-01-06" in message and "510300" in message and "159915" in message
    assert not any(leak in message for leak in ("SECRET", "Users", "secret", "token", "../"))


@pytest.mark.parametrize("error", [
    RuntimeError("SECRET /Users/private/data.csv"),
    PortfolioBacktestError("SECRET", reason="/Users/private/data.csv", symbols=("/private/secret",)),
    PortfolioBacktestError("SECRET", reason="missing_open", date="/Users/secret", symbols="123456"),
    PortfolioBacktestError("SECRET", reason="missing_open", symbols=("５１０３００", "1234567", None)),
    PortfolioBacktestError("SECRET", reason=["/Users/private"]),
])
def test_unexpected_errors_and_malformed_context_never_leak(error):
    message = display().portfolio_error_message(error)
    assert isinstance(message, str) and message
    assert not any(leak in message for leak in ("SECRET", "Users", "private", "secret", "1234567", "５１０３００"))


@pytest.fixture
def comparison_pair(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=3)
    strategy = analyze_portfolio_result(run_factor_portfolio(
        preview, factor_name="momentum_20", config=PortfolioConfig(top_n=1)))
    benchmark = deepcopy(strategy)
    benchmark.backtest.metadata["schema_version"] = "portfolio-benchmark-v1"
    benchmark.backtest.metadata.pop("factor_definition")
    strategy = replace(strategy, performance=replace(
        strategy.performance, net_value=pd.Series([1., 1.1, 1.2], index=strategy.performance.net_value.index),
        cumulative_return=.2, annualized_return=.3, annualized_volatility=.1,
        max_drawdown=-.05, sharpe_ratio=1.5))
    benchmark = replace(benchmark, performance=replace(
        benchmark.performance, net_value=pd.Series([1., .95, 1.05], index=benchmark.performance.net_value.index),
        cumulative_return=.05, annualized_return=None, annualized_volatility=.2,
        max_drawdown=-.1, sharpe_ratio=None))
    strategy.backtest.equity_curve.loc[:, "equity"] = [100000., 110000., 120000.]
    benchmark.backtest.equity_curve.loc[:, "equity"] = [100000., 95000., 105000.]
    return strategy, benchmark


def test_comparison_preserves_exact_dates_values_and_inputs(comparison_pair):
    strategy, benchmark = comparison_pair
    before = deepcopy(comparison_pair)
    ui = display()
    assert hasattr(ui, "portfolio_comparison_table"), "比较展示尚未实现"
    table = ui.portfolio_comparison_table(strategy, benchmark)
    assert table.columns.tolist() == ["date", "strategy_net_value", "benchmark_net_value"]
    assert table.date.tolist() == pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]).tolist()
    assert table.strategy_net_value.tolist() == [1., 1.1, 1.2]
    assert table.benchmark_net_value.tolist() == [1., .95, 1.05]
    table.iloc[0, 1] = 999
    for actual, original in zip(comparison_pair, before):
        pd.testing.assert_series_equal(actual.performance.net_value, original.performance.net_value)
        pd.testing.assert_frame_equal(actual.backtest.equity_curve, original.backtest.equity_curve)
        assert actual.backtest.metadata == original.backtest.metadata


@pytest.mark.parametrize("damage", [
    "preview", "market", "calendar", "cost", "analytics", "anchor", "missing_anchor",
    "series_index", "series_short", "series_reordered", "nan", "inf", "zero", "negative",
    "curve_date", "curve_short", "schema", "status",
])
@pytest.mark.parametrize("helper", ["portfolio_comparison_table", "portfolio_comparison_metrics",
                                    "cumulative_return_difference_pp"])
def test_comparison_rejects_incomparable_inputs_without_repair(comparison_pair, damage, helper):
    strategy, benchmark = comparison_pair
    meta = benchmark.backtest.metadata
    if damage in {"preview", "market"}:
        meta["data"][f"{damage}_sha256"] = "different"
    elif damage == "calendar":
        meta["data"]["calendar"]["mode"] = "different"
    elif damage == "cost":
        meta["strategy"]["config"]["backtest"]["commission_rate"] = .5
    elif damage == "analytics":
        meta["analytics_config"]["periods_per_year"] = 365
    elif damage == "anchor":
        meta["strategy"]["signal_anchor_position"] = 60
    elif damage == "missing_anchor":
        del meta["strategy"]["signal_anchor_position"]
    elif damage in {"nan", "inf", "zero", "negative"}:
        benchmark.performance.net_value.iloc[1] = {"nan": np.nan, "inf": np.inf, "zero": 0., "negative": -1.}[damage]
    elif damage == "series_index":
        benchmark.performance.net_value.index = pd.date_range("2025-02-01", periods=3)
    elif damage == "series_short":
        benchmark = replace(benchmark, performance=replace(benchmark.performance,
            net_value=benchmark.performance.net_value.iloc[:-1]))
    elif damage == "series_reordered":
        benchmark = replace(benchmark, performance=replace(benchmark.performance,
            net_value=benchmark.performance.net_value.iloc[::-1]))
    elif damage == "curve_date":
        benchmark.backtest.equity_curve.loc[1, "date"] = pd.Timestamp("2020-01-01")
    elif damage == "curve_short":
        benchmark = replace(benchmark, backtest=replace(benchmark.backtest,
            equity_curve=benchmark.backtest.equity_curve.iloc[:-1]))
    elif damage == "schema":
        meta["schema_version"] = "unknown"
    else:
        meta["status"] = "partial"
    ui = display()
    assert hasattr(ui, helper), "比较展示尚未实现"
    with pytest.raises(PortfolioBacktestError) as error:
        getattr(ui, helper)(strategy, benchmark)
    assert error.value.reason == "invalid_comparison"


def test_comparison_metrics_formats_both_sides_and_difference_is_percentage_points(comparison_pair):
    ui = display()
    assert hasattr(ui, "portfolio_comparison_metrics"), "比较指标尚未实现"
    table = ui.portfolio_comparison_metrics(*comparison_pair)
    assert table.columns.tolist() == ["指标", "组合策略", "全池等权买入持有"]
    assert table.values.tolist() == [
        ["期末总资产（元）", "120000.00", "105000.00"],
        ["累计收益率", "20.00%", "5.00%"], ["年化收益率", "30.00%", "—"],
        ["年化波动率", "10.00%", "20.00%"], ["最大回撤", "-5.00%", "-10.00%"],
        ["夏普比率", "1.50", "—"],
    ]
    assert ui.cumulative_return_difference_pp(*comparison_pair) == pytest.approx(15.)
    strategy, benchmark = comparison_pair
    missing = replace(benchmark, performance=replace(benchmark.performance, cumulative_return=None))
    assert ui.cumulative_return_difference_pp(strategy, missing) is None
    zero = replace(benchmark, performance=replace(benchmark.performance, cumulative_return=.2))
    assert ui.cumulative_return_difference_pp(strategy, zero) == 0.


@pytest.mark.parametrize("periods", [1, 6001])
def test_comparison_chart_keeps_all_points_and_distinguishes_series(periods):
    table = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=periods),
                          "strategy_net_value": np.arange(periods) / 1000 + 1.,
                          "benchmark_net_value": np.full(periods, 1.)})
    before = table.copy(deep=True)
    ui = display()
    assert hasattr(ui, "build_portfolio_comparison_chart"), "比较曲线尚未实现"
    spec = json.loads(ui.build_portfolio_comparison_chart(table).to_json())
    rows = chart_rows(spec)
    assert len(rows) == 2 * periods
    assert sum(row["series"] == "组合策略" for row in rows) == periods
    assert rows[-1]["net_value"] == 1.
    encoding = spec["encoding"]
    assert encoding["y"]["title"] == "净值（起点=1）"
    assert encoding["color"]["scale"]["domain"] == ["组合策略", "全池等权买入持有"]
    assert encoding["color"]["scale"]["range"] == ["#2563eb", "#64748b"]
    assert encoding["color"]["legend"]["orient"] == "top"
    assert encoding["strokeDash"]["scale"]["range"] == [[1, 0], [6, 4]]
    assert spec["mark"]["point"] is (periods == 1)
    ticks = encoding["x"]["axis"]["values"]
    expected = table.date.iloc[sorted({0, (periods - 1) // 2, periods - 1})]
    assert ticks == [{"year": d.year, "month": d.month, "date": d.day} for d in expected]
    assert "transform" not in spec
    pd.testing.assert_frame_equal(table, before)


def test_invalid_comparison_error_is_safe_and_explains_incompatible_research():
    message = display().portfolio_error_message(PortfolioBacktestError(
        "SECRET /Users/private/data.csv", reason="invalid_comparison"))
    assert "比较" in message and "日期" in message and "成本" in message
    assert "SECRET" not in message and "Users" not in message


def test_real_strategy_and_universe_benchmark_comparison(make_portfolio_preview):
    from quantlab.portfolio import run_equal_weight_benchmark

    preview, _ = make_portfolio_preview(periods=23)
    costs = BacktestConfig(commission_rate=0, minimum_commission=10, slippage_rate=0)
    strategy = analyze_portfolio_result(run_factor_portfolio(
        preview, factor_name="momentum_20", config=PortfolioConfig(top_n=1, backtest=costs)))
    benchmark = run_equal_weight_benchmark(preview, factor_name="momentum_20", backtest_config=costs)
    table = display().portfolio_comparison_table(strategy, benchmark)
    assert len(table) == 23
    assert table.strategy_net_value.iloc[:21].eq(1.).all()
    assert table.benchmark_net_value.iloc[:21].eq(1.).all()
    assert table.strategy_net_value.iloc[21] == pytest.approx(.9999)
    assert table.benchmark_net_value.iloc[21] == pytest.approx(.9998)
    assert len(display().portfolio_comparison_metrics(strategy, benchmark)) == 6
@pytest.mark.parametrize("background", ["#ffffff", "#0e1117"])
def test_comparison_lines_have_nontext_contrast_on_light_and_dark(background):
    from quantlab.ui.portfolio_display import build_portfolio_comparison_chart
    table = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=2),
                          "strategy_net_value": [1., 1.1], "benchmark_net_value": [1., 1.01]})
    colors = build_portfolio_comparison_chart(table).to_dict()["encoding"]["color"]["scale"]["range"]
    def luminance(color):
        rgb = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in rgb]
        return sum(c * w for c, w in zip(linear, [.2126, .7152, .0722]))
    for color in colors:
        low, high = sorted([luminance(color), luminance(background)])
        assert (high + .05) / (low + .05) >= 3
