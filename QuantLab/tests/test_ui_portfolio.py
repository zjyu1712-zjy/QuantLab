"""组合页沿真实快照→回测→指标链路验收；不读取个人行情。"""

from copy import deepcopy
import json

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from quantlab.data.research_storage import load_research_set, save_research_set
from quantlab.ui import runtime


PAGE = "pages/7_因子研究.py"


@pytest.fixture
def portfolio_sets(tmp_path, monkeypatch, make_portfolio_preview):
    root = tmp_path / "sets"
    monkeypatch.delenv("QUANTLAB_PUBLIC_MODE", raising=False)
    monkeypatch.delenv("QUANTLAB_ACCESS_PASSWORD", raising=False)
    monkeypatch.setattr(runtime, "get_research_storage_dir", lambda project: root)

    def make(**kwargs):
        preview, source = make_portfolio_preview(**kwargs)
        saved = save_research_set(preview, root, name="组合界面合成验收")
        return saved, preview, source

    return root, make


def page_for(saved):
    page = AppTest.from_file(PAGE, default_timeout=20).run()
    assert not page.exception
    page.selectbox(key="factor_set_id").set_value(saved.set_id).run()
    assert not page.exception
    return page


def run_portfolio(page):
    page.button(key="portfolio_run").click().run()
    assert not page.exception and not page.error
    return page.session_state["portfolio_analysis"]


def assert_no_result(page):
    assert "portfolio_analysis" not in page.session_state
    assert "portfolio_result_identity" not in page.session_state
    assert "portfolio_benchmark" not in page.session_state
    assert "portfolio_benchmark_error" not in page.session_state
    assert not page.metric


def test_explicit_run_without_factor_calculation_uses_full_snapshot(portfolio_sets):
    root, make = portfolio_sets
    saved, preview, _ = make()
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    page = page_for(saved)
    assert [tab.label for tab in page.tabs] == [
        "当前研究", "因子观察", "组合回测", "双因子预览", "历史实验",
    ]
    assert_no_result(page)
    assert "factor_result" not in page.session_state
    analysis = run_portfolio(page)
    result = analysis.backtest
    assert len(result.equity_curve) == len(preview.alignment.calendar.dates)
    assert result.equity_curve.iloc[0].equity == 100000.0
    assert not result.trades.empty
    assert result.trades.execution_date.gt(result.trades.signal_date).all()
    assert len(page.get("vega_lite_chart")) == 3
    metrics = {metric.label: metric.value for metric in page.metric}
    assert metrics["累计收益率"] == f"{analysis.performance.cumulative_return:.2%}"
    assert metrics["最大回撤"] == f"{analysis.performance.max_drawdown:.2%}"
    assert "factor_result" not in page.session_state
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_benchmark_uses_same_snapshot_costs_and_shows_percentage_points(portfolio_sets):
    _, make = portfolio_sets
    saved, _, _ = make(periods=45)
    page = page_for(saved)
    page.number_input(key="portfolio_top_n").set_value(1).run()
    page.number_input(key="portfolio_commission_pct").set_value(.1).run()
    strategy = run_portfolio(page)
    baseline = page.session_state["portfolio_benchmark"]
    assert baseline.backtest.metadata["schema_version"] == "portfolio-benchmark-v1"
    assert baseline.backtest.metadata["strategy"]["config"]["backtest"] == strategy.backtest.metadata["strategy"]["config"]["backtest"]
    pd.testing.assert_series_equal(baseline.backtest.equity_curve.date, strategy.backtest.equity_curve.date)
    diff = (strategy.performance.cumulative_return - baseline.performance.cumulative_return) * 100
    assert any(m.label == "累计收益差（策略 − 基准）" and m.value == f"{diff:+.2f} 个百分点" for m in page.metric)
    assert any("净值对照" in s.value for s in page.subheader)
    assert any("全池等权买入持有" in c.value and "不再调仓" in c.value for c in page.caption)


def test_benchmark_failure_keeps_strategy_but_never_stale_comparison(portfolio_sets, monkeypatch):
    from quantlab.ui import portfolio
    _, make = portfolio_sets
    saved, _, _ = make()
    page = page_for(saved)
    run_portfolio(page)
    assert "portfolio_benchmark" in page.session_state
    def broken(*args, **kwargs):
        raise ValueError("/secret/path confidential token")
    monkeypatch.setattr(portfolio, "run_equal_weight_benchmark", broken)
    run_portfolio(page)
    assert "portfolio_benchmark" not in page.session_state
    assert len(page.get("vega_lite_chart")) == 2
    assert not any("累计收益差" in m.label for m in page.metric)
    assert any("基准不可用" in w.value for w in page.warning)
    assert "secret" not in str(page.warning)
    page.number_input(key="portfolio_top_n").set_value(1).run()
    assert_no_result(page)


def test_real_unselected_missing_price_can_make_only_benchmark_unavailable(portfolio_sets):
    _, make = portfolio_sets
    saved, _, _ = make(periods=25, gaps={"159915": [24]})
    page = page_for(saved)
    page.number_input(key="portfolio_top_n").set_value(1).run()
    assert not run_portfolio(page).backtest.trades.empty
    assert "portfolio_benchmark" not in page.session_state
    assert any("基准不可用" in w.value and "收盘" in w.value for w in page.warning)


def test_short_history_explicitly_explains_benchmark_cash(portfolio_sets):
    _, make = portfolio_sets
    saved, _, _ = make(periods=1)
    page = page_for(saved)
    run_portfolio(page)
    assert page.session_state["portfolio_benchmark"].backtest.trades.empty
    assert any("未到计划建仓日" in i.value for i in page.info)
    assert any(m.label == "累计收益差（策略 − 基准）" and m.value == "—" for m in page.metric)


def test_benchmark_zero_lot_explained_without_pointing_to_empty_orders(portfolio_sets):
    _, make = portfolio_sets
    saved, _, _ = make(periods=25)
    page = page_for(saved)
    page.number_input(key="portfolio_initial_cash").set_value(1.0).run()
    run_portfolio(page)
    assert page.session_state["portfolio_benchmark"].backtest.orders.empty
    assert any("基准" in i.value and "不足一手" in i.value for i in page.info)
    assert not any("原因见基准委托明细" in i.value for i in page.info)


@pytest.mark.parametrize("key,value", [
    ("portfolio_top_n", 1), ("portfolio_rebalance_interval", 5),
    ("portfolio_initial_cash", 200000.0), ("portfolio_commission_pct", .05),
    ("portfolio_minimum_commission", 5.0), ("portfolio_slippage_pct", .1),
    ("portfolio_sell_tax_pct", .1), ("portfolio_lot_size", 10),
])
def test_parameter_change_clears_result_without_automatic_run(portfolio_sets, key, value):
    _, make = portfolio_sets
    saved, _, _ = make()
    page = page_for(saved)
    run_portfolio(page)
    page.number_input(key=key).set_value(value).run()
    assert not page.exception
    assert_no_result(page)


def test_direction_and_factor_change_clear_and_default_direction(portfolio_sets):
    _, make = portfolio_sets
    saved, _, _ = make(periods=145)
    page = page_for(saved)
    assert page.selectbox(key="portfolio_direction").value == "descending"
    run_portfolio(page)
    page.selectbox(key="portfolio_direction").set_value("ascending").run()
    assert_no_result(page)
    run_portfolio(page)
    page.selectbox(key="factor_name").set_value("volatility_20").run()
    assert_no_result(page)
    assert page.selectbox(key="portfolio_direction").value == "ascending"
    analysis = run_portfolio(page)
    assert analysis.backtest.metadata["strategy"]["resolved_ascending"] is True
    page.selectbox(key="factor_name").set_value("momentum_20").run()
    assert page.selectbox(key="portfolio_direction").value == "descending"
    assert_no_result(page)


@pytest.mark.parametrize("factor", ["momentum_20", "momentum_60", "momentum_120", "volatility_20", "volatility_60"])
def test_all_registered_factors_can_run(portfolio_sets, factor):
    _, make = portfolio_sets
    saved, _, _ = make(periods=145)
    page = page_for(saved)
    page.selectbox(key="factor_name").set_value(factor).run()
    result = run_portfolio(page).backtest
    assert result.metadata["factor_definition"]["name"] == factor
    assert not result.trades.empty


def test_display_filters_preserve_complete_portfolio_result(portfolio_sets):
    _, make = portfolio_sets
    saved, preview, _ = make()
    page = page_for(saved)
    page.button(key="factor_calculate").click().run()
    analysis = run_portfolio(page)
    before = deepcopy(analysis.backtest.equity_curve)
    run_id = analysis.backtest.metadata["run_id"]
    page.selectbox(key="factor_view_date").set_value(preview.alignment.calendar.dates[0]).run()
    page.selectbox(key="factor_view_symbol").set_value("510300").run()
    page.radio(key="factor_daily_order").set_value("因子排名").run()
    page.number_input(key="factor_selection_n").set_value(1).run()
    assert not page.exception
    after = page.session_state["portfolio_analysis"].backtest
    assert after.metadata["run_id"] == run_id
    pd.testing.assert_frame_equal(after.equity_curve, before)


def test_set_switch_scope_change_and_invalid_snapshot_remove_results(portfolio_sets, monkeypatch, tmp_path):
    root, make = portfolio_sets
    first, _, _ = make()
    second, _, _ = make(periods=70)
    page = page_for(first)
    run_portfolio(page)
    page.selectbox(key="factor_set_id").set_value(second.set_id).run()
    assert_no_result(page)
    run_portfolio(page)
    monkeypatch.setattr(runtime, "get_research_storage_dir", lambda project: tmp_path / "other")
    page.run()
    assert_no_result(page)
    monkeypatch.setattr(runtime, "get_research_storage_dir", lambda project: root)
    page.run()
    run_portfolio(page)
    selected = page.selectbox(key="factor_set_id").value
    # 损坏测试快照，不改变或删除任何用户文件。
    manifest = root / selected / "manifest.json"
    assert manifest.exists()
    manifest.write_text("{}", encoding="utf-8")
    page.run()
    assert not page.exception
    assert_no_result(page)


@pytest.mark.parametrize("stage", ["run", "analysis"])
def test_failure_clears_previous_success_and_hides_private_exception(portfolio_sets, monkeypatch, stage):
    from quantlab.ui import portfolio

    _, make = portfolio_sets
    saved, _, _ = make()
    page = page_for(saved)
    run_portfolio(page)

    def fail(*args, **kwargs):
        raise RuntimeError("/Users/private-secret token=DO_NOT_DISPLAY")

    monkeypatch.setattr(portfolio, "run_factor_portfolio" if stage == "run" else "analyze_portfolio_result", fail)
    page.button(key="portfolio_run").click().run()
    assert not page.exception and page.error
    assert_no_result(page)
    assert "DO_NOT_DISPLAY" not in page.error[0].value
    assert "private-secret" not in page.error[0].value


def test_short_history_all_cash_and_undefined_metrics_are_explicit(portfolio_sets):
    _, make = portfolio_sets
    saved, _, _ = make(periods=1)
    page = page_for(saved)
    analysis = run_portfolio(page)
    assert analysis.backtest.trades.empty
    metrics = {metric.label: metric.value for metric in page.metric}
    assert metrics["累计收益率"] == "—"
    assert metrics["年化收益率"] == "—"
    assert metrics["夏普比率"] == "—"
    assert any("未发生成交" in item.value for item in page.info)


def test_real_zero_volume_failure_has_no_partial_result(portfolio_sets):
    _, make = portfolio_sets
    saved, _, _ = make(zero_volume={"510300": [21], "159915": [21]})
    page = page_for(saved)
    page.button(key="portfolio_run").click().run()
    assert not page.exception and page.error
    assert_no_result(page)
    assert "成交量" in page.error[0].value


def test_fee_percent_conversion_and_moved_source_snapshot_only(portfolio_sets):
    root, make = portfolio_sets
    saved, _, source = make()
    source.rename(source.with_name(source.name + "_moved"))
    page = page_for(saved)
    page.number_input(key="portfolio_commission_pct").set_value(.5).run()
    page.number_input(key="portfolio_slippage_pct").set_value(.2).run()
    page.number_input(key="portfolio_sell_tax_pct").set_value(.1).run()
    analysis = run_portfolio(page)
    config = analysis.backtest.metadata["strategy"]["config"]["backtest"]
    assert config["commission_rate"] == .005
    assert config["slippage_rate"] == .002
    assert config["sell_tax_rate"] == .001
    first_run = analysis.backtest.metadata["run_id"]
    assert run_portfolio(page).backtest.metadata["run_id"] != first_run


def test_same_set_id_with_replaced_valid_snapshot_invalidates_result(portfolio_sets):
    root, make = portfolio_sets
    first, _, _ = make()
    replacement, changed, _ = make(periods=10)
    page = page_for(first)
    run_portfolio(page)
    manifest = json.loads((replacement.directory / "manifest.json").read_text())
    manifest.update(set_id=first.set_id, name=first.name, created_at=first.created_at)
    for name in ("aligned.parquet", "coverage.csv"):
        (first.directory / name).write_bytes((replacement.directory / name).read_bytes())
    (first.directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert load_research_set(root, first.set_id).preview.preview_sha256 == changed.preview_sha256
    page.run()
    assert not page.exception
    assert_no_result(page)
    result = run_portfolio(page).backtest
    assert len(result.equity_curve) == 10 and result.trades.empty


def test_snapshot_load_failure_clears_result_before_rendering(portfolio_sets, monkeypatch):
    import quantlab.data.research_storage as storage

    _, make = portfolio_sets
    saved, _, _ = make()
    page = page_for(saved)
    run_portfolio(page)
    scan = storage.list_research_sets(portfolio_sets[0])

    def fail(*args, **kwargs):
        raise OSError("private-source-path")

    # 保持扫描成功，只模拟随后的载入失败（两次读取之间文件可能变化）。
    monkeypatch.setattr(storage, "list_research_sets", lambda *args: scan)
    monkeypatch.setattr(storage, "load_research_set", fail)
    page.run()
    assert not page.exception and page.error
    assert_no_result(page)
    assert "private-source-path" not in page.error[0].value


def test_new_session_never_inherits_another_sessions_result(portfolio_sets):
    _, make = portfolio_sets
    saved, _, _ = make()
    first = page_for(saved)
    run_portfolio(first)
    second = page_for(saved)
    assert_no_result(second)
    assert "portfolio_analysis" in first.session_state
