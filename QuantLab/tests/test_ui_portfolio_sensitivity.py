"""显式运行参数网格，结果不影响单次回测、历史档案或 ZIP。"""

import json

import pandas as pd
import pytest

from test_ui_portfolio import portfolio_sets, page_for, run_portfolio
from test_ui_portfolio_records import workspace, page as history_page, open_history
from quantlab.data.research_storage import load_research_set


def run_grid(page):
    page.button(key="portfolio_sensitivity_run").click().run()
    assert not page.exception
    return page.session_state["portfolio_sensitivity_result"]


def tables(page):
    return [item.value for item in page.dataframe if "调仓间隔（期）" in item.value.columns]


def test_explicit_independent_button_and_no_disk_writes(portfolio_sets):
    root, make = portfolio_sets
    saved, _, source = make(periods=45)
    page = page_for(saved)
    assert "portfolio_sensitivity_result" not in page.session_state and not tables(page)
    source.rename(source.with_name(source.name + "_moved"))
    before = {p: p.read_bytes() for p in root.parent.rglob("*") if p.is_file()}
    table = run_grid(page)
    assert len(table) == 6 and len(tables(page)) == 1
    assert "portfolio_analysis" not in page.session_state
    assert "portfolio_benchmark" not in page.session_state
    assert {p: p.read_bytes() for p in root.parent.rglob("*") if p.is_file()} == before
    assert list(tables(page)[0]["状态"].unique()) == ["完成"]


def test_grid_never_replaces_single_result_or_baseline_and_nk_dont_clear_grid(portfolio_sets):
    _, make = portfolio_sets
    saved, _, _ = make(periods=45)
    page = page_for(saved)
    single = run_portfolio(page)
    baseline = page.session_state["portfolio_benchmark"]
    table = run_grid(page).copy(deep=True)
    assert page.session_state["portfolio_analysis"] is single
    assert page.session_state["portfolio_benchmark"] is baseline
    page.number_input(key="portfolio_top_n").set_value(1).run()
    page.number_input(key="portfolio_rebalance_interval").set_value(7).run()
    pd.testing.assert_frame_equal(page.session_state["portfolio_sensitivity_result"], table)
    run_portfolio(page)
    pd.testing.assert_frame_equal(page.session_state["portfolio_sensitivity_result"], table)


@pytest.mark.parametrize("key,value", [("portfolio_direction", "ascending"),
    ("portfolio_initial_cash", 200000.), ("portfolio_commission_pct", .1),
    ("portfolio_minimum_commission", 5.), ("portfolio_slippage_pct", .2),
    ("portfolio_sell_tax_pct", .1), ("portfolio_lot_size", 10), ("factor_name", "volatility_20")])
def test_fixed_input_changes_clear_grid_without_automatic_rerun(portfolio_sets, key, value):
    _, make = portfolio_sets
    saved, _, _ = make(periods=25)
    page = page_for(saved)
    run_grid(page)
    controls = page.selectbox if key in {"portfolio_direction", "factor_name"} else page.number_input
    controls(key=key).set_value(value).run()
    assert not page.exception
    assert "portfolio_sensitivity_result" not in page.session_state and not tables(page)


@pytest.mark.parametrize("problem", ["corrupt", "scan", "load", "scope", "switch", "replace"])
def test_data_identity_and_failures_clear_grid(portfolio_sets, monkeypatch, tmp_path, problem):
    import quantlab.data.research_storage as storage
    from quantlab.ui import runtime
    root, make = portfolio_sets
    saved, _, _ = make(periods=25)
    page = page_for(saved)
    run_grid(page)
    def fail(*args, **kwargs):
        raise OSError("SECRET_PRIVATE_PATH")
    if problem == "corrupt":
        (saved.directory / "manifest.json").write_text("{}")
    elif problem == "scan":
        monkeypatch.setattr(storage, "list_research_sets", fail)
    elif problem == "load":
        scan = storage.list_research_sets(root)
        monkeypatch.setattr(storage, "list_research_sets", lambda *args: scan)
        monkeypatch.setattr(storage, "load_research_set", fail)
    elif problem == "scope":
        monkeypatch.setattr(runtime, "get_research_storage_dir", lambda project: tmp_path / "other")
    else:
        second, changed, _ = make(periods=10)
        if problem == "switch":
            page.run()
            page.selectbox(key="factor_set_id").set_value(second.set_id)
        else:
            manifest = json.loads((second.directory / "manifest.json").read_text())
            manifest.update(set_id=saved.set_id, name=saved.name, created_at=saved.created_at)
            for name in ("aligned.parquet", "coverage.csv"):
                (saved.directory / name).write_bytes((second.directory / name).read_bytes())
            (saved.directory / "manifest.json").write_text(json.dumps(manifest))
            assert load_research_set(root, saved.set_id).preview.preview_sha256 == changed.preview_sha256
    page.run()
    assert not page.exception
    assert "portfolio_sensitivity_result" not in page.session_state and not tables(page)
    assert "SECRET_PRIVATE_PATH" not in str([e.value for e in page.error])


def test_view_only_changes_keep_grid(portfolio_sets):
    _, make = portfolio_sets
    saved, _, _ = make(periods=25)
    page = page_for(saved)
    table = run_grid(page).copy(deep=True)
    page.button(key="factor_calculate").click().run()
    page.selectbox(key="factor_view_date").set_value(page.selectbox(key="factor_view_date").options[0]).run()
    page.selectbox(key="factor_view_symbol").set_value("510300").run()
    assert not page.exception
    pd.testing.assert_frame_equal(page.session_state["portfolio_sensitivity_result"], table)


def test_failures_and_missing_metrics_are_visible_not_zero(portfolio_sets):
    _, make = portfolio_sets
    saved, _, _ = make(periods=25, zero_volume={"159915": [21]})
    page = page_for(saved)
    run_grid(page)
    table = tables(page)[0]
    assert table["状态"].tolist() == ["完成"] * 3 + ["失败"] * 3
    assert table.iloc[3:]["成交笔数"].eq("—").all()
    assert table.iloc[3:]["累计收益率"].eq("—").all()
    assert table.iloc[3:]["说明"].str.contains("成交量").all()
    short, _, _ = make(periods=1)
    page.run()
    page.selectbox(key="factor_set_id").set_value(short.set_id).run()
    run_grid(page)
    table = tables(page)[0]
    assert table["状态"].eq("完成（零成交）").all()
    assert table["夏普比率"].eq("—").all()
    assert table["说明"].str.contains("夏普比率").all()


def test_batch_level_failure_clears_grid_not_single(portfolio_sets, monkeypatch):
    from quantlab.ui import portfolio_sensitivity
    _, make = portfolio_sets
    saved, _, _ = make(periods=25)
    page = page_for(saved)
    single = run_portfolio(page)
    run_grid(page)
    def fail(*args, **kwargs):
        raise RuntimeError("PRIVATE_SECRET")
    monkeypatch.setattr(portfolio_sensitivity, "run_parameter_sensitivity", fail)
    page.button(key="portfolio_sensitivity_run").click().run()
    assert not page.exception and page.error
    assert "portfolio_sensitivity_result" not in page.session_state and not tables(page)
    assert page.session_state["portfolio_analysis"] is single
    assert "PRIVATE_SECRET" not in page.error[0].value


def test_current_history_archive_and_zip_stay_independent(workspace):
    from quantlab.portfolio import build_portfolio_export, save_portfolio_run, load_portfolio_run
    _, archives, create = workspace
    _, strategy, benchmark, _ = create(periods=45)
    record = save_portfolio_run(strategy, archives, name="原历史实验", benchmark=benchmark)
    original = record.path.read_bytes()
    zip_before = build_portfolio_export(load_portfolio_run(archives, record.run_id)).content
    page = history_page()
    single = run_portfolio(page)
    baseline = page.session_state["portfolio_benchmark"]
    open_history(page, record.run_id)
    metrics = [(item.label, item.value) for item in page.metric]
    run_grid(page)
    assert not page.error and page.session_state["portfolio_history_id"] == record.run_id
    assert page.session_state["portfolio_analysis"] is single
    assert page.session_state["portfolio_benchmark"] is baseline
    assert [(item.label, item.value) for item in page.metric] == metrics
    assert record.path.read_bytes() == original
    assert build_portfolio_export(load_portfolio_run(archives, record.run_id)).content == zip_before
    assert len(list(archives.glob("*.json"))) == 1
