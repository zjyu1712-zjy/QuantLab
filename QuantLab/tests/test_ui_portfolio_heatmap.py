"""热力图只重绘已完成的批量汇总，不重跑、不改档案。"""

from copy import deepcopy
import json

import pandas as pd
import pytest
from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df

from test_ui_portfolio import portfolio_sets, page_for, run_portfolio
from test_ui_portfolio_sensitivity import run_grid, tables
from test_ui_portfolio_records import workspace, page as history_page, open_history


def spec(page):
    specs = [json.loads(chart.proto.spec) for chart in page.get("vega_lite_chart")]
    matches = [chart for chart in specs if chart.get("title", "").startswith("参数热力图")]
    assert len(matches) == 1, "应只展示一个独立参数热力图"
    return matches[0]


def test_heatmap_only_after_explicit_run_and_switching_never_recalculates(portfolio_sets, monkeypatch):
    from quantlab.ui import portfolio_sensitivity
    _, make = portfolio_sets
    saved, _, _ = make(periods=25)
    page = page_for(saved)
    assert not any(s.key == "portfolio_sensitivity_metric" for s in page.selectbox)
    raw = run_grid(page)
    before = deepcopy(raw.to_dict("records"))
    identity = deepcopy(page.session_state["portfolio_sensitivity_identity"])
    assert page.selectbox(key="portfolio_sensitivity_metric").value == "cumulative_return"
    def forbidden(*args, **kwargs):
        pytest.fail("切换展示指标不得重新回测")
    monkeypatch.setattr(portfolio_sensitivity, "run_parameter_sensitivity", forbidden)
    for metric in ("annualized_return", "annualized_volatility", "max_drawdown", "sharpe_ratio", "cumulative_return"):
        page.selectbox(key="portfolio_sensitivity_metric").set_value(metric).run()
        assert not page.exception and not page.error
        assert page.session_state["portfolio_sensitivity_result"].to_dict("records") == before
        assert page.session_state["portfolio_sensitivity_identity"] == identity
        assert len(tables(page)) == 1
        assert spec(page)["layer"][0]["encoding"]["color"]["condition"]["legend"]["format"] == (
            ".2f" if metric == "sharpe_ratio" else ".2%")


@pytest.mark.parametrize("periods,zero_volume", [(1, None), (25, {"159915": [21]}),
    (25, {"510300": [21], "159915": [21]})])
def test_short_history_partial_and_all_failure_still_render_all_cells(portfolio_sets, periods, zero_volume):
    _, make = portfolio_sets
    saved, _, _ = make(periods=periods, zero_volume=zero_volume)
    page = page_for(saved)
    raw = run_grid(page)
    before = deepcopy(raw.to_dict("records"))
    page.selectbox(key="portfolio_sensitivity_metric").set_value("sharpe_ratio").run()
    assert not page.exception and not page.error
    spec(page)
    pd.testing.assert_frame_equal(page.session_state["portfolio_sensitivity_result"], pd.DataFrame(before).astype(raw.dtypes.to_dict()))
    assert len(tables(page)[0]) == 6


def test_fixed_input_change_removes_chart_and_selector(portfolio_sets):
    _, make = portfolio_sets
    saved, _, _ = make(periods=25)
    page = page_for(saved)
    run_grid(page)
    page.selectbox(key="portfolio_sensitivity_metric").set_value("sharpe_ratio").run()
    page.number_input(key="portfolio_initial_cash").set_value(200000.).run()
    assert not page.exception and not page.get("vega_lite_chart")
    assert "portfolio_sensitivity_metric" not in page.session_state
    assert not tables(page)
    run_grid(page)
    assert page.selectbox(key="portfolio_sensitivity_metric").value == "cumulative_return"


@pytest.mark.parametrize("metric,title,label", [
    ("max_drawdown", "参数热力图 · 最大回撤", "0.00%"),
    ("sharpe_ratio", "参数热力图 · 夏普比率", "不可用"),
])
def test_rerunning_same_grid_keeps_selected_metric_and_matching_chart(portfolio_sets, metric, title, label):
    """重跑仅替换结果；不能删除仍显示在浏览器中的指标选择状态。"""
    _, make = portfolio_sets
    saved, _, _ = make(periods=1)
    page = page_for(saved)
    run_grid(page)
    page.selectbox(key="portfolio_sensitivity_metric").set_value(metric).run()
    single = run_portfolio(page)
    baseline = page.session_state["portfolio_benchmark"]
    for _ in range(2):
        run_grid(page)
        assert page.selectbox(key="portfolio_sensitivity_metric").value == metric
        chart = spec(page)
        assert chart["title"] == title
        element = next(item for item in page.get("vega_lite_chart")
                       if json.loads(item.proto.spec).get("title") == title)
        cells = convert_arrow_bytes_to_pandas_df(element.proto.data.data)
        assert len(cells) == 6 and cells["label"].eq(label).all()
        assert page.session_state["portfolio_analysis"] is single
        assert page.session_state["portfolio_benchmark"] is baseline


def test_chart_failure_preserves_raw_table_and_single_result(portfolio_sets, monkeypatch):
    from quantlab.ui import portfolio_sensitivity
    _, make = portfolio_sets
    saved, _, _ = make(periods=25)
    page = page_for(saved)
    single = run_portfolio(page)
    raw = run_grid(page)
    def fail(*args, **kwargs):
        raise ValueError("PRIVATE_SECRET_PATH")
    monkeypatch.setattr(portfolio_sensitivity, "build_parameter_heatmap", fail)
    page.selectbox(key="portfolio_sensitivity_metric").set_value("sharpe_ratio").run()
    assert not page.exception and page.warning
    assert "PRIVATE_SECRET" not in str([item.value for item in page.warning])
    assert page.session_state["portfolio_analysis"] is single
    pd.testing.assert_frame_equal(page.session_state["portfolio_sensitivity_result"], raw)
    assert len(tables(page)) == 1


def test_switching_metrics_preserves_current_history_and_zip(workspace, monkeypatch):
    from quantlab.portfolio import build_portfolio_export, save_portfolio_run, load_portfolio_run
    from quantlab.ui import portfolio_sensitivity
    _, archives, create = workspace
    _, strategy, benchmark, _ = create(periods=25)
    record = save_portfolio_run(strategy, archives, name="热力图隔离检查", benchmark=benchmark)
    original = record.path.read_bytes()
    archive_zip = build_portfolio_export(load_portfolio_run(archives, record.run_id)).content
    page = history_page()
    single = run_portfolio(page)
    baseline = page.session_state["portfolio_benchmark"]
    open_history(page, record.run_id)
    metrics = [(item.label, item.value) for item in page.metric]
    run_grid(page)
    def forbidden(*args, **kwargs):
        pytest.fail("绘图不应批量运行")
    monkeypatch.setattr(portfolio_sensitivity, "run_parameter_sensitivity", forbidden)
    page.selectbox(key="portfolio_sensitivity_metric").set_value("max_drawdown").run()
    assert not page.exception and not page.error
    spec(page)
    assert page.session_state["portfolio_analysis"] is single
    assert page.session_state["portfolio_benchmark"] is baseline
    assert page.session_state["portfolio_history_id"] == record.run_id
    assert [(item.label, item.value) for item in page.metric] == metrics
    assert record.path.read_bytes() == original
    assert build_portfolio_export(load_portfolio_run(archives, record.run_id)).content == archive_zip
