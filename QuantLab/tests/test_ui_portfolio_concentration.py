"""集中度派生展示：当前与历史各用自己的曲线，失败不影响原回测。"""

from copy import deepcopy
import importlib

import pandas as pd
import pytest

from quantlab.portfolio import (PortfolioBacktestError, PortfolioConfig, analyze_portfolio_result,
    run_factor_portfolio, save_portfolio_run, load_portfolio_run, build_portfolio_export)
from test_ui_portfolio_yearly import show
from test_ui_portfolio_records import workspace, page, run_live, open_history


def ui():
    assert importlib.util.find_spec("quantlab.ui.portfolio_concentration") is not None, "集中度界面尚未实现"
    return importlib.import_module("quantlab.ui.portfolio_concentration")


def details(app):
    return [item.value for item in app.dataframe if "占全部正向增量比例" in item.value.columns]


@pytest.fixture
def analysis(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=45)
    return analyze_portfolio_result(run_factor_portfolio(preview, factor_name="momentum_20",
                                                       config=PortfolioConfig(top_n=1)))


def test_current_result_displays_concentration_without_changing_metrics(analysis):
    before = deepcopy(analysis)
    app = show(analysis)
    assert not app.exception and not app.error
    assert len(details(app)) == 1, "当前结果应接入集中度明细"
    assert 0 < len(details(app)[0]) <= 5
    assert any("收益集中度检查" in item.value for item in app.subheader)
    assert any("不是最终净利润" in c.value for c in app.caption)
    assert len(app.metric) == 6
    pd.testing.assert_frame_equal(analysis.backtest.equity_curve, before.backtest.equity_curve)
    pd.testing.assert_series_equal(analysis.performance.drawdown, before.performance.drawdown)
    assert analysis.backtest.metadata == before.backtest.metadata


@pytest.mark.parametrize("periods,reason", [(1,"没有收益区间"),(3,"没有正向资产增量")])
def test_short_and_cash_only_results_explain_unavailable(make_portfolio_preview, periods, reason):
    preview, _ = make_portfolio_preview(periods=periods)
    strategy = analyze_portfolio_result(run_factor_portfolio(preview, factor_name="momentum_20",
                                                            config=PortfolioConfig(top_n=1)))
    app = show(strategy)
    assert not app.exception and not details(app)
    assert any("集中度不可用" in item.value and reason in item.value for item in app.info)
    assert len(app.metric) == 6


@pytest.mark.parametrize("change", ["calendar", "truncated", "cash", "status", "schema"])
def test_adapter_rejects_incomplete_declared_curve(analysis, change):
    if change == "calendar": analysis.backtest.metadata["data"]["calendar"]["dates"].pop()
    elif change == "truncated": analysis.backtest.equity_curve.drop(index=0, inplace=True)
    elif change == "cash": analysis.backtest.metadata["strategy"]["config"]["backtest"]["initial_cash"] *= 2
    elif change == "status": analysis.backtest.metadata["status"] = "failed"
    else: analysis.backtest.metadata["schema_version"] = "portfolio-benchmark-v1"
    with pytest.raises(PortfolioBacktestError) as caught:
        ui().portfolio_gain_concentration(analysis)
    assert caught.value.reason == "invalid_portfolio_result"


def test_derived_failure_preserves_original_result_and_hides_private_error(analysis, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("PRIVATE_TOKEN /private/personal")
    monkeypatch.setattr(ui(), "portfolio_gain_concentration", fail)
    app = show(analysis)
    assert not app.exception and len(app.metric) == 6 and not details(app)
    assert any("集中度" in item.value for item in app.warning)
    assert "PRIVATE_TOKEN" not in str(app.warning)


@pytest.mark.parametrize("history_periods", [1, 45])
def test_history_uses_own_curve_without_rerun_or_archive_changes(workspace, monkeypatch, history_periods):
    from quantlab.ui import portfolio
    _, archives, create = workspace
    _, historical, baseline, _ = create(periods=history_periods)
    historical_tables = details(show(historical, historical=True))
    record = save_portfolio_run(historical, archives, name="短样本集中度", benchmark=baseline)
    saved_bytes = record.path.read_bytes()
    saved_zip = build_portfolio_export(load_portfolio_run(archives, record.run_id)).content
    current_set, _, _, _ = create(periods=65, cash=200000.)
    app = page()
    app.selectbox(key="factor_set_id").set_value(current_set.set_id).run()
    current = run_live(app)
    current_benchmark = app.session_state["portfolio_benchmark"]
    assert len(details(app)) == 1, "当前结果缺少集中度明细"
    before_table = details(app)[0].copy(deep=True)
    def forbidden(*args, **kwargs):
        pytest.fail("只读集中度展示不得重新回测")
    monkeypatch.setattr(portfolio, "run_factor_portfolio", forbidden)
    monkeypatch.setattr(portfolio, "run_equal_weight_benchmark", forbidden)
    open_history(app, record.run_id)
    app.run()
    assert not app.exception and not app.error
    assert len(details(app)) == 1 + len(historical_tables), "历史必须使用自身资金曲线"
    pd.testing.assert_frame_equal(details(app)[0], before_table)
    if history_periods == 1:
        assert any("集中度不可用" in i.value and "没有收益区间" in i.value for i in app.info)
    else:
        pd.testing.assert_frame_equal(details(app)[1], historical_tables[0])
        assert not details(app)[1].equals(before_table)
    assert any("集中度" in c.value and "已保存资金曲线" in c.value for c in app.caption)
    assert app.session_state["portfolio_analysis"] is current
    assert app.session_state["portfolio_benchmark"] is current_benchmark
    assert app.session_state["portfolio_history_id"] == record.run_id
    assert record.path.read_bytes() == saved_bytes
    assert build_portfolio_export(load_portfolio_run(archives, record.run_id)).content == saved_zip
