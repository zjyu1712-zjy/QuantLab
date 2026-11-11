"""分组展示复用自身策略/基准，失败不污染原指标或历史档案。"""

from copy import deepcopy
import importlib

import pandas as pd
import pytest

from quantlab.portfolio import (PortfolioBacktestError, PortfolioConfig, analyze_portfolio_result,
    run_equal_weight_benchmark, run_factor_portfolio, save_portfolio_run, load_portfolio_run, build_portfolio_export)
from test_ui_portfolio_yearly import show
from test_ui_portfolio_records import workspace, page, run_live, open_history


def ui():
    assert importlib.util.find_spec("quantlab.ui.portfolio_regimes") is not None, "分组界面尚未实现"
    return importlib.import_module("quantlab.ui.portfolio_regimes")


def tables(app):
    return [item.value for item in app.dataframe if "基准区间分类" in item.value.columns]


@pytest.fixture
def analyses(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=45)
    strategy = analyze_portfolio_result(run_factor_portfolio(preview, factor_name="momentum_20",
                                                            config=PortfolioConfig(top_n=1)))
    return strategy, run_equal_weight_benchmark(preview, factor_name="momentum_20")


def test_current_group_table_keeps_all_intervals_and_original_metrics(analyses):
    strategy, benchmark = analyses
    before = deepcopy(analyses)
    app = show(strategy, benchmark)
    assert not app.exception and not app.error
    assert len(tables(app)) == 1, "当前结果应展示基准涨跌分组"
    table = tables(app)[0].set_index("基准区间分类")
    assert table["区间数量"].sum() == 44
    assert table.loc["基准现金区间（不参与涨跌分组）", "区间数量"] == 20
    assert table.loc["基准首次建仓区间（不参与涨跌分组）", "区间数量"] == 1
    assert table["区间数量"].iloc[:3].sum() == 23
    assert len(app.metric) == 7
    assert any("事后" in c.value and "预测" in c.value for c in app.caption)
    assert any("算术平均" in c.value and "百分点" in c.value for c in app.caption)
    for value, original in zip(analyses, before):
        pd.testing.assert_frame_equal(value.backtest.equity_curve, original.backtest.equity_curve)
        pd.testing.assert_series_equal(value.performance.net_value, original.performance.net_value)
        assert value.backtest.metadata == original.backtest.metadata


@pytest.mark.parametrize("status,reason", [("not_run","未运行"),("unavailable","不可用")])
def test_missing_benchmark_preserves_strategy_and_explains_no_grouping(analyses, status, reason):
    app = show(analyses[0],status=status)
    assert not app.exception and not app.error and not tables(app)
    assert len(app.metric) == 6
    assert any("涨跌分组" in item.value and reason in item.value for item in app.info)


@pytest.mark.parametrize("periods,cash,entry,reason", [(1,0,0,"没有收益区间"),
    (3,2,0,"没有可分组的持仓区间"),(22,20,1,"没有可分组的持仓区间")])
def test_short_cash_and_entry_only_show_missing_means(make_portfolio_preview, periods, cash, entry, reason):
    preview, _ = make_portfolio_preview(periods=periods)
    strategy = analyze_portfolio_result(run_factor_portfolio(preview,factor_name="momentum_20",config=PortfolioConfig(top_n=1)))
    baseline = run_equal_weight_benchmark(preview,factor_name="momentum_20")
    app = show(strategy, baseline)
    assert not app.exception and not app.error and len(tables(app)) == 1
    table = tables(app)[0]
    assert table["区间数量"].tolist() == [0,0,0,cash,entry]
    assert table[["策略平均区间收益","基准平均区间收益","平均收益差（百分点）"]].isna().all().all()
    assert any(reason in item.value and "分组" in item.value for item in app.info)


@pytest.mark.parametrize("change", ["calendar","cost","anchor","dates","equity","cash", "position","rule","not_analysis"])
def test_adapter_rejects_invalid_or_incomparable_pair(analyses, change):
    strategy, baseline = analyses
    if change == "calendar": baseline.backtest.metadata["data"]["calendar"]["dates"].pop()
    elif change == "cost": baseline.backtest.metadata["strategy"]["config"]["backtest"]["slippage_rate"] = .9
    elif change == "anchor": baseline.backtest.metadata["strategy"]["signal_anchor_position"] += 1
    elif change == "dates": baseline.backtest.equity_curve.drop(index=1, inplace=True)
    elif change == "equity": baseline.backtest.equity_curve.loc[2,"equity"] = float("nan")
    elif change == "cash": strategy.backtest.equity_curve.loc[0,"equity"] *= 2
    elif change == "position": baseline.backtest.equity_curve.loc[2,"position_value"] = float("nan")
    elif change == "rule": baseline.backtest.metadata["strategy"]["execution_rule"] = "other"
    else: baseline = object()
    with pytest.raises(PortfolioBacktestError):
        ui().portfolio_regime_table(strategy,baseline)


def test_grouping_failure_hides_private_error_and_preserves_other_results(analyses, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("PRIVATE_TOKEN /private/person")
    monkeypatch.setattr(ui(),"portfolio_regime_table",fail)
    app = show(*analyses)
    assert not app.exception and not tables(app) and len(app.metric) == 7
    assert any("涨跌分组" in item.value for item in app.warning)
    assert "PRIVATE_TOKEN" not in str(app.warning)
    assert any("收益集中度检查" in item.value for item in app.subheader)


@pytest.mark.parametrize("history_periods,with_benchmark", [(1,True),(45,True),(45,False)])
def test_history_uses_own_pair_without_rerun_or_archive_changes(workspace, monkeypatch, history_periods, with_benchmark):
    from quantlab.ui import portfolio
    _, archives, create = workspace
    _, historical, baseline, _ = create(periods=history_periods,benchmark=with_benchmark)
    record = save_portfolio_run(historical,archives,name="历史分组",benchmark=baseline)
    saved_bytes = record.path.read_bytes()
    saved_zip = build_portfolio_export(load_portfolio_run(archives,record.run_id)).content
    current_set, _, _, _ = create(periods=65,cash=200000.)
    app = page()
    app.selectbox(key="factor_set_id").set_value(current_set.set_id).run()
    live = run_live(app)
    live_baseline = app.session_state["portfolio_benchmark"]
    assert len(tables(app)) == 1, "当前分组缺失"
    original_table = tables(app)[0].copy(deep=True)
    def forbidden(*args, **kwargs):
        pytest.fail("查看分组不应重新回测或计算原绩效")
    for name in ("run_factor_portfolio","run_equal_weight_benchmark","analyze_portfolio_result"):
        monkeypatch.setattr(portfolio,name,forbidden)
    open_history(app,record.run_id)
    app.run()
    assert not app.exception and not app.error
    assert len(tables(app)) == 1 + int(with_benchmark)
    pd.testing.assert_frame_equal(tables(app)[0],original_table)
    if with_benchmark:
        assert tables(app)[1]["区间数量"].sum() == history_periods - 1
        assert not tables(app)[1].equals(original_table)
    else:
        assert any("涨跌分组" in i.value and "未运行" in i.value for i in app.info)
    assert any("涨跌分组" in c.value and "本实验" in c.value for c in app.caption)
    assert app.session_state["portfolio_analysis"] is live
    assert app.session_state["portfolio_benchmark"] is live_baseline
    assert record.path.read_bytes() == saved_bytes
    assert build_portfolio_export(load_portfolio_run(archives,record.run_id)).content == saved_zip
