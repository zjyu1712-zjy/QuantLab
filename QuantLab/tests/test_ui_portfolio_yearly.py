"""年度展示使用各自资金曲线，失败只影响派生分析，不覆盖原结果。"""

from copy import deepcopy
import importlib

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from quantlab.portfolio import (
    PortfolioBacktestError, PortfolioConfig, analyze_portfolio_result,
    run_equal_weight_benchmark, run_factor_portfolio,
)


def ui():
    assert importlib.util.find_spec("quantlab.ui.portfolio_yearly") is not None, "年度展示尚未实现"
    return importlib.import_module("quantlab.ui.portfolio_yearly")


@pytest.fixture
def analyses(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=300)
    strategy = analyze_portfolio_result(run_factor_portfolio(
        preview, factor_name="momentum_20", config=PortfolioConfig(top_n=1)))
    benchmark = run_equal_weight_benchmark(preview, factor_name="momentum_20")
    return strategy, benchmark


def annual_tables(app):
    return [item.value for item in app.dataframe if "策略区间收益" in item.value.columns]


def show(strategy, benchmark=None, *, status="complete", historical=False):
    app = AppTest.from_string("""
import streamlit as st
from quantlab.ui.portfolio import render_portfolio_result
render_portfolio_result(st.session_state.strategy, benchmark=st.session_state.benchmark,
    benchmark_status=st.session_state.status, historical=st.session_state.historical)
""", default_timeout=20)
    app.session_state["strategy"] = strategy
    app.session_state["benchmark"] = benchmark
    app.session_state["status"] = status
    app.session_state["historical"] = historical
    return app.run()


def test_yearly_page_uses_raw_equity_and_percentage_points_without_mutation(analyses):
    strategy, benchmark = analyses
    before = deepcopy(strategy)
    table = ui().portfolio_yearly_table(strategy, benchmark)
    equity = strategy.backtest.equity_curve
    base = equity.loc[equity.date.dt.year.eq(2025), "equity"].iloc[-1]
    end = equity.equity.iloc[-1]
    benchmark_curve = benchmark.backtest.equity_curve
    benchmark_base = benchmark_curve.loc[benchmark_curve.date.dt.year.eq(2025), "equity"].iloc[-1]
    benchmark_return = benchmark_curve.equity.iloc[-1] / benchmark_base - 1
    assert table["年份"].tolist() == [2025, 2026]
    assert table.iloc[1]["策略区间收益"] == pytest.approx(end / base - 1)
    assert table.iloc[1]["基准区间收益"] == pytest.approx(benchmark_return)
    assert table.iloc[1]["收益差（百分点）"] == pytest.approx(100 * (end / base - 1 - benchmark_return))
    pd.testing.assert_frame_equal(strategy.backtest.equity_curve, before.backtest.equity_curve)
    pd.testing.assert_series_equal(strategy.performance.drawdown, before.performance.drawdown)
    assert strategy.backtest.metadata == before.backtest.metadata
    app = show(strategy, benchmark)
    assert not app.exception and not app.error
    assert len(annual_tables(app)) == 1
    assert len(app.metric) == 7, "年度表不替代全区间原指标"
    assert any("不年化" in caption.value for caption in app.caption)


@pytest.mark.parametrize("status,explanation", [("not_run", "未运行"), ("unavailable", "不可用")])
def test_missing_benchmark_keeps_strategy_and_has_no_fake_zero(analyses, status, explanation):
    strategy, _ = analyses
    app = show(strategy, status=status)
    assert not app.exception
    assert len(annual_tables(app)) == 1
    table = annual_tables(app)[0]
    assert table["策略区间收益"].notna().all()
    assert table["基准区间收益"].isna().all() and table["收益差（百分点）"].isna().all()
    assert any(explanation in item.value and "年度" in item.value for item in app.info)


def test_historical_derived_label_is_separate_from_saved_metrics(analyses):
    app = show(*analyses, historical=True)
    assert not app.exception and annual_tables(app)
    assert any("当前版本" in c.value and "已保存资金曲线" in c.value and "不改变" in c.value for c in app.caption)


@pytest.mark.parametrize("change", ["calendar", "cost", "anchor", "dates", "equity"])
def test_mismatched_benchmark_is_not_silently_aligned(analyses, change):
    strategy, benchmark = analyses
    if change == "calendar":
        benchmark.backtest.metadata["data"]["calendar"]["dates"].pop()
    elif change == "cost":
        benchmark.backtest.metadata["strategy"]["config"]["backtest"]["slippage_rate"] = .9
    elif change == "anchor":
        benchmark.backtest.metadata["strategy"]["signal_anchor_position"] += 1
    elif change == "dates":
        benchmark.backtest.equity_curve.drop(index=0, inplace=True)
    else:
        benchmark.backtest.equity_curve.loc[1, "equity"] = float("nan")
    with pytest.raises(PortfolioBacktestError):
        ui().portfolio_yearly_table(strategy, benchmark)


def test_annual_error_does_not_hide_original_metrics_or_private_message(analyses, monkeypatch):
    module = ui()
    def bad(*args, **kwargs):
        raise ValueError("secret-token /private/person")
    monkeypatch.setattr(module, "portfolio_yearly_table", bad)
    app = show(*analyses)
    assert not app.exception and len(app.metric) == 7
    assert not annual_tables(app)
    assert any("年度" in item.value for item in app.warning)
    assert "secret-token" not in str(app.warning)


def test_short_cash_year_is_zero_only_when_return_interval_exists(make_portfolio_preview):
    for periods in (1, 3):
        preview, _ = make_portfolio_preview(periods=periods)
        strategy = analyze_portfolio_result(run_factor_portfolio(
            preview, factor_name="momentum_20", config=PortfolioConfig(top_n=1)))
        app = show(strategy, status="not_run")
        assert not app.exception and len(annual_tables(app)) == 1
        row = annual_tables(app)[0].iloc[0]
        if periods == 1:
            assert pd.isna(row["策略区间收益"]) and pd.isna(row["年内最大回撤"])
        else:
            assert row["策略区间收益"] == row["年内最大回撤"] == 0
