"""全池一次建仓基准：用手算账本和真实合成快照验证，而非只比较同源公式。"""

from copy import deepcopy
import json
import shutil

import pandas as pd
import pytest

from quantlab.config.settings import AnalyticsConfig, BacktestConfig
from quantlab.portfolio import PortfolioBacktestError, PortfolioConfig, analyze_portfolio_result, run_factor_portfolio


def run(preview, **kwargs):
    from quantlab.portfolio import run_equal_weight_benchmark
    return run_equal_weight_benchmark(preview, factor_name=kwargs.pop("factor_name", "momentum_20"), **kwargs)


def test_one_purchase_all_pool_at_fixed_anchor_and_manual_cost_ledger(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=45)
    costs = BacktestConfig(commission_rate=.01, slippage_rate=.01, sell_tax_rate=.1)
    analysis = run(preview, backtest_config=costs)
    result = analysis.backtest
    dates = pd.to_datetime(preview.alignment.calendar.dates)
    assert result.trades.symbol.tolist() == ["159915", "510300"]
    assert result.trades.quantity.tolist() == [200, 400]
    assert result.trades.side.eq("BUY").all()
    assert result.trades.execution_date.eq(dates[21]).all()
    assert result.trades.signal_date.eq(dates[20]).all()
    assert len(result.decisions) == 1
    assert result.targets.target_weight.eq(.5).all()
    curve = result.equity_curve
    assert curve.date.tolist() == dates.tolist()
    assert curve.equity.iloc[:21].eq(100000).all()
    assert curve.iloc[21].cash == pytest.approx(16249.79)
    assert curve.iloc[21].equity == pytest.approx(98349.79)
    assert curve.iloc[-1].equity == pytest.approx(100649.79)
    assert curve.iloc[-1].cumulative_fees_taxes == pytest.approx(829.21)
    assert curve.iloc[-1].cumulative_slippage == pytest.approx(821)
    assert result.trades.tax.eq(0).all()  # 不平仓，卖出税率不应产生税款。
    assert analysis.performance.cumulative_return == pytest.approx(.0064979)


@pytest.mark.parametrize("name,lookback", [("momentum_20", 20), ("momentum_60", 60),
    ("momentum_120", 120), ("volatility_20", 20), ("volatility_60", 60)])
def test_anchor_comes_from_selected_factor_not_actual_strategy_trade(make_portfolio_preview, name, lookback):
    preview, _ = make_portfolio_preview(periods=lookback + 3, gaps={"510300": [2]})
    strategy = run_factor_portfolio(preview, factor_name=name, config=PortfolioConfig(top_n=2))
    assert strategy.trades.empty  # 第一次因子窗口缺失，不影响全池基准的固定计划。
    benchmark = run(preview, factor_name=name).backtest
    assert benchmark.trades.execution_date.eq(pd.Timestamp(preview.alignment.calendar.dates[lookback + 1])).all()
    assert set(benchmark.trades.symbol) == {"159915", "510300"}


@pytest.mark.parametrize("periods", [1, 20, 21])
def test_no_execution_date_keeps_full_cash_and_explains(make_portfolio_preview, periods):
    preview, _ = make_portfolio_preview(periods=periods)
    result = run(preview)
    assert len(result.backtest.equity_curve) == periods
    assert result.backtest.equity_curve.equity.eq(100000).all()
    assert result.backtest.trades.empty
    assert result.backtest.metadata["strategy"]["planned_execution_date"] is None
    assert any("未到计划建仓日" in text for text in result.backtest.metadata["warnings"])
    if periods == 1:
        assert result.performance.cumulative_return is None


@pytest.mark.parametrize("kwargs,reason", [({"gaps": {"159915": [21]}}, "missing_open"),
    ({"gaps": {"159915": [24]}}, "missing_close"),
    ({"zero_volume": {"159915": [21]}}, "zero_volume")])
def test_bad_required_prices_or_execution_volume_fail_no_exclusion(make_portfolio_preview, kwargs, reason):
    preview, _ = make_portfolio_preview(periods=25, **kwargs)
    with pytest.raises(PortfolioBacktestError) as error:
        run(preview)
    assert error.value.reason == reason
    assert "159915" in error.value.symbols


def test_identity_independent_snapshot_no_fake_factor_and_same_assumptions(make_portfolio_preview):
    preview, source = make_portfolio_preview(periods=25, supplied=True)
    before = deepcopy(preview)
    shutil.rmtree(source)
    analytics = AnalyticsConfig(periods_per_year=250, annual_risk_free_rate=.02)
    result = run(preview, analytics_config=analytics)
    meta = result.backtest.metadata
    assert meta["schema_version"] == "portfolio-benchmark-v1"
    assert meta["strategy"]["name"] == "equal_weight_buy_and_hold"
    assert meta["strategy"]["signal_anchor_position"] == 20
    assert meta["strategy"]["execution_rule"] == "L+1_open_once"
    assert meta["data"]["preview_sha256"] == preview.preview_sha256
    assert meta["data"]["symbols"] == ["159915", "510300"]
    assert meta["analytics_config"] == {"periods_per_year": 250, "annual_risk_free_rate": .02}
    assert "factor_definition" not in meta and "factor_sha256" not in meta["data"]
    assert result.backtest.factor_data is None
    assert str(source) not in json.dumps(meta, ensure_ascii=False, allow_nan=False)
    assert result.backtest.targets.factor_value.isna().all()  # 引擎占位分值不作为研究因子发布。
    result.backtest.input_data.loc[0, "close"] = 123
    pd.testing.assert_frame_equal(preview.alignment.data, before.alignment.data)
    assert preview.sources == before.sources
    with pytest.raises(PortfolioBacktestError):
        analyze_portfolio_result(result.backtest)  # 不能冒充因子策略通过原有边界。


def test_invalid_inputs_are_not_repaired(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=25)
    with pytest.raises(PortfolioBacktestError, match="因子"):
        run(preview, factor_name="unknown")
    preview.alignment.data.loc[0, "volume"] += 1
    with pytest.raises(PortfolioBacktestError) as error:
        run(preview)
    assert error.value.reason == "invalid_research_input"


def test_unaffordable_pool_remains_cash_without_delayed_entry(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=45)
    result = run(preview, backtest_config=BacktestConfig(initial_cash=1)).backtest
    assert result.trades.empty
    assert len(result.decisions) == 1
    assert result.equity_curve.equity.eq(1).all()
