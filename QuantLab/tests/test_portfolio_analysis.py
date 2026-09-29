"""组合指标适配：完整区间、可计算边界和数值错误互不混淆。"""

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math

import numpy as np
import pandas as pd
import pytest

from quantlab.analytics.performance import analyze_performance
from quantlab.config.settings import AnalyticsConfig, BacktestConfig
from quantlab.data.catalog import scan_market_catalog
from quantlab.data.importer import import_market_csv
from quantlab.data.research_sets import DatasetSelection, build_research_set
from quantlab.portfolio import PortfolioBacktestError, PortfolioConfig, run_factor_portfolio


def analyze(result, **kwargs):
    from quantlab.portfolio import analyze_portfolio_result

    return analyze_portfolio_result(result, **kwargs)


def cash_result(make_portfolio_preview, periods=10):
    preview, _ = make_portfolio_preview(periods=periods)
    return run_factor_portfolio(preview, factor_name="momentum_20", config=PortfolioConfig(top_n=2))


def assert_result_unchanged(actual, before):
    for field in ("decisions", "targets", "orders", "trades", "positions", "equity_curve",
                  "input_data", "factor_data"):
        pd.testing.assert_frame_equal(getattr(actual, field), getattr(before, field))
    assert actual.metadata == before.metadata


def price_path_result(tmp_path, values):
    """真实导入两只同价 ETF，位置 21 开盘买一份，无费用；不伪造运行结果。"""
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    dates = pd.date_range("2025-01-02", periods=len(values), freq="B")
    for symbol in ("159915", "510300"):
        frame = pd.DataFrame({"date": dates, "symbol": symbol, "open": values,
                              "high": values, "low": values, "close": values, "volume": 1000.0})
        path = tmp_path / f"{symbol}.csv"
        frame.to_csv(path, index=False)
        import_market_csv(path, raw, processed, imported_at=datetime(2025, 9, 1, tzinfo=timezone.utc))
    records = scan_market_catalog(processed, raw).records
    preview = build_research_set(tuple(DatasetSelection(record, "none") for record in records),
                                start=dates[0].date(), end=dates[-1].date())
    return run_factor_portfolio(preview, factor_name="momentum_20", config=PortfolioConfig(
        top_n=1, backtest=BacktestConfig(initial_cash=100, lot_size=1, commission_rate=0,
                                        minimum_commission=0, slippage_rate=0, sell_tax_rate=0)))


def test_cash_path_is_valid_but_sharpe_is_undefined(make_portfolio_preview):
    result = cash_result(make_portfolio_preview)
    actual = analyze(result)
    from quantlab.portfolio import PortfolioAnalysis, PortfolioPerformance

    assert isinstance(actual, PortfolioAnalysis)
    assert isinstance(actual.performance, PortfolioPerformance)
    p = actual.performance
    assert p.cumulative_return == p.annualized_return == p.annualized_volatility == p.max_drawdown == 0
    assert p.sharpe_ratio is None
    assert set(p.unavailable_reasons) == {"sharpe_ratio"}
    assert "波动" in p.unavailable_reasons["sharpe_ratio"]
    assert p.daily_returns.eq(0).all() and len(p.daily_returns) == 9
    assert p.net_value.eq(1).all() and p.drawdown.eq(0).all()
    assert result.metadata["analytics_config"] is None
    assert actual.backtest.metadata["analytics_config"] == {"periods_per_year": 252, "annual_risk_free_rate": 0.0}


@pytest.mark.parametrize("periods,unavailable", [
    (1, {"cumulative_return", "annualized_return", "annualized_volatility", "sharpe_ratio"}),
    (2, {"annualized_volatility", "sharpe_ratio"}),
    (3, {"sharpe_ratio"}),
])
def test_short_samples_have_per_metric_reasons(make_portfolio_preview, periods, unavailable):
    result = cash_result(make_portfolio_preview, periods)
    p = analyze(result).performance
    assert set(p.unavailable_reasons) == unavailable
    for field in unavailable:
        assert getattr(p, field) is None
        assert p.unavailable_reasons[field]
    assert p.max_drawdown == 0 and p.net_value.tolist() == [1.0] * periods
    assert p.drawdown.tolist() == [0.0] * periods
    assert p.daily_returns.name == "daily_return"
    assert p.daily_returns.dtype == np.dtype("float64")
    assert len(p.daily_returns) == periods - 1
    assert p.daily_returns.index.equals(pd.DatetimeIndex(result.equity_curve.date.iloc[1:], name="date"))
    if periods >= 2:
        assert p.cumulative_return == p.annualized_return == 0


@pytest.mark.parametrize("periods_per_year,rate", [(252, 0.0), (12, 0.04), (1, -0.5)])
def test_normal_path_matches_existing_metrics_without_changing_run(make_portfolio_preview, periods_per_year, rate):
    preview, _ = make_portfolio_preview(periods=45)
    result = run_factor_portfolio(preview, factor_name="momentum_20", config=PortfolioConfig(top_n=1))
    before = deepcopy(result)
    config = AnalyticsConfig(periods_per_year=periods_per_year, annual_risk_free_rate=rate)
    actual = analyze(result, analytics_config=config)
    expected = analyze_performance(result.equity_curve.set_index("date").equity,
                                   periods_per_year=periods_per_year, annual_risk_free_rate=rate)
    for field in ("daily_returns", "net_value", "drawdown"):
        pd.testing.assert_series_equal(getattr(actual.performance, field), getattr(expected, field))
    for field in ("cumulative_return", "annualized_return", "annualized_volatility", "max_drawdown", "sharpe_ratio"):
        assert getattr(actual.performance, field) == pytest.approx(getattr(expected, field))
    assert actual.performance.unavailable_reasons == {}
    assert actual.backtest.metadata["analytics_config"] == asdict(config)
    assert {k: v for k, v in actual.backtest.metadata.items() if k != "analytics_config"} == {
        k: v for k, v in result.metadata.items() if k != "analytics_config"}
    assert_result_unchanged(result, before)


def test_first_cost_and_all_warmup_days_are_included(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=23)
    result = run_factor_portfolio(preview, factor_name="momentum_20", config=PortfolioConfig(
        top_n=1, backtest=BacktestConfig(lot_size=100, commission_rate=0, minimum_commission=10, slippage_rate=0)))
    assert result.equity_curve.equity.tolist() == [100000.0] * 21 + [99990.0, 100440.0]
    p = analyze(result).performance
    assert len(p.daily_returns) == 22
    assert p.daily_returns.iloc[:20].eq(0).all()
    assert p.daily_returns.iloc[20] == pytest.approx(-0.0001)
    assert p.daily_returns.iloc[21] == pytest.approx(450 / 99990)
    assert p.cumulative_return == pytest.approx(0.0044)
    assert p.annualized_return == pytest.approx(1.0044 ** (252 / 22) - 1)
    assert p.max_drawdown == pytest.approx(-0.0001)
    assert p.net_value.iloc[0] == 1.0


@pytest.mark.parametrize("delta,undefined", [(2e-15, True), (1e-14, False)])
def test_sharpe_uses_daily_std_threshold_not_annualized_vol(tmp_path, delta, undefined):
    result = price_path_result(tmp_path, [100.0] * 22 + [100.0 * (1 + delta)])
    p = analyze(result).performance
    assert p.annualized_volatility > 1e-15
    assert (p.sharpe_ratio is None) is undefined
    if undefined:
        assert set(p.unavailable_reasons) == {"sharpe_ratio"}
    else:
        assert math.isfinite(p.sharpe_ratio) and not p.unavailable_reasons


def test_results_and_reanalysis_are_independent(make_portfolio_preview):
    result = cash_result(make_portfolio_preview)
    before = deepcopy(result)
    first = analyze(result)
    first_before = deepcopy(first)
    second = analyze(first.backtest, analytics_config=AnalyticsConfig(periods_per_year=12, annual_risk_free_rate=0.02))
    assert_result_unchanged(first.backtest, first_before.backtest)
    assert second.backtest.metadata["run_id"] == result.metadata["run_id"]
    assert second.backtest.metadata["analytics_config"]["periods_per_year"] == 12
    for field in ("decisions", "targets", "orders", "trades", "positions", "equity_curve", "input_data", "factor_data"):
        table = getattr(second.backtest, field)
        table.drop(index=table.index, inplace=True)
    second.backtest.metadata["data"]["calendar"]["dates"].clear()
    second.backtest.metadata["strategy"]["config"]["backtest"]["initial_cash"] = 1
    second.performance.daily_returns.iloc[:] = 123
    second.performance.net_value.iloc[:] = 123
    second.performance.drawdown.iloc[:] = 123
    second.performance.unavailable_reasons.clear()
    assert_result_unchanged(result, before)
    assert_result_unchanged(first.backtest, first_before.backtest)
    for field in ("daily_returns", "net_value", "drawdown"):
        pd.testing.assert_series_equal(getattr(first.performance, field), getattr(first_before.performance, field))
    assert first.performance.unavailable_reasons == first_before.performance.unavailable_reasons


@pytest.mark.parametrize("mutation", [
    "none", "metadata", "schema", "status", "calendar", "axis_empty", "axis_duplicate", "axis_invalid",
    "short", "duplicate", "reverse", "empty", "date_missing", "timezone", "intraday", "string_dates",
    "nan", "inf", "zero", "negative", "bool_equity", "text_equity", "missing_equity", "initial_cash",
    "input_data", "factor_data", "input_hash", "factor_hash", "input_snapshot", "factor_snapshot",
])
def test_invalid_or_incomplete_result_is_not_a_successful_analysis(make_portfolio_preview, mutation):
    result = cash_result(make_portfolio_preview, 3)
    if mutation == "none":
        result = None
    elif mutation == "metadata":
        result = replace(result, metadata=None)
    elif mutation == "schema":
        result.metadata["schema_version"] = "backtest-run-v1"
    elif mutation == "status":
        result.metadata["status"] = "failed"
    elif mutation == "calendar":
        del result.metadata["data"]["calendar"]
    elif mutation.startswith("axis_"):
        result.metadata["data"]["calendar"]["dates"] = {
            "axis_empty": [], "axis_duplicate": ["2025-01-02"] * 3,
            "axis_invalid": ["not-a-date"],
        }[mutation]
    elif mutation in ("short", "duplicate", "reverse", "empty"):
        positions = {"short": [0, 2], "duplicate": [0, 1, 1], "reverse": [2, 1, 0], "empty": []}[mutation]
        result = replace(result, equity_curve=result.equity_curve.iloc[positions].copy())
    elif mutation == "date_missing":
        result.equity_curve.loc[1, "date"] = pd.NaT
    elif mutation == "timezone":
        result.equity_curve["date"] = result.equity_curve.date.dt.tz_localize("UTC")
    elif mutation == "intraday":
        result.equity_curve.loc[1, "date"] += pd.Timedelta(hours=1)
    elif mutation == "string_dates":
        result.equity_curve["date"] = result.equity_curve.date.astype(str)
    elif mutation in ("nan", "inf", "zero", "negative"):
        result.equity_curve.loc[1, "equity"] = {"nan": np.nan, "inf": np.inf, "zero": 0, "negative": -1}[mutation]
    elif mutation == "bool_equity":
        result.equity_curve["equity"] = True
    elif mutation == "text_equity":
        result.equity_curve["equity"] = "100000"
    elif mutation == "missing_equity":
        result.equity_curve.drop(columns="equity", inplace=True)
    elif mutation == "initial_cash":
        result.equity_curve.loc[0, "equity"] = 99000
    elif mutation in ("input_hash", "factor_hash"):
        result.metadata["data"]["market_sha256" if mutation == "input_hash" else "factor_sha256"] = "0" * 64
    elif mutation == "input_snapshot":
        result.input_data.loc[0, "volume"] += 1
    elif mutation == "factor_snapshot":
        result.factor_data.loc[0, "factor_value"] = 1.0
    else:
        result = replace(result, **{mutation: None})
    before = deepcopy(result)
    with pytest.raises(PortfolioBacktestError) as caught:
        analyze(result)
    assert caught.value.reason == "invalid_portfolio_result"
    if result is not None and result.metadata is not None and mutation not in ("input_data", "factor_data"):
        assert_result_unchanged(result, before)


@pytest.mark.parametrize("field,value", [
    ("periods_per_year", 0), ("periods_per_year", -1), ("periods_per_year", True),
    ("periods_per_year", 252.5), ("periods_per_year", np.nan), ("periods_per_year", np.inf),
    ("periods_per_year", "252"), ("periods_per_year", 10 ** 400),
    ("annual_risk_free_rate", -1), ("annual_risk_free_rate", -2), ("annual_risk_free_rate", True),
    ("annual_risk_free_rate", np.nan), ("annual_risk_free_rate", np.inf), ("annual_risk_free_rate", "0"),
])
def test_invalid_analytics_parameters_are_not_hidden_by_short_sample(make_portfolio_preview, field, value):
    result = cash_result(make_portfolio_preview, 1)
    config = AnalyticsConfig()
    object.__setattr__(config, field, value)
    with pytest.raises(PortfolioBacktestError) as caught:
        analyze(result, analytics_config=config)
    assert caught.value.reason == "invalid_analytics_config"


@pytest.mark.parametrize("config", [None, {}, 252])
def test_wrong_config_type_fails(make_portfolio_preview, config):
    with pytest.raises(PortfolioBacktestError) as caught:
        analyze(cash_result(make_portfolio_preview), analytics_config=config)
    assert caught.value.reason == "invalid_analytics_config"


def test_numpy_parameters_have_serializable_record(make_portfolio_preview):
    result = analyze(cash_result(make_portfolio_preview), analytics_config=AnalyticsConfig(
        periods_per_year=np.int64(12), annual_risk_free_rate=np.float64(0.01)))
    assert json.loads(json.dumps(result.backtest.metadata, allow_nan=False))["analytics_config"] == {
        "periods_per_year": 12, "annual_risk_free_rate": 0.01}


@pytest.mark.parametrize("damage", ["cash_overflow", "input_columns", "factor_columns"])
def test_malformed_result_has_structured_error(make_portfolio_preview, damage):
    result = cash_result(make_portfolio_preview, 3)
    if damage == "cash_overflow":
        result.metadata["strategy"]["config"]["backtest"]["initial_cash"] = 10 ** 400
    else:
        field = "input_data" if damage == "input_columns" else "factor_data"
        table = getattr(result, field)
        result = replace(result, **{field: pd.concat([table, table[["date"]]], axis=1)})
    with pytest.raises(PortfolioBacktestError) as caught:
        analyze(result)
    assert caught.value.reason == "invalid_portfolio_result"


def test_equity_row_labels_are_not_date_values(make_portfolio_preview):
    result = cash_result(make_portfolio_preview)
    result.equity_curve.index = pd.MultiIndex.from_arrays([[0] * 10, range(10)], names=["date", "equity"])
    before = deepcopy(result)
    actual = analyze(result)
    assert len(actual.performance.net_value) == 10
    assert actual.performance.net_value.index.equals(pd.DatetimeIndex(result.equity_curve.date, name="date"))
    assert_result_unchanged(result, before)


def test_extreme_risk_free_mean_overflow_is_not_a_finite_sharpe(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=23)
    result = run_factor_portfolio(preview, factor_name="momentum_20", config=PortfolioConfig(top_n=1))
    with pytest.raises(PortfolioBacktestError, match="数值") as caught:
        analyze(result, analytics_config=AnalyticsConfig(periods_per_year=1, annual_risk_free_rate=1e308))
    assert caught.value.reason == "numeric_range"


@pytest.mark.parametrize("values,annual_periods", [
    ([100.0] * 22 + [1e20], 1000),  # 可精确表示的行情，复合年化仍可溢出。
    ([100.0] * 22 + [1e-200, 100.0], 1),  # 大幅下跌后反弹，有限收益的方差溢出。
])
def test_finite_equity_numeric_overflow_is_error_not_unavailable(tmp_path, values, annual_periods):
    result = price_path_result(tmp_path, values)
    assert np.isfinite(result.equity_curve.equity).all()
    before = deepcopy(result)
    with pytest.raises(PortfolioBacktestError, match="数值") as caught:
        analyze(result, analytics_config=AnalyticsConfig(periods_per_year=annual_periods))
    assert caught.value.reason == "numeric_range"
    assert_result_unchanged(result, before)


def test_legacy_uniform_entry_still_raises_on_undefined_sharpe():
    # 第一阶段既有行为：不因组合适配层新增 None 而被放宽。
    with pytest.raises(ValueError, match="无法计算夏普比率"):
        analyze_performance(pd.Series([100.0, 100.0, 100.0]))
