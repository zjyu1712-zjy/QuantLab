"""小型固定网格必须等同于逐次独立回测；全部使用合成行情。"""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd
import pytest

from quantlab.config.settings import AnalyticsConfig, BacktestConfig
from quantlab.data.catalog import scan_market_catalog
from quantlab.data.importer import import_market_csv
from quantlab.data.research_sets import DatasetSelection, build_research_set
from quantlab.portfolio import PortfolioConfig, analyze_portfolio_result, run_factor_portfolio
from quantlab.portfolio.types import PortfolioBacktestError
from quantlab.portfolio.sensitivity import run_parameter_sensitivity, METRIC_FIELDS


@pytest.mark.parametrize("count", [2, 3, 4])
def test_fixed_grid_caps_at_three_assets(count, make_portfolio_preview):
    preview, root = make_portfolio_preview(periods=25)
    for symbol in ("510500", "512000")[:max(0, count - 2)]:
        frame = pd.read_csv(root / "510300.csv")
        frame["symbol"] = symbol
        path = root / f"{symbol}.csv"
        frame.to_csv(path, index=False)
        import_market_csv(path, root / "raw", root / "processed",
                          imported_at=datetime(2025, 9, 1, tzinfo=timezone.utc))
    records = scan_market_catalog(root / "processed", root / "raw").records[:count]
    preview = build_research_set(tuple(DatasetSelection(r, "none") for r in records),
                                 start=preview.start, end=preview.end)
    table = run_parameter_sensitivity(preview, factor_name="momentum_20")
    assert list(zip(table.top_n, table.rebalance_interval)) == [
        (n, k) for n in range(1, min(count, 3) + 1) for k in (5, 10, 20)]
    assert table.status.eq("complete").all()


@pytest.mark.parametrize("factor,ascending", [("momentum_20", False), ("volatility_20", True),
                                               ("momentum_60", None)])
def test_every_cell_matches_independent_run(factor, ascending, make_portfolio_preview):
    preview, root = make_portfolio_preview(periods=65)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    original = deepcopy(preview)
    costs = BacktestConfig(initial_cash=123456., commission_rate=.001,
                           minimum_commission=5., slippage_rate=.002, sell_tax_rate=.001, lot_size=10)
    analytics = AnalyticsConfig(periods_per_year=240, annual_risk_free_rate=.02)
    table = run_parameter_sensitivity(preview, factor_name=factor, ascending=ascending,
                                      backtest_config=costs, analytics_config=analytics)
    for row in table.itertuples():
        expected = analyze_portfolio_result(run_factor_portfolio(
            preview, factor_name=factor, config=PortfolioConfig(
                top_n=row.top_n, rebalance_interval=row.rebalance_interval, ascending=ascending, backtest=costs)),
            analytics_config=analytics)
        for name in METRIC_FIELDS:
            value = getattr(expected.performance, name)
            assert pd.isna(getattr(row, name)) if value is None else getattr(row, name) == value
        assert row.trade_count == len(expected.backtest.trades)
        assert row.unavailable_reasons == expected.performance.unavailable_reasons
        assert row.failure_reason == ""
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    assert preview.preview_sha256 == original.preview_sha256
    pd.testing.assert_frame_equal(preview.alignment.data, original.alignment.data)


@pytest.mark.parametrize("periods,cash", [(1, 100000.), (10, 100000.), (45, .01)])
def test_zero_trades_are_success_not_failure_and_missing_is_not_zero(periods, cash, make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=periods)
    table = run_parameter_sensitivity(preview, factor_name="momentum_20",
                                      backtest_config=BacktestConfig(initial_cash=cash))
    assert table.status.eq("complete").all() and table.trade_count.eq(0).all()
    assert table.sharpe_ratio.isna().all()
    assert all("sharpe_ratio" in reasons for reasons in table.unavailable_reasons)
    assert table.max_drawdown.eq(0.).all()
    if periods == 1:
        assert table.cumulative_return.isna().all()
    else:
        assert table.cumulative_return.eq(0.).all()


def test_real_one_asset_execution_failure_does_not_hide_other_cells(make_portfolio_preview):
    # 动量最高的是 510300；159915 的首个执行日零成交量只阻断 N=2。
    preview, _ = make_portfolio_preview(periods=45, zero_volume={"159915": [21]})
    table = run_parameter_sensitivity(preview, factor_name="momentum_20")
    assert table.loc[table.top_n.eq(1), "status"].eq("complete").all()
    failed = table.loc[table.top_n.eq(2)]
    assert failed.status.eq("failed").all() and failed.failure_reason.eq("zero_volume").all()
    assert failed[list(METRIC_FIELDS) + ["trade_count"]].isna().all().all()


@pytest.mark.parametrize("error", [RuntimeError("/private/account/secret.csv TOKEN"),
    PortfolioBacktestError("SECRET", reason="/private/secret", symbols=("TOKEN",))])
def test_unexpected_failure_is_safe_and_next_cell_still_runs(error, monkeypatch, make_portfolio_preview):
    from quantlab.portfolio import sensitivity
    preview, _ = make_portfolio_preview(periods=25)
    real = sensitivity.run_factor_portfolio
    def fail_first(*args, **kwargs):
        config = kwargs["config"]
        if (config.top_n, config.rebalance_interval) == (1, 5):
            raise error
        return real(*args, **kwargs)
    monkeypatch.setattr(sensitivity, "run_factor_portfolio", fail_first)
    table = run_parameter_sensitivity(preview, factor_name="momentum_20")
    assert table.iloc[0].status == "failed" and table.iloc[0].failure_reason == "unavailable"
    assert table.iloc[1:].status.eq("complete").all()
    assert "SECRET" not in repr(table.to_dict()) and "private" not in repr(table.to_dict())


@pytest.mark.parametrize("kwargs,reason", [({"ascending": 1}, "invalid_config"),
    ({"backtest_config": None}, "invalid_config"),
    ({"analytics_config": AnalyticsConfig(periods_per_year=True)}, "invalid_analytics_config")])
def test_invalid_shared_settings_fail_before_running(kwargs, reason, make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=25)
    with pytest.raises(PortfolioBacktestError) as caught:
        run_parameter_sensitivity(preview, factor_name="momentum_20", **kwargs)
    assert caught.value.reason == reason


def test_invalid_preview_rejected_and_tampering_never_produces_success(make_portfolio_preview):
    with pytest.raises(PortfolioBacktestError):
        run_parameter_sensitivity(None, factor_name="momentum_20")
    preview, _ = make_portfolio_preview(periods=25)
    table = run_parameter_sensitivity(replace(preview, preview_sha256="0" * 64), factor_name="momentum_20")
    assert table.status.eq("failed").all()


def test_analysis_failure_discards_metrics_but_continues(monkeypatch, make_portfolio_preview):
    from quantlab.portfolio import sensitivity
    preview, _ = make_portfolio_preview(periods=25)
    real = sensitivity.analyze_portfolio_result
    def fail_second(result, **kwargs):
        config = result.metadata["strategy"]["config"]
        if (config["top_n"], config["rebalance_interval"]) == (1, 10):
            raise PortfolioBacktestError("private path", reason="numeric_range")
        return real(result, **kwargs)
    monkeypatch.setattr(sensitivity, "analyze_portfolio_result", fail_second)
    table = run_parameter_sensitivity(preview, factor_name="momentum_20")
    assert table.iloc[1].status == "failed" and table.iloc[1].failure_reason == "numeric_range"
    assert table.iloc[1][list(METRIC_FIELDS) + ["trade_count"]].isna().all()
    assert table.iloc[[0, 2, 3, 4, 5]].status.eq("complete").all()


@pytest.mark.parametrize("factor", [None, "unknown", " momentum_20", []])
def test_invalid_factor_rejected(factor, make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=25)
    with pytest.raises(PortfolioBacktestError) as caught:
        run_parameter_sensitivity(preview, factor_name=factor)
    assert caught.value.reason == "invalid_factor"
