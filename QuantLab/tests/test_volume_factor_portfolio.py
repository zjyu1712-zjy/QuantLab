"""成交量相对变化的真实导入、筛选、执行及档案链契约。"""

from copy import deepcopy
from datetime import datetime, timezone
from itertools import count

import numpy as np
import pandas as pd
import pytest

from quantlab.data.catalog import scan_market_catalog
from quantlab.data.importer import import_market_csv
from quantlab.data.research_sets import DatasetSelection, build_research_set
from quantlab.data.research_storage import load_research_set, save_research_set
from quantlab.factors import create_default_registry, run_research_factor
from quantlab.portfolio import PortfolioConfig, run_factor_portfolio, run_equal_weight_benchmark
from quantlab.portfolio.selection import build_rebalance_schedule
from quantlab.ui.factors import top_n_factor_preview
from test_bias_portfolio import (
    test_archive_preserves_new_and_old_definitions_without_recomputing as check_archive_roundtrip,
)

NAME = "volume_change_20"


@pytest.fixture
def volume_preview_factory(tmp_path):
    """独立成交量样本：下跌 ETF 放量、上涨 ETF 缩量，不修改旧夹具。"""
    sequence = count()

    def make(*, periods=22, gap=False, zero_baseline=False, current_zero=False, scale=1.0):
        root = tmp_path / f"volume_sources_{next(sequence)}"
        root.mkdir()
        dates = pd.date_range("2025-01-02", periods=periods, freq="B")
        for symbol in ("159915", "510300"):
            positions = np.arange(periods)
            close = 200 - positions * .5 if symbol == "159915" else 100 + positions * .5
            volume = 100 + positions * 10.0 if symbol == "159915" else 200 + positions * 20.0
            if zero_baseline and symbol == "159915":
                volume[:20] = 0
            if periods > 20:
                volume[20] = 780 if symbol == "159915" else (0 if current_zero else 195)
            if symbol == "159915":
                volume *= scale
            frame = pd.DataFrame({"date": dates, "symbol": symbol, "open": close,
                                  "high": close, "low": close, "close": close, "volume": volume})
            if gap and symbol == "159915":
                frame = frame.drop(index=5)
            path = root / f"{symbol}.csv"
            frame.to_csv(path, index=False)
            import_market_csv(path, root / "raw", root / "processed",
                              imported_at=datetime(2025, 9, 1, tzinfo=timezone.utc))
        scan = scan_market_catalog(root / "processed", root / "raw")
        assert not scan.issues
        preview = build_research_set(tuple(DatasetSelection(record, "none") for record in scan.records),
                                     start=dates[0].date(), end=dates[-1].date())
        saved = save_research_set(preview, tmp_path / "volume_sets", name="成交量研究样本")
        return load_research_set(saved.directory.parent, saved.set_id).preview, saved

    return make


def volume_factor(preview):
    assert NAME in create_default_registry().names(), "成交量变化因子尚未注册"
    return run_research_factor(preview, NAME)


def test_real_snapshot_excludes_today_from_mean_and_preserves_input(volume_preview_factory):
    preview, _ = volume_preview_factory()
    before = deepcopy(preview)
    result = volume_factor(preview)
    first = result.data.loc[result.data.date.eq(pd.Timestamp(result.calendar.dates[20]))].set_index("symbol")
    # 前20期均量分别为195和390；当期不得进入分母。
    assert first.loc["159915", "factor_value"] == pytest.approx(3.0)
    assert first.loc["510300", "factor_value"] == pytest.approx(-.5)
    pd.testing.assert_frame_equal(preview.alignment.data, before.alignment.data)
    assert preview.preview_sha256 == before.preview_sha256
    scaled, _ = volume_preview_factory(scale=100.0)
    pd.testing.assert_frame_equal(result.data, volume_factor(scaled).data)


@pytest.mark.parametrize("ascending,expected", [(None, "159915"), (False, "159915"), (True, "510300")])
def test_top_n_and_next_open_select_volume_not_price_strength(volume_preview_factory, ascending, expected):
    preview, _ = volume_preview_factory()
    values = volume_factor(preview)
    day = values.calendar.dates[20]
    assert top_n_factor_preview(values, day, 1, ascending=ascending).table["ETF 代码"].tolist() == [expected]
    result = run_factor_portfolio(preview, factor_name=NAME, config=PortfolioConfig(top_n=1, ascending=ascending))
    assert result.trades.symbol.tolist() == [expected]
    assert result.trades.signal_date.eq(pd.Timestamp(day)).all()
    assert result.trades.execution_date.eq(pd.Timestamp(values.calendar.dates[21])).all()
    assert result.metadata["strategy"]["signal_anchor_position"] == 20
    assert result.metadata["strategy"]["resolved_ascending"] is (ascending is True)
    assert result.equity_curve.iloc[:21].equity.eq(100000).all()


@pytest.mark.parametrize("periods", [20, 21])
def test_warmup_then_final_signal_expires_without_execution(volume_preview_factory, periods):
    preview, _ = volume_preview_factory(periods=periods)
    values = volume_factor(preview)
    assert values.data.iloc[:40].status.eq("warmup").all()
    assert values.data.iloc[:40].factor_value.isna().all()
    result = run_factor_portfolio(preview, factor_name=NAME, config=PortfolioConfig(top_n=1))
    assert result.trades.empty
    assert result.decisions.status.tolist() == ([] if periods == 20 else ["EXPIRED"])


def test_zero_current_volume_is_valid_minus_one(volume_preview_factory):
    preview, _ = volume_preview_factory(current_zero=True)
    result = volume_factor(preview)
    day = result.calendar.dates[20]
    row = result.data.loc[result.data.date.eq(pd.Timestamp(day)) & result.data.symbol.eq("510300")].iloc[0]
    assert row.status == "valid" and row.factor_value == -1.0
    assert top_n_factor_preview(result, day, 1, ascending=True).table["ETF 代码"].tolist() == ["510300"]


def test_zero_denominator_skips_strategy_without_moving_benchmark_anchor(volume_preview_factory):
    preview, _ = volume_preview_factory(zero_baseline=True)
    values = volume_factor(preview)
    row = values.data.loc[values.data.date.eq(pd.Timestamp(values.calendar.dates[20])) & values.data.symbol.eq("159915")].iloc[0]
    assert row.status == "numeric_range" and pd.isna(row.factor_value)
    strategy = run_factor_portfolio(preview, factor_name=NAME, config=PortfolioConfig(top_n=2))
    assert strategy.trades.empty
    assert strategy.decisions.status.tolist() == ["SKIPPED_INSUFFICIENT_FACTORS"]
    benchmark = run_equal_weight_benchmark(preview, factor_name=NAME).backtest
    assert benchmark.metadata["strategy"]["signal_anchor_position"] == 20
    assert set(benchmark.trades.symbol) == {"159915", "510300"}
    assert benchmark.trades.execution_date.eq(pd.Timestamp(values.calendar.dates[21])).all()


def test_missing_full_window_recovers_without_compressing_axis(volume_preview_factory):
    preview, _ = volume_preview_factory(periods=42, gap=True)
    values = volume_factor(preview)
    asset = values.data.loc[values.data.symbol.eq("159915")].reset_index(drop=True)
    assert len(asset) == 42
    assert asset.status.iloc[20:26].eq("missing_window").all()
    assert asset.factor_value.iloc[20:26].isna().all()
    assert asset.status.iloc[26:].eq("valid").all()
    schedule = build_rebalance_schedule(values, config=PortfolioConfig(top_n=2))
    assert schedule.decisions.signal_date.tolist() == [pd.Timestamp(values.calendar.dates[i]) for i in (20, 40)]
    assert schedule.decisions.status.tolist() == ["SKIPPED_INSUFFICIENT_FACTORS", "PENDING"]


@pytest.mark.parametrize("name,lookback", [(NAME, 20), ("max_drawdown_20", 19), ("bias_20", 19),
    ("momentum_20", 20), ("momentum_60", 60), ("momentum_120", 120),
    ("volatility_20", 20), ("volatility_60", 60)])
def test_all_eight_archive_definitions_roundtrip_without_recomputation(volume_preview_factory, make_portfolio_preview, tmp_path, monkeypatch, name, lookback):
    assert name in create_default_registry().names()

    def volume_sample(*, periods):
        preview, saved = volume_preview_factory(periods=periods)
        return preview, saved.directory

    factory = volume_sample if name == NAME else make_portfolio_preview
    check_archive_roundtrip(factory, tmp_path, monkeypatch, name, lookback)
