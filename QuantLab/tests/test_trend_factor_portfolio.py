"""有符号趋势效率：真实快照、排名执行及档案往返契约。"""

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
from quantlab.ui.factors import top_n_factor_preview
from test_bias_portfolio import (
    test_archive_preserves_new_and_old_definitions_without_recomputing as check_archive_roundtrip,
)

NAME = "trend_efficiency_20"


@pytest.fixture
def trend_preview_factory(tmp_path):
    sequence = count()

    def make(*, periods=22, downside=False, gap=False):
        root = tmp_path / f"trend_sources_{next(sequence)}"
        root.mkdir()
        dates = pd.date_range("2025-01-02", periods=periods, freq="B")
        positions = np.arange(periods)
        for symbol in ("159915", "510300"):
            if downside:
                close = 200.0 - positions if symbol == "159915" else np.full(periods, 100.0)
            else:
                close = 100.0 + positions * 2 + (positions % 2) * 20 if symbol == "159915" else 100.0 + positions
            frame = pd.DataFrame({"date": dates, "symbol": symbol, "open": close,
                                  "high": close, "low": close, "close": close,
                                  "volume": 1000.0 + positions * 10})
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
        saved = save_research_set(preview, tmp_path / "trend_sets", name="趋势效率样本")
        return load_research_set(saved.directory.parent, saved.set_id).preview, saved

    return make


def trend_factor(preview):
    assert NAME in create_default_registry().names(), "趋势效率因子尚未注册"
    return run_research_factor(preview, NAME)


@pytest.mark.parametrize("ascending,expected", [(None, "510300"), (False, "510300"), (True, "159915")])
def test_top_n_and_execution_rank_efficiency_not_larger_price_gain(trend_preview_factory, ascending, expected):
    preview, _ = trend_preview_factory()
    result = trend_factor(preview)
    day = result.calendar.dates[20]
    first = result.data.loc[result.data.date.eq(pd.Timestamp(day))].set_index("symbol")
    # 大涨40%的震荡路径：10次+22与10次-18，净位移40/总路程400=0.1。
    assert first.loc["159915", "factor_value"] == pytest.approx(.1)
    assert first.loc["510300", "factor_value"] == 1.0
    momentum = run_research_factor(preview, "momentum_20")
    assert top_n_factor_preview(momentum, day, 1).table["ETF 代码"].tolist() == ["159915"]
    selected = top_n_factor_preview(result, day, 1, ascending=ascending)
    assert selected.table["ETF 代码"].tolist() == [expected]
    assert selected.table["目标权重"].tolist() == [1.0]
    portfolio = run_factor_portfolio(preview, factor_name=NAME, config=PortfolioConfig(top_n=1, ascending=ascending))
    assert portfolio.trades.symbol.tolist() == [expected]
    assert portfolio.trades.signal_date.eq(pd.Timestamp(day)).all()
    assert portfolio.trades.execution_date.eq(pd.Timestamp(result.calendar.dates[21])).all()
    assert portfolio.metadata["strategy"]["signal_anchor_position"] == 20
    assert portfolio.metadata["strategy"]["resolved_ascending"] is (ascending is True)
    assert portfolio.equity_curve.iloc[:21].equity.eq(100000).all()
    benchmark = run_equal_weight_benchmark(preview, factor_name=NAME).backtest
    assert benchmark.metadata["strategy"]["signal_anchor_position"] == 20
    assert benchmark.trades.execution_date.eq(pd.Timestamp(result.calendar.dates[21])).all()


@pytest.mark.parametrize("periods", [20, 21])
def test_no_execution_during_warmup_or_expired_first_signal(trend_preview_factory, periods):
    preview, _ = trend_preview_factory(periods=periods)
    result = trend_factor(preview)
    assert result.data.iloc[:40].status.eq("warmup").all()
    assert result.data.iloc[:40].factor_value.isna().all()
    portfolio = run_factor_portfolio(preview, factor_name=NAME, config=PortfolioConfig(top_n=1))
    assert portfolio.trades.empty
    assert portfolio.decisions.status.tolist() == ([] if periods == 20 else ["EXPIRED"])


def test_flat_zero_ranks_above_signed_downtrend(trend_preview_factory):
    preview, _ = trend_preview_factory(downside=True)
    result = trend_factor(preview)
    day = result.calendar.dates[20]
    first = result.data.loc[result.data.date.eq(pd.Timestamp(day))].set_index("symbol")
    assert first.status.eq("valid").all()
    assert first.loc["159915", "factor_value"] == -1.0
    assert first.loc["510300", "factor_value"] == 0.0
    assert top_n_factor_preview(result, day, 1).table["ETF 代码"].tolist() == ["510300"]
    portfolio = run_factor_portfolio(preview, factor_name=NAME, config=PortfolioConfig(top_n=1))
    assert portfolio.trades.symbol.tolist() == ["510300"]


@pytest.mark.parametrize("name,lookback", [(NAME, 20), ("volume_change_20", 20),
    ("max_drawdown_20", 19), ("bias_20", 19), ("momentum_20", 20), ("momentum_60", 60),
    ("momentum_120", 120), ("volatility_20", 20), ("volatility_60", 60)])
def test_all_nine_factor_archives_roundtrip_without_recomputation(trend_preview_factory, make_portfolio_preview, tmp_path, monkeypatch, name, lookback):
    assert name in create_default_registry().names()

    def trend_sample(*, periods):
        preview, saved = trend_preview_factory(periods=periods)
        return preview, saved.directory

    check_archive_roundtrip(trend_sample if name == NAME else make_portfolio_preview,
                            tmp_path, monkeypatch, name, lookback)
