"""固定双因子预览：真实快照、手算排名、共同样本及隔离契约。"""

from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timezone
from itertools import count

import numpy as np
import pandas as pd
import pytest

from quantlab import factors
from quantlab.data.catalog import scan_market_catalog
from quantlab.data.importer import import_market_csv
from quantlab.data.research_sets import DatasetSelection, build_research_set
from quantlab.data.research_storage import load_research_set, save_research_set


@pytest.fixture
def pair_preview_factory(tmp_path):
    sequence = count()

    def make(paths=None, *, length=45, gaps=None):
        root = tmp_path / f"pair_{next(sequence)}"
        root.mkdir()
        days = pd.date_range("2025-01-02", periods=length, freq="B")
        if paths is None:
            positions = np.arange(length)
            paths = {"510300": 100. + positions, "159915": np.full(length, 100.),
                     "510500": 100. + positions * 2 + (positions % 2) * 20}
        for symbol, close in paths.items():
            assert len(close) == length
            frame = pd.DataFrame({"date": days, "symbol": symbol, "open": close,
                                  "high": close, "low": close, "close": close, "volume": 1000.})
            frame = frame.drop(index=(gaps or {}).get(symbol, []))
            csv = root / f"{symbol}.csv"
            frame.to_csv(csv, index=False)
            import_market_csv(csv, root / "raw", root / "processed",
                              imported_at=datetime(2025, 10, 1, tzinfo=timezone.utc))
        catalog = scan_market_catalog(root / "processed", root / "raw")
        assert not catalog.issues
        preview = build_research_set(tuple(DatasetSelection(r, "none") for r in catalog.records),
                                     start=days[0].date(), end=days[-1].date())
        saved = save_research_set(preview, tmp_path / "pair_sets", name="双因子合成验收")
        return load_research_set(saved.directory.parent, saved.set_id).preview, saved

    return make


def paired(preview, day=None):
    assert hasattr(factors, "run_dual_factor_preview"), "尚未实现双因子评分预览"
    return factors.run_dual_factor_preview(preview, day or preview.alignment.calendar.dates[-1])


def test_three_assets_keep_component_disagreement_and_exact_combined_ties(pair_preview_factory):
    preview, saved = pair_preview_factory()
    before = deepcopy(preview)
    saved_bytes = {p.name: p.read_bytes() for p in saved.directory.iterdir()}
    day = preview.alignment.calendar.dates[20]
    result = paired(preview, day)
    table = result.data.set_index("symbol")
    assert result.selected_date == day and result.common_count == 3
    assert result.input_preview_sha256 == preview.preview_sha256
    assert set(table.index) == {"159915", "510300", "510500"}
    # 首尾涨幅 0%、20%、40%；平价波动为0，线性上涨波动小于大幅往返。
    assert table.loc[["159915", "510300", "510500"], "momentum_value"].tolist() == pytest.approx([0., .2, .4])
    assert table.loc[["159915", "510300", "510500"], "momentum_score"].tolist() == [0., 50., 100.]
    assert table.loc[["159915", "510300", "510500"], "volatility_score"].tolist() == [100., 50., 0.]
    assert table.combined_score.tolist() == [50., 50., 50.]
    assert table["rank"].tolist() == [1., 1., 1.]
    assert result.data.symbol.tolist() == ["159915", "510300", "510500"]
    assert table.is_common.all()
    assert any("未来" in w for w in result.warnings)
    assert any("总回报" in w for w in result.warnings)
    result.data.loc[0, "combined_score"] = 999.
    pd.testing.assert_frame_equal(preview.alignment.data, before.alignment.data)
    pd.testing.assert_frame_equal(preview.coverage, before.coverage)
    assert preview.sources == before.sources
    assert {p.name: p.read_bytes() for p in saved.directory.iterdir()} == saved_bytes


def test_partial_factor_validity_recalculates_both_scores_on_common_pool(pair_preview_factory):
    c = np.full(21, 1e-300)
    c[10], c[-1] = 1e10, 3e-300
    paths = {"510300": 100. + np.arange(21) * 5, "159915": np.full(21, 100.),
             "510500": c, "512000": np.logspace(-300, 10, 21)}
    preview, _ = pair_preview_factory(paths, length=21)
    result = paired(preview)
    table = result.data.set_index("symbol")
    assert result.common_count == 2
    assert table.loc["510500", "momentum_status"] == "valid"
    assert table.loc["510500", "volatility_status"] == "numeric_range"
    assert table.loc["512000", "momentum_status"] == "numeric_range"
    assert table.loc["512000", "volatility_status"] == "valid"
    assert table.loc["510500", "momentum_value"] == pytest.approx(2.)
    # 单项动量池中510500更高，但共同池排除它后，510300须重新得100而不是50。
    assert table.loc["510300", "momentum_score"] == 100.
    assert table.loc["510300", "volatility_score"] == 0.
    assert table.loc["159915", "momentum_score"] == 0.
    assert table.loc["159915", "volatility_score"] == 100.
    assert table.loc[["510300", "159915"], "combined_score"].tolist() == [50., 50.]
    assert table.loc[["510500", "512000"], ["momentum_score", "volatility_score", "combined_score", "rank"]].isna().all().all()
    assert not table.loc[["510500", "512000"], "is_common"].any()


def test_all_equal_values_are_valid_neutral_scores(pair_preview_factory):
    preview, _ = pair_preview_factory({"510300": np.full(21, 100.), "159915": np.full(21, 200.)}, length=21)
    table = paired(preview).data
    assert table.momentum_value.eq(0.).all() and table.volatility_value.eq(0.).all()
    assert table[["momentum_score", "volatility_score", "combined_score"]].eq(50.).all().all()
    assert table["rank"].eq(1.).all()


@pytest.mark.parametrize("position,common", [(0, 0), (19, 0), (20, 3), (25, 1), (44, 1)])
def test_exact_day_warmup_and_missing_samples_never_backfill(pair_preview_factory, position, common):
    preview, _ = pair_preview_factory(gaps={"510300": [25], "510500": [25]})
    result = paired(preview, preview.alignment.calendar.dates[position])
    assert result.common_count == common
    if common < 2:
        assert result.data[["momentum_score", "volatility_score", "combined_score", "rank"]].isna().all().all()
    if common == 1:
        row = result.data.set_index("symbol").loc["159915"]
        assert row.is_common and row.momentum_value == 0. and row.volatility_value == 0.
        assert row.momentum_status == "valid" and row.volatility_status == "valid"
        assert result.data.set_index("symbol").loc["510300", "momentum_status"] == "missing_window"


def test_joint_sample_recovers_only_after_missing_point_leaves_window(pair_preview_factory):
    preview, _ = pair_preview_factory(length=47, gaps={"510300": [25], "510500": [25]})
    assert paired(preview, preview.alignment.calendar.dates[45]).common_count == 1
    assert paired(preview, preview.alignment.calendar.dates[46]).common_count == 3


def test_later_prices_do_not_change_earlier_pair_preview(pair_preview_factory):
    positions = np.arange(45)
    paths = {"510300": 100. + positions, "159915": np.full(45, 100.)}
    first, _ = pair_preview_factory(paths)
    paths["510300"][21:] *= 3
    second, _ = pair_preview_factory(paths)
    day = first.alignment.calendar.dates[20]
    pd.testing.assert_frame_equal(paired(first, day).data, paired(second, day).data)


@pytest.mark.parametrize("day", [date(1999, 1, 1), "2025-01-02", pd.Timestamp("2025-01-02"), True])
def test_date_must_be_pure_date_on_declared_axis(pair_preview_factory, day):
    preview, _ = pair_preview_factory()
    with pytest.raises(ValueError, match="日期"):
        paired(preview, day)


def test_preview_fingerprint_is_checked_not_repaired(pair_preview_factory):
    preview, _ = pair_preview_factory()
    invalid = replace(preview, preview_sha256="0" * 64)
    with pytest.raises(ValueError):
        paired(invalid)


def test_joint_preview_preserves_single_factor_top_n_and_schedule(pair_preview_factory):
    from quantlab.ui.factors import top_n_factor_preview
    from quantlab.portfolio.selection import build_rebalance_schedule
    from quantlab.portfolio import PortfolioConfig

    preview, _ = pair_preview_factory()
    factor = factors.run_research_factor(preview)
    day = factor.calendar.dates[20]
    single_before = deepcopy(factor.data)
    schedule = build_rebalance_schedule(factor, config=PortfolioConfig(top_n=1))
    top = top_n_factor_preview(factor, day, 1)
    assert top.table["ETF 代码"].tolist() == ["510500"]
    paired(preview, day)
    pd.testing.assert_frame_equal(factor.data, single_before)
    pd.testing.assert_frame_equal(top_n_factor_preview(factor, day, 1).table, top.table)
    after = build_rebalance_schedule(factor, config=PortfolioConfig(top_n=1))
    pd.testing.assert_frame_equal(after.decisions, schedule.decisions)
    pd.testing.assert_frame_equal(after.targets, schedule.targets)
