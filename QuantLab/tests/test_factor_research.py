"""合成行情上的真实研究链路；不访问私人行情或外部数据服务。"""

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from itertools import count
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.data.calendar import parse_calendar_csv
from quantlab.data.catalog import scan_market_catalog
from quantlab.data.importer import import_market_csv
from quantlab.data.research_sets import DatasetSelection, SOURCE_FIELDS, build_research_set
from quantlab.data.research_storage import load_research_set, save_research_set
from quantlab.factors import (
    FactorEvaluation, FactorRegistry, Momentum20Factor, run_research_factor,
)


@pytest.fixture
def make_preview(tmp_path):
    sequence = count()

    def build(*, gaps=None, outside=None, supplied=False, adjustment="none",
              bump_after=None, start_position=0, end_position=44):
        source_root = tmp_path / f"sources_{next(sequence)}"
        source_root.mkdir()
        raw, processed = source_root / "raw", source_root / "processed"
        dates = pd.date_range("2025-01-02", periods=45, freq="B").astype("datetime64[ns]")
        for symbol, values in (("510300", 100.0 + np.arange(45) * 0.5),
                               ("159915", 200.0 - np.arange(45))):
            if bump_after is not None:
                values[bump_after:] *= 2
            asset_dates = dates + pd.Timedelta(days=365) if outside == symbol else dates
            frame = pd.DataFrame({
                "date": asset_dates, "symbol": symbol, "open": values,
                "high": values, "low": values, "close": values, "volume": 0.0,
            })
            frame = frame.drop(index=(gaps or {}).get(symbol, []))
            path = source_root / f"{symbol}.csv"
            frame.to_csv(path, index=False)
            import_market_csv(path, raw, processed,
                              imported_at=datetime(2025, 5, 1, tzinfo=timezone.utc))
        calendar = None
        if supplied:
            content = ("date\n" + "\n".join(day.date().isoformat() for day in dates) + "\n").encode()
            calendar = parse_calendar_csv(
                content, source="合成声明日历",
                coverage_start=dates[0].date(), coverage_end=dates[-1].date(),
            )
        scan = scan_market_catalog(processed, raw)
        assert not scan.issues and len(scan.records) == 2
        preview = build_research_set(
            tuple(DatasetSelection(record, adjustment) for record in scan.records),
            start=dates[start_position].date(), end=dates[end_position].date(), calendar=calendar,
        )
        return preview, source_root

    return build


def test_multi_etf_values_grid_and_provenance(make_preview):
    preview, _ = make_preview(supplied=True)
    before = deepcopy(preview)
    result = run_research_factor(preview)
    assert result.data.columns.tolist() == ["date", "symbol", "factor_value", "is_valid", "status"]
    pd.testing.assert_frame_equal(
        result.data[["date", "symbol"]], preview.alignment.data[["date", "symbol"]],
    )
    up = result.data.loc[result.data.symbol.eq("510300")]
    down = result.data.loc[result.data.symbol.eq("159915")]
    assert up.factor_value.iloc[20] == pytest.approx(0.10)
    assert down.factor_value.iloc[20] == pytest.approx(-0.10)
    assert up.status.iloc[:20].eq("warmup").all()
    assert down.status.iloc[:20].eq("warmup").all()
    pd.testing.assert_series_equal(
        result.data.is_valid, result.data.status.eq("valid").astype(bool).rename("is_valid"),
    )
    assert result.input_preview_sha256 == preview.preview_sha256
    assert result.alignment_content_sha256 == preview.alignment.content_sha256
    assert asdict(result.calendar) == asdict(preview.alignment.calendar)
    assert (result.start, result.end, result.adjustment) == (preview.start, preview.end, "none")
    assert result.definition == Momentum20Factor.definition
    assert result.definition is not Momentum20Factor.definition
    assert set(result.sources[0]) == set(SOURCE_FIELDS) - {"processed_path"}
    assert result.sources == tuple(
        {k: v for k, v in source.items() if k != "processed_path"} for source in preview.sources
    )
    assert all(w in result.warnings for w in preview.warnings)
    assert any("总回报" in w for w in result.warnings)
    assert any("收盘" in w for w in result.warnings)
    pd.testing.assert_frame_equal(preview.alignment.data, before.alignment.data)
    pd.testing.assert_frame_equal(preview.coverage, before.coverage)
    assert preview.sources == before.sources
    result.data.loc[0, "factor_value"] = 999.0
    result.sources[0]["provider"] = "changed-output"
    assert preview.sources == before.sources
    pd.testing.assert_frame_equal(preview.alignment.data, before.alignment.data)


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("labels", ["duplicate", "noncontiguous", "multi"])
def test_research_index_labels_are_not_data_identity(make_preview, labels, supplied):
    preview, _ = make_preview(supplied=supplied)
    expected = run_research_factor(preview)
    size = len(preview.alignment.data)
    new_index = pd.Index([0] * size if labels == "duplicate" else np.arange(size) * 3, name="date")
    if labels == "multi":
        new_index = pd.MultiIndex.from_arrays(
            [[0] * size, np.arange(size)], names=["symbol", "date"],
        )
    preview.alignment.data.index = new_index
    actual = run_research_factor(preview)
    pd.testing.assert_frame_equal(actual.data, expected.data)
    assert actual.input_preview_sha256 == expected.input_preview_sha256
    assert preview.alignment.data.index.identical(new_index)


def test_all_missing_asset_stays_and_short_snapshot_does_not_reload_history(make_preview):
    preview, _ = make_preview(outside="159915")
    result = run_research_factor(preview)
    absent = result.data.loc[result.data.symbol.eq("159915")]
    assert len(absent) == 45 and absent.factor_value.isna().all()
    assert absent.status.iloc[:20].eq("warmup").all()
    assert absent.status.iloc[20:].eq("missing_window").all()
    short, root = make_preview(start_position=25)
    root.rename(root.with_name(root.name + "_moved"))
    unavailable = run_research_factor(short)
    assert len(unavailable.data) == 40
    assert unavailable.data.status.eq("warmup").all()
    assert not unavailable.data.is_valid.any()
    assert any("没有可用" in w for w in unavailable.warnings)


def test_union_and_supplied_calendar_are_not_interchangeable(make_preview):
    gaps = {"510300": [5], "159915": [5]}
    union, _ = make_preview(gaps=gaps)
    supplied, _ = make_preview(gaps=gaps, supplied=True)
    union_out, supplied_out = run_research_factor(union), run_research_factor(supplied)
    day = pd.Timestamp(supplied.alignment.calendar.dates[21])
    a = union_out.data.loc[lambda f: f.date.eq(day) & f.symbol.eq("510300")].iloc[0]
    b = supplied_out.data.loc[lambda f: f.date.eq(day) & f.symbol.eq("510300")].iloc[0]
    assert a.status == "valid" and a.factor_value == pytest.approx(0.105)
    assert b.status == "missing_window" and pd.isna(b.factor_value)
    assert union_out.input_preview_sha256 != supplied_out.input_preview_sha256
    assert supplied_out.calendar.file_sha256 == supplied.alignment.calendar.file_sha256
    assert supplied_out.calendar.mode == "supplied" and union_out.calendar.mode == "observed_union"


@pytest.mark.parametrize("adjustment", ["qfq", "hfq"])
def test_adjustment_declaration_retained_not_recomputed(make_preview, adjustment):
    preview, _ = make_preview(adjustment=adjustment)
    result = run_research_factor(preview)
    assert result.adjustment == adjustment
    assert all(s["adjustment_basis"] == "user_declared" for s in result.sources)
    assert all(w in result.warnings for w in preview.warnings)
    assert any("当时可得" in w for w in result.warnings)
    assert any("同一复权标签" in w for w in result.warnings)


def test_later_price_change_cannot_change_earlier_results(make_preview):
    original, _ = make_preview()
    changed, _ = make_preview(bump_after=30)
    left, right = run_research_factor(original), run_research_factor(changed)
    cutoff = pd.Timestamp(original.alignment.calendar.dates[30])
    pd.testing.assert_frame_equal(
        left.data.loc[left.data.date.lt(cutoff)], right.data.loc[right.data.date.lt(cutoff)],
    )
    assert not left.data.loc[left.data.date.eq(cutoff)].equals(right.data.loc[right.data.date.eq(cutoff)])
    assert left.input_preview_sha256 != right.input_preview_sha256


@pytest.mark.parametrize("mutation", ["price", "source", "extra_source", "calendar", "hash", "coverage", "warnings"])
def test_mutated_preview_is_rejected_not_repaired(research_preview, mutation):
    preview = deepcopy(research_preview)
    if mutation == "price":
        preview.alignment.data.loc[preview.alignment.data.has_observation, "close"] = 3.15
    elif mutation == "source":
        preview.sources[0]["version_id"] = "changed-version"
    elif mutation == "extra_source":
        preview.sources[0]["secret"] = "synthetic-only"
    elif mutation == "calendar":
        preview = replace(preview, alignment=replace(
            preview.alignment, calendar=replace(preview.alignment.calendar, source="changed"),
        ))
    elif mutation == "hash":
        preview = replace(preview, preview_sha256="0" * 64)
    elif mutation == "coverage":
        preview.coverage.loc[0, "missing_count"] = 123
    else:
        preview = replace(preview, warnings=())
    original_hash = preview.preview_sha256
    with pytest.raises(ValueError):
        run_research_factor(preview)
    assert preview.preview_sha256 == original_hash


def test_invalid_entry_arguments(research_preview):
    with pytest.raises(ValueError, match="预览|研究"):
        run_research_factor(None)
    with pytest.raises(KeyError, match="未注册"):
        run_research_factor(research_preview, "unknown")
    with pytest.raises(ValueError, match="名称"):
        run_research_factor(research_preview, [])
    with pytest.raises(ValueError, match="注册表"):
        run_research_factor(research_preview, registry={})


@pytest.mark.parametrize("damage", [
    "wrong_type", "array_values", "array_status", "short", "reversed", "index_name",
    "int_values", "nullable_values", "infinity", "bad_state", "na_state",
    "numeric_state", "object_state", "valid_nan", "invalid_number",
])
def test_bad_factor_output_fails_instead_of_silent_alignment(make_preview, damage):
    class BrokenFactor(Momentum20Factor):
        name = "broken"
        definition = replace(Momentum20Factor.definition, name=name)

        def evaluate(self, frame):
            out = super().evaluate(frame)
            values, status = out.values.copy(), out.status.copy()
            if damage == "wrong_type":
                return values
            if damage == "array_values":
                values = values.to_numpy()
            elif damage == "array_status":
                status = status.to_numpy()
            elif damage == "short":
                values = values.iloc[:-1]
            elif damage == "reversed":
                status = status.iloc[::-1]
            elif damage == "index_name":
                values.index = values.index.rename("wrong")
            elif damage == "int_values":
                values = values.fillna(0).astype("int64")
            elif damage == "nullable_values":
                values = values.astype("Float64")
            elif damage == "infinity":
                values.iloc[-1] = np.inf
            elif damage == "bad_state":
                status.iloc[0] = "unknown"
            elif damage == "na_state":
                status.iloc[0] = pd.NA
            elif damage == "numeric_state":
                status = pd.Series(1, index=frame.index)
            elif damage == "object_state":
                status = status.astype(object)
            elif damage == "valid_nan":
                status.iloc[0] = "valid"
            elif damage == "invalid_number":
                values.iloc[0] = 0.0
            return FactorEvaluation(values, status)

    registry = FactorRegistry()
    registry.register(BrokenFactor)
    preview, _ = make_preview()
    with pytest.raises(ValueError, match="因子.*输出"):
        run_research_factor(preview, "broken", registry=registry)


def test_factor_cannot_rewrite_merge_keys_or_caller_data(make_preview):
    class MutatingFactor(Momentum20Factor):
        name = "mutating"
        definition = replace(Momentum20Factor.definition, name=name)

        def evaluate(self, frame):
            out = super().evaluate(frame)
            frame["date"] = frame.date + pd.Timedelta(days=100)
            frame["symbol"] = "000000"
            frame["close"] = 1.0
            return out

    preview, _ = make_preview()
    before = preview.alignment.data.copy(deep=True)
    registry = FactorRegistry()
    registry.register(MutatingFactor)
    expected = run_research_factor(preview)
    result = run_research_factor(preview, "mutating", registry=registry)
    pd.testing.assert_frame_equal(result.data, expected.data)
    pd.testing.assert_frame_equal(preview.alignment.data, before)


def test_saved_snapshot_calculates_without_original_files(make_preview, tmp_path, monkeypatch):
    preview, sources = make_preview(supplied=True)
    root = tmp_path / "sets"
    saved = save_research_set(preview, root, name="合成动量验收")
    saved_bytes = {p.name: p.read_bytes() for p in saved.directory.iterdir()}
    sources.rename(sources.with_name(sources.name + "_moved"))
    loaded = load_research_set(root, saved.set_id)
    assert all(not Path(s["processed_path"]).exists() for s in loaded.preview.sources)
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "synthetic-secret-not-for-results")

    def no_file_access(*args, **kwargs):
        raise AssertionError("因子计算不应读写文件")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", no_file_access)
        patch.setattr("builtins.open", no_file_access)
        patch.setattr(pd, "read_parquet", no_file_access)
        result = run_research_factor(loaded.preview)
    assert result.data.loc[result.data.symbol.eq("510300"), "factor_value"].iloc[20] == pytest.approx(0.10)
    assert result.input_preview_sha256 == preview.preview_sha256
    assert result.definition.version == "1.0"
    assert result.calendar.file_sha256 == preview.alignment.calendar.file_sha256
    assert "synthetic-secret-not-for-results" not in repr(result)
    assert all("processed_path" not in source for source in result.sources)
    assert {p.name: p.read_bytes() for p in saved.directory.iterdir()} == saved_bytes
