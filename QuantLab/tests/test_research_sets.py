"""研究预览契约：真实临时版本文件、独立期望值，不访问私人或联网行情。"""

from dataclasses import replace
from datetime import date, datetime, timezone
from hashlib import sha256
import json

import pandas as pd
import pytest

from quantlab.data.calendar import parse_calendar_csv
from quantlab.data.catalog import scan_market_catalog
from quantlab.data.importer import import_market_csv
from quantlab.data.market_service import MarketDataService
from quantlab.data.providers.base import MarketDataProvider
from quantlab.data import research_sets


START, END = date(2025, 1, 2), date(2025, 1, 7)
SOURCE_FIELDS = {
    "symbol", "version_id", "provider", "imported_at", "processed_path",
    "file_sha256", "metadata_sha256", "input_content_sha256", "adjustment", "adjustment_basis",
}


@pytest.fixture
def api():
    return research_sets


@pytest.fixture
def records(imported_pair):
    raw, processed, _ = imported_pair
    return scan_market_catalog(processed, raw).records


def selections(api, records, adjustment="none"):
    return tuple(api.DatasetSelection(record, adjustment) for record in records)


def preview(api, records, **kwargs):
    return api.build_research_set(selections(api, records), start=START, end=END, **kwargs)


def test_unknown_csv_requires_explicit_adjustment(api, records):
    with pytest.raises(ValueError, match="复权|口径"):
        api.build_research_set(selections(api, records, None), start=START, end=END)


def test_csv_preview_pins_sources_and_enriches_report(api, records):
    result = preview(api, records)
    assert result.adjustment == "none"
    assert [s["symbol"] for s in result.sources] == ["159915", "510300"]
    assert {s["version_id"] for s in result.sources} == {r.version_id for r in records}
    assert {s["adjustment_basis"] for s in result.sources} == {"user_declared"}
    for source in result.sources:
        assert set(source) == SOURCE_FIELDS
        record = next(r for r in records if r.symbol == source["symbol"])
        assert source["file_sha256"] == sha256(record.processed_path.read_bytes()).hexdigest()
        assert source["metadata_sha256"] == record.metadata_sha256
        assert source["imported_at"] == record.imported_at
        assert source["processed_path"] == str(record.processed_path)
    assert result.coverage.symbol.tolist() == ["159915", "510300"]
    assert result.coverage.observation_count.tolist() == [3, 3]
    assert result.coverage.missing_count.tolist() == [1, 1]
    assert result.coverage.missing_ratio.tolist() == [0.25, 0.25]
    assert result.coverage.adjustment_basis.tolist() == ["user_declared", "user_declared"]
    assert result.alignment.common_dates == (date(2025, 1, 3), date(2025, 1, 7))
    assert any("用户声明" in w for w in result.warnings)
    assert any("未经" in w for w in result.warnings)
    api.validate_research_preview(result)
    assert api.research_preview_sha256(result) == result.preview_sha256


@pytest.mark.parametrize("value", ["unknown", "", "QFQ", 1, False, [], {}])
def test_invalid_declarations_rejected_before_reading_files(api, records, value):
    for record in records:
        record.processed_path.rename(record.processed_path.with_suffix(".unavailable"))
    with pytest.raises(ValueError, match="复权|口径"):
        api.build_research_set(selections(api, records, value), start=START, end=END)


def test_mixed_adjustments_rejected_before_file_access(api, records):
    selected = (api.DatasetSelection(records[0], "none"), api.DatasetSelection(records[1], "qfq"))
    for record in records:
        record.processed_path.rename(record.processed_path.with_suffix(".unavailable"))
    with pytest.raises(ValueError, match="口径.*一致|混合"):
        api.build_research_set(selected, start=START, end=END)


@pytest.mark.parametrize("different_version", [False, True])
def test_one_version_per_symbol_checked_before_read(api, records, different_version):
    record = records[0]
    other = replace(record, version_id="other") if different_version else record
    record.processed_path.rename(record.processed_path.with_suffix(".unavailable"))
    with pytest.raises(ValueError, match="重复|每只|每个"):
        api.build_research_set(selections(api, (record, other)), start=START, end=END)


@pytest.mark.parametrize("bad", [(), [], None, ("not-selection", "not-selection")])
def test_invalid_selection_container_has_clear_error(api, bad):
    with pytest.raises(ValueError, match="选择|至少|ETF"):
        api.build_research_set(bad, start=START, end=END)


def test_single_asset_and_wrong_record_type_rejected(api, records):
    for selected in ((api.DatasetSelection(records[0], "none"),),
                     (api.DatasetSelection(None, "none"), api.DatasetSelection(records[1], "none"))):
        with pytest.raises(ValueError, match="至少|版本|选择"):
            api.build_research_set(selected, start=START, end=END)


@pytest.mark.parametrize(("start", "end"), [
    (END, START), ("2025-01-02", END), (datetime(2025, 1, 2), END), (START, None),
])
def test_invalid_range_rejected_before_file_access(api, records, start, end):
    for record in records:
        record.processed_path.rename(record.processed_path.with_suffix(".unavailable"))
    with pytest.raises(ValueError, match="日期|区间"):
        api.build_research_set(selections(api, records), start=start, end=end)


class SyntheticProvider(MarketDataProvider):
    key = "synthetic"
    name = "合成测试来源，不是真实行情"

    def __init__(self, data):
        self.data = data

    def fetch_etf_history(self, symbol, start_date, end_date, period="daily", adjustment="none"):
        assert symbol == "510300" and period == "daily" and adjustment == "qfq"
        return self.data.drop(columns="symbol").copy(deep=True)


@pytest.fixture
def mixed_records(imported_pair, sample_assets):
    raw, processed, _ = imported_pair
    service = MarketDataService(SyntheticProvider(sample_assets["510300"]), raw, processed)
    fetched = service.fetch_etf_history("510300", START, END, adjustment="qfq")
    records = scan_market_catalog(processed, raw).records
    return tuple(r for r in records if r.symbol == "159915" or r.version_id == fetched.metadata.data_version)


@pytest.mark.parametrize("declaration", [None, "qfq"])
def test_known_provider_basis_and_mixed_source_warning(api, mixed_records, declaration):
    selected = tuple(api.DatasetSelection(r, declaration if r.provider == "synthetic" else "qfq")
                     for r in mixed_records)
    result = api.build_research_set(selected, start=START, end=END)
    sources = {s["symbol"]: s for s in result.sources}
    assert result.adjustment == "qfq"
    assert sources["510300"]["adjustment_basis"] == "provider"
    assert sources["159915"]["adjustment_basis"] == "user_declared"
    assert any("来源" in w and "不保证" in w for w in result.warnings)
    api.validate_research_preview(result)


def test_cannot_override_provider_adjustment(api, mixed_records):
    with pytest.raises(ValueError, match="不能覆盖"):
        preview(api, mixed_records)


def test_new_import_does_not_replace_selected_old_version(api, records, imported_pair, tmp_path, sample_assets):
    original = preview(api, records)
    raw, processed, _ = imported_pair
    newer_frame = sample_assets["510300"].copy()
    newer_frame["close"] = 3.15
    source = tmp_path / "newer.csv"
    newer_frame.to_csv(source, index=False)
    newer = import_market_csv(source, raw, processed, imported_at=datetime(2025, 2, 1, tzinfo=timezone.utc))
    repeated = preview(api, records)
    assert repeated.preview_sha256 == original.preview_sha256
    assert newer.version.version_id not in {s["version_id"] for s in repeated.sources}
    assert repeated.alignment.data.loc[lambda d: d.has_observation, "close"].eq(3.1).all()


def test_selection_order_does_not_change_preview(api, records):
    first, second = preview(api, records), preview(api, tuple(reversed(records)))
    assert first.preview_sha256 == second.preview_sha256
    assert first.sources == second.sources
    pd.testing.assert_frame_equal(first.coverage, second.coverage)
    pd.testing.assert_frame_equal(first.alignment.data, second.alignment.data)


def test_full_input_fingerprint_includes_rows_outside_research_interval(api, records):
    first = api.build_research_set(selections(api, records), start=date(2025, 1, 3), end=END)
    record = next(r for r in records if r.symbol == "510300")
    frame = pd.read_parquet(record.processed_path)
    frame.loc[frame.date.eq(pd.Timestamp(START)), "close"] = 3.15
    frame.to_parquet(record.processed_path, index=False)
    second = api.build_research_set(selections(api, records), start=date(2025, 1, 3), end=END)
    assert first.alignment.content_sha256 == second.alignment.content_sha256
    a = next(s for s in first.sources if s["symbol"] == "510300")
    b = next(s for s in second.sources if s["symbol"] == "510300")
    assert a["input_content_sha256"] != b["input_content_sha256"]
    assert first.preview_sha256 != second.preview_sha256


def test_semantic_input_fingerprint_ignores_row_order_and_extra_columns(api, records):
    first = preview(api, records)
    for record in records:
        frame = pd.read_parquet(record.processed_path).iloc[::-1].copy()
        frame["unused"] = "not part of OHLCV"
        frame.to_parquet(record.processed_path, index=False)
    second = preview(api, records)
    assert [s["input_content_sha256"] for s in first.sources] == [s["input_content_sha256"] for s in second.sources]
    assert [s["file_sha256"] for s in first.sources] != [s["file_sha256"] for s in second.sources]


def test_metadata_change_after_selection_rejected(api, records):
    path = records[0].raw_version_dir / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["imported_at"] = "2025-02-01T00:00:00+00:00"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="元数据.*改变"):
        preview(api, records)


def test_invalid_complete_history_rejected_even_outside_interval(api, records):
    record = next(r for r in records if r.symbol == "510300")
    frame = pd.read_parquet(record.processed_path)
    frame.loc[frame.date.eq(pd.Timestamp(START)), "close"] = -1
    frame.to_parquet(record.processed_path, index=False)
    with pytest.raises(ValueError, match="510300"):
        api.build_research_set(selections(api, records), start=date(2025, 1, 3), end=END)


def test_one_asset_without_interval_observations_remains_in_report(api, records):
    result = api.build_research_set(selections(api, records), start=START, end=START)
    row = result.coverage.set_index("symbol").loc["159915"]
    assert row.observation_count == 0
    assert row.missing_ratio == 1.0
    assert row.first_observed is None
    assert result.alignment.common_dates == ()
    assert any("159915" in w and "无行情" in w for w in result.warnings)
    assert any("共同日期" in w for w in result.warnings)
    api.validate_research_preview(result)


def test_all_assets_without_interval_data_rejected(api, records):
    with pytest.raises(ValueError, match="无行情|为空|没有"):
        api.build_research_set(selections(api, records), start=date(2024, 1, 1), end=date(2024, 1, 5))


def test_supplied_calendar_keeps_all_missing_day_and_declared_coverage(api, records):
    calendar = parse_calendar_csv(
        b"date\n2025-01-02\n2025-01-03\n2025-01-06\n2025-01-07\n2025-01-08\n",
        source="Synthetic calendar", coverage_start=date(2025, 1, 1), coverage_end=date(2025, 1, 10),
    )
    result = api.build_research_set(selections(api, records), start=START, end=date(2025, 1, 8), calendar=calendar)
    assert result.alignment.calendar.coverage_end == date(2025, 1, 10)
    assert result.coverage.missing_count.tolist() == [2, 2]
    assert not result.alignment.data.loc[lambda d: d.date.eq("2025-01-08"), "has_observation"].any()
    assert any("不代表" in w for w in result.warnings)
    api.validate_research_preview(result)


@pytest.mark.parametrize("target", [
    "price", "missing_fill", "core_coverage", "enriched_coverage", "common_dates",
    "alignment_hash", "preview_hash", "source_version", "source_input_hash", "source_provider",
    "source_basis", "extra_secret_field", "source_order", "missing_source", "start", "end",
    "adjustment", "warnings", "alignment_warnings", "calendar_source",
])
def test_mutated_preview_rejected_without_repair(api, research_preview, target):
    result = research_preview
    original_hash = result.preview_sha256
    if target == "price":
        result.alignment.data.loc[result.alignment.data.has_observation, "close"] = 3.15
    elif target == "missing_fill":
        result.alignment.data.loc[~result.alignment.data.has_observation, "volume"] = 0.0
    elif target == "core_coverage":
        result.alignment.coverage.loc[0, "missing_count"] = 0
    elif target == "enriched_coverage":
        result.coverage.loc[0, "version_id"] = "another-version"
    elif target == "common_dates":
        result = replace(result, alignment=replace(result.alignment, common_dates=(START,)))
    elif target == "alignment_hash":
        result = replace(result, alignment=replace(result.alignment, content_sha256="0" * 64))
    elif target == "preview_hash":
        result = replace(result, preview_sha256="0" * 64)
    elif target.startswith("source_") and target not in ("source_order",):
        fields = {"source_version": "version_id", "source_input_hash": "input_content_sha256",
                  "source_provider": "provider", "source_basis": "adjustment_basis"}
        result.sources[0][fields[target]] = "0" * 64 if target.endswith("hash") else "changed"
    elif target == "extra_secret_field":
        result.sources[0]["secret"] = "must not be accepted"
    elif target == "source_order":
        result = replace(result, sources=tuple(reversed(result.sources)))
    elif target == "missing_source":
        result = replace(result, sources=result.sources[:1])
    elif target == "start":
        result = replace(result, start=date(2025, 1, 3))
    elif target == "end":
        result = replace(result, end=date(2025, 1, 6))
    elif target == "adjustment":
        result = replace(result, adjustment="qfq")
    elif target == "warnings":
        result = replace(result, warnings=())
    elif target == "alignment_warnings":
        result = replace(result, alignment=replace(result.alignment, warnings=()))
    else:
        axis = replace(result.alignment.calendar, source="modified")
        result = replace(result, alignment=replace(result.alignment, calendar=axis))
    with pytest.raises(ValueError):
        api.validate_research_preview(result)
    assert result.preview_sha256 == ("0" * 64 if target == "preview_hash" else original_hash)


def test_validation_never_depends_on_original_files_or_follows_source_paths(api, records, tmp_path):
    result = preview(api, records)
    for record in records:
        record.processed_path.rename(record.processed_path.with_suffix(".moved"))
    for source in result.sources:
        source["processed_path"] = str(tmp_path / "does-not-exist" / "private.parquet")
    api.validate_research_preview(result)
    assert api.research_preview_sha256(result) == result.preview_sha256


def test_build_and_validation_leave_source_files_untouched(api, records):
    paths = [r.processed_path for r in records] + [r.raw_version_dir / "metadata.json" for r in records]
    before = {path: path.read_bytes() for path in paths}
    result = preview(api, records)
    api.validate_research_preview(result)
    assert {path: path.read_bytes() for path in paths} == before
