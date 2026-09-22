"""目录测试使用真实导入/版本文件，仅行情网络边界使用合成 Provider。"""

from datetime import date, datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from quantlab.data import catalog
from quantlab.data.importer import import_market_csv
from quantlab.data.market_service import MarketDataService
from quantlab.data.providers.base import MarketDataProvider


@pytest.fixture
def catalog_api():
    return catalog


def version_dir(imported):
    return Path(imported.version.archived_path).parent


def change_json(path, **updates):
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_catalog_keeps_all_csv_versions_and_reports_unknown_adjustment(catalog_api, imported_pair):
    raw, processed, imports = imported_pair
    scan = catalog_api.scan_market_catalog(processed, raw)
    assert not scan.issues
    assert [r.symbol for r in scan.records] == ["159915", "510300"]
    assert {r.version_id for r in scan.records} == {r.version.version_id for r in imports}
    for record in scan.records:
        assert record.provider == "csv"
        assert record.adjustment is None
        assert record.row_count == 3
        assert record.processed_path.is_absolute()
        data, digest = catalog_api.read_market_version(record)
        assert data.symbol.unique().tolist() == [record.symbol]
        assert digest == sha256(record.processed_path.read_bytes()).hexdigest()
        data.loc[0, "close"] = 999
        assert pd.read_parquet(record.processed_path).loc[0, "close"] == 3.1


def test_empty_catalog_is_usable_before_first_import(catalog_api, tmp_path):
    scan = catalog_api.scan_market_catalog(tmp_path / "processed", tmp_path / "raw")
    assert scan.records == ()
    assert scan.issues == ()


def test_new_import_does_not_hide_older_version(catalog_api, imported_pair, tmp_path):
    raw, processed, imports = imported_pair
    newer = import_market_csv(
        tmp_path / "510300.csv", raw, processed,
        imported_at=datetime(2025, 2, 1, tzinfo=timezone.utc),
    )
    scan = catalog_api.scan_market_catalog(processed, raw)
    assert len(scan.records) == 3
    assert scan.records[0].version_id == newer.version.version_id
    old = next(r for r in scan.records if r.version_id == imports[0].version.version_id)
    assert catalog_api.read_market_version(old)[0].shape[0] == 3


@pytest.mark.parametrize("failure", [
    "missing_marker", "bad_json", "non_object", "bad_version_object", "symbol",
    "row_count", "boolean_row_count", "version_id", "metadata_version", "naive_time",
    "invalid_time", "missing_metadata", "path_outside", "path_wrong_version",
])
def test_invalid_version_is_isolated_without_hiding_healthy_one(catalog_api, imported_pair, tmp_path, failure):
    raw, processed, imports = imported_pair
    bad, good = imports
    directory = version_dir(bad)
    commit = directory / "import_result.json"
    metadata = directory / "metadata.json"
    if failure == "missing_marker":
        commit.rename(directory / "unfinished.json")
    elif failure == "bad_json":
        commit.write_text("{unfinished", encoding="utf-8")
    elif failure == "non_object":
        commit.write_text("[]", encoding="utf-8")
    elif failure == "bad_version_object":
        change_json(commit, version="not-an-object")
    elif failure == "symbol":
        change_json(commit, symbol="999999")
    elif failure == "row_count":
        change_json(commit, row_count=-1)
    elif failure == "boolean_row_count":
        change_json(commit, row_count=True)
    elif failure == "version_id":
        change_json(commit, version={"version_id": "wrong-version"})
    elif failure == "metadata_version":
        change_json(metadata, version_id="wrong-version")
    elif failure == "naive_time":
        change_json(metadata, imported_at="2025-01-10T00:00:00")
    elif failure == "invalid_time":
        change_json(metadata, imported_at="not-a-date")
    elif failure == "missing_metadata":
        metadata.rename(directory / "missing-metadata.json")
    elif failure == "path_outside":
        change_json(commit, processed_path=str(tmp_path / "outside.parquet"))
    else:
        change_json(commit, processed_path=good.processed_path)
    scan = catalog_api.scan_market_catalog(processed, raw)
    assert [r.version_id for r in scan.records] == [good.version.version_id]
    assert any(bad.version.version_id in issue for issue in scan.issues)


def test_duplicate_commit_markers_are_ambiguous_not_first_match_wins(catalog_api, imported_pair):
    raw, processed, imports = imported_pair
    bad, good = imports
    shutil.copytree(version_dir(bad), raw / "duplicate" / bad.version.version_id)
    scan = catalog_api.scan_market_catalog(processed, raw)
    assert [r.version_id for r in scan.records] == [good.version.version_id]
    assert any(bad.version.version_id in issue for issue in scan.issues)


def test_source_format_is_read_without_claiming_a_price_adjustment(catalog_api, imported_pair):
    raw, processed, imports = imported_pair
    (version_dir(imports[0]) / "source_format.json").write_text(json.dumps({
        "detected_source": "eastmoney", "source_label": "东方财富", "encoding": "utf-8",
        "field_mapping": {}, "original_path": "source_original/510300.csv",
    }), encoding="utf-8")
    entry = next(r for r in catalog_api.scan_market_catalog(processed, raw).records if r.symbol == "510300")
    assert entry.provider == "eastmoney"
    assert entry.adjustment is None


class SyntheticProvider(MarketDataProvider):
    key = "fixture"
    name = "Synthetic, not market data"

    def __init__(self, data):
        self.data = data

    def fetch_etf_history(self, symbol, start_date, end_date, period="daily", adjustment="none"):
        assert symbol == "510300"
        return self.data.drop(columns="symbol").copy(deep=True)


@pytest.fixture
def provider_version(tmp_path, sample_assets):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    service = MarketDataService(SyntheticProvider(sample_assets["510300"]), raw, processed)
    fetched = service.fetch_etf_history("510300", date(2025, 1, 1), date(2025, 1, 7), adjustment="qfq")
    return raw, processed, fetched


def test_provider_record_retains_known_adjustment(catalog_api, provider_version):
    raw, processed, fetched = provider_version
    scan = catalog_api.scan_market_catalog(processed, raw)
    assert not scan.issues
    assert len(scan.records) == 1
    record = scan.records[0]
    assert record.version_id == fetched.metadata.data_version
    assert record.provider == "fixture"
    assert record.adjustment == "qfq"
    data, _ = catalog_api.read_market_version(record)
    pd.testing.assert_frame_equal(data, fetched.data)


@pytest.mark.parametrize(("field", "value"), [
    ("data_version", "other"), ("processed_path", "outside.parquet"),
    ("symbol", "159915"), ("row_count", 77), ("period", "hourly"),
    ("adjustment", "unknown"), ("fetched_at", "2025-01-02"),
])
def test_provider_metadata_conflicts_are_rejected(catalog_api, provider_version, field, value):
    raw, processed, fetched = provider_version
    path = raw / "fixture" / fetched.metadata.data_version / "provider_metadata.json"
    change_json(path, **{field: value})
    scan = catalog_api.scan_market_catalog(processed, raw)
    assert scan.records == ()
    assert any(fetched.metadata.data_version in issue for issue in scan.issues)


def test_scan_rejects_processed_file_symlink_outside_root(catalog_api, imported_pair, tmp_path):
    raw, processed, imports = imported_pair
    path = Path(imports[0].processed_path)
    moved = tmp_path / "outside.parquet"
    path.rename(moved)
    path.symlink_to(moved)
    scan = catalog_api.scan_market_catalog(processed, raw)
    assert [r.symbol for r in scan.records] == ["159915"]
    assert scan.issues


def test_scan_does_not_follow_metadata_symlink_outside_root(catalog_api, imported_pair, tmp_path):
    raw, processed, imports = imported_pair
    path = version_dir(imports[0]) / "metadata.json"
    moved = tmp_path / "private-metadata.json"
    path.rename(moved)
    path.symlink_to(moved)
    scan = catalog_api.scan_market_catalog(processed, raw)
    assert [r.symbol for r in scan.records] == ["159915"]
    assert scan.issues


def test_reader_rejects_symlink_replacement_after_selection(catalog_api, imported_pair, tmp_path):
    raw, processed, _ = imported_pair
    record = catalog_api.scan_market_catalog(processed, raw).records[0]
    moved = tmp_path / "replacement.parquet"
    record.processed_path.rename(moved)
    record.processed_path.symlink_to(moved)
    with pytest.raises(ValueError, match="路径|更改|改变"):
        catalog_api.read_market_version(record)


def test_reader_rejects_metadata_change_after_selection(catalog_api, provider_version):
    raw, processed, _ = provider_version
    record = catalog_api.scan_market_catalog(processed, raw).records[0]
    change_json(record.raw_version_dir / "provider_metadata.json", adjustment="hfq")
    with pytest.raises(ValueError, match="更改|改变|一致"):
        catalog_api.read_market_version(record)


@pytest.mark.parametrize("failure", ["wrong_symbol", "numeric_symbol", "missing_symbol", "row_count", "corrupt"])
def test_reader_checks_the_actual_file_identity(catalog_api, imported_pair, failure):
    raw, processed, _ = imported_pair
    record = catalog_api.scan_market_catalog(processed, raw).records[0]
    data = pd.read_parquet(record.processed_path)
    if failure == "wrong_symbol":
        data["symbol"] = "999999"
    elif failure == "numeric_symbol":
        data["symbol"] = 159915
    elif failure == "missing_symbol":
        data = data.drop(columns="symbol")
    elif failure == "row_count":
        data = data.iloc[:-1]
    else:
        record.processed_path.write_bytes(b"not-parquet")
    if failure != "corrupt":
        data.to_parquet(record.processed_path, index=False)
    with pytest.raises(ValueError):
        catalog_api.read_market_version(record)


def test_catalog_orders_times_by_instant_not_timezone_text(catalog_api, imported_pair):
    raw, processed, imports = imported_pair
    # 00:30+08 is earlier than 00:00 UTC on the same displayed date.
    change_json(version_dir(imports[0]) / "metadata.json", imported_at="2025-01-10T00:30:00+08:00")
    change_json(version_dir(imports[1]) / "metadata.json", imported_at="2025-01-10T00:00:00+00:00")
    records = catalog_api.scan_market_catalog(processed, raw).records
    assert [r.symbol for r in records] == ["159915", "510300"]


def test_catalog_keeps_legacy_relative_paths_usable(catalog_api, tmp_path, sample_assets, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sample_assets["510300"].to_csv("source.csv", index=False)
    imported = import_market_csv("source.csv", Path("data/raw"), Path("data/processed"))
    assert not Path(imported.processed_path).is_absolute()
    scan = catalog_api.scan_market_catalog(Path("data/processed"), Path("data/raw"))
    assert not scan.issues
    assert len(scan.records) == 1
    data, _ = catalog_api.read_market_version(scan.records[0])
    assert data.symbol.tolist() == ["510300"] * 3


def test_csv_source_metadata_added_after_selection_invalidates_record(catalog_api, imported_pair):
    raw, processed, _ = imported_pair
    record = catalog_api.scan_market_catalog(processed, raw).records[0]
    (record.raw_version_dir / "source_format.json").write_text(
        json.dumps({"detected_source": "eastmoney"}), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="改变"):
        catalog_api.read_market_version(record)


def test_conflicting_csv_and_provider_metadata_is_not_silently_prioritized(catalog_api, provider_version):
    raw, processed, fetched = provider_version
    directory = raw / "fixture" / fetched.metadata.data_version
    (directory / "metadata.json").write_text(json.dumps({
        "version_id": fetched.metadata.data_version,
        "imported_at": "2025-01-10T00:00:00+00:00",
    }), encoding="utf-8")
    scan = catalog_api.scan_market_catalog(processed, raw)
    assert scan.records == ()
    assert any(fetched.metadata.data_version in issue for issue in scan.issues)


@pytest.mark.parametrize("target", ["parquet", "metadata", "record_path"])
def test_symlink_loop_is_reported_as_invalid_version(catalog_api, imported_pair, target):
    raw, processed, imports = imported_pair
    scan = catalog_api.scan_market_catalog(processed, raw)
    record = next(r for r in scan.records if r.symbol == "510300")
    path = (version_dir(imports[0]) / "metadata.json"
            if target == "metadata" else record.processed_path)
    path.rename(path.with_suffix(".original"))
    path.symlink_to(path.name)
    if target == "record_path":
        with pytest.raises(ValueError, match="路径|链接"):
            catalog_api.read_market_version(record)
    else:
        scan = catalog_api.scan_market_catalog(processed, raw)
        assert [r.symbol for r in scan.records] == ["159915"]
        assert any(record.version_id in issue for issue in scan.issues)
