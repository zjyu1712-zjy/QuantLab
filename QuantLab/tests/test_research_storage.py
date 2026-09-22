"""仅使用临时合成文件，验证快照一致性、故障发布边界和不跟随来源路径。"""

from dataclasses import replace
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
import pytest

from quantlab.data.calendar import parse_calendar_csv
from quantlab.data.catalog import scan_market_catalog
from quantlab.data.importer import import_market_csv
from quantlab.data.research_sets import DatasetSelection, build_research_set
from quantlab.data import research_storage


@pytest.fixture
def api():
    return research_storage


def manifest(saved):
    return json.loads((saved.directory / "manifest.json").read_text(encoding="utf-8"))


def write_manifest(saved, content):
    (saved.directory / "manifest.json").write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")


def update_file_hash(saved, filename):
    content = manifest(saved)
    content["files"][filename] = sha256((saved.directory / filename).read_bytes()).hexdigest()
    write_manifest(saved, content)


def test_saved_set_does_not_need_original_files(api, tmp_path, research_preview):
    root = tmp_path / "sets"
    saved = api.save_research_set(research_preview, root, name="  合成示例  ")
    assert saved.name == "合成示例"
    assert str(UUID(saved.set_id)) == saved.set_id
    assert saved.directory == root / saved.set_id
    assert saved.created_at.endswith("+00:00")
    for source in research_preview.sources:
        path = Path(source["processed_path"])
        path.rename(path.with_suffix(".moved"))
    restored = api.load_research_set(root, saved.set_id)
    assert restored.record == saved
    assert restored.preview.sources == research_preview.sources
    assert restored.preview.preview_sha256 == research_preview.preview_sha256
    assert restored.preview.alignment.calendar == research_preview.alignment.calendar
    pd.testing.assert_frame_equal(restored.preview.alignment.data, research_preview.alignment.data)
    pd.testing.assert_frame_equal(restored.preview.coverage, research_preview.coverage, check_exact=True)
    assert api.list_research_sets(root).records == (saved,)
    assert api.list_research_sets(root).issues == ()


def test_manifest_keeps_only_fixed_traceability_and_environment_fields(api, tmp_path, research_preview, monkeypatch):
    secret = "synthetic-secret-never-export-this-987654"
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", secret)
    monkeypatch.setenv("CUSTOM_PRIVATE_SETTING", secret)
    saved = api.save_research_set(research_preview, tmp_path / "sets", name="版本追踪")
    content = manifest(saved)
    assert set(content) == {
        "rule_version", "set_id", "name", "created_at", "start", "end", "adjustment", "sources",
        "calendar", "common_dates", "warnings", "alignment_warnings", "alignment_content_sha256",
        "preview_sha256", "files", "environment",
    }
    assert content["rule_version"] == "multi-etf-alignment-v1"
    assert set(content["environment"]) == {"python", "pandas", "numpy", "pyarrow"}
    assert all(isinstance(v, str) and v for v in content["environment"].values())
    assert set(content["files"]) == {"aligned.parquet", "coverage.csv"}
    assert content["sources"] == list(research_preview.sources)
    assert content["common_dates"] == ["2025-01-03", "2025-01-07"]
    assert set(path.name for path in saved.directory.iterdir()) == {"manifest.json", "coverage.csv", "aligned.parquet"}
    for path in saved.directory.iterdir():
        assert secret.encode() not in path.read_bytes()
    assert b"NaN" not in (saved.directory / "manifest.json").read_bytes()


def test_repeated_save_appends_distinct_ids_without_altering_earlier_bytes(api, tmp_path, research_preview):
    root = tmp_path / "sets"
    first = api.save_research_set(research_preview, root, name="同名")
    old_bytes = {p.name: p.read_bytes() for p in first.directory.iterdir()}
    second = api.save_research_set(research_preview, root, name="同名")
    assert first.set_id != second.set_id
    assert {p.name: p.read_bytes() for p in first.directory.iterdir()} == old_bytes
    assert api.load_research_set(root, second.set_id).preview.preview_sha256 == research_preview.preview_sha256
    assert api.list_research_sets(root).records == (second, first)


def test_uuid_collision_never_overwrites_existing_set(api, tmp_path, research_preview, monkeypatch):
    save = api.save_research_set
    from quantlab.data import research_storage
    fixed = UUID("00000000-0000-4000-8000-000000000001")
    monkeypatch.setattr(research_storage, "uuid4", lambda: fixed)
    saved = save(research_preview, tmp_path / "sets", name="第一份")
    old_bytes = {p.name: p.read_bytes() for p in saved.directory.iterdir()}
    with pytest.raises(FileExistsError):
        save(research_preview, tmp_path / "sets", name="碰撞的第二份")
    assert {p.name: p.read_bytes() for p in saved.directory.iterdir()} == old_bytes
    assert api.list_research_sets(tmp_path / "sets").records == (saved,)


@pytest.mark.parametrize("name", ["", "  \n ", None, 12])
def test_invalid_name_creates_no_directory(api, tmp_path, research_preview, name):
    root = tmp_path / "sets"
    with pytest.raises(ValueError, match="名称"):
        api.save_research_set(research_preview, root, name=name)
    assert not root.exists()


def test_name_is_not_used_as_a_path(api, tmp_path, research_preview):
    root = tmp_path / "sets"
    saved = api.save_research_set(research_preview, root, name="../../这是显示名称")
    assert saved.directory.parent == root
    assert saved.directory.name == saved.set_id
    assert saved.name == "../../这是显示名称"


@pytest.mark.parametrize("mutation", ["data", "coverage", "source", "extra_source_key", "warnings"])
def test_invalid_preview_creates_no_directory(api, tmp_path, research_preview, mutation):
    if mutation == "data":
        research_preview.alignment.data.loc[lambda d: d.has_observation, "close"] = 3.15
    elif mutation == "coverage":
        research_preview.coverage.loc[0, "missing_count"] = 0
    elif mutation == "source":
        research_preview.sources[0]["version_id"] = "not-the-selected-version"
    elif mutation == "extra_source_key":
        research_preview.sources[0]["private_token"] = "synthetic"
    else:
        research_preview = replace(research_preview, warnings=())
    root = tmp_path / "sets"
    with pytest.raises(ValueError):
        api.save_research_set(research_preview, root, name="不应保存")
    assert not root.exists()


@pytest.mark.parametrize("failure", ["parquet", "coverage", "manifest_write", "manifest_publish"])
def test_failed_write_never_publishes_completed_set(api, tmp_path, research_preview, monkeypatch, failure):
    save = api.save_research_set
    root = tmp_path / "sets"
    original_open, original_replace = Path.open, Path.replace

    def fail_parquet(*args, **kwargs):
        raise OSError("synthetic write failure")

    def fail_open(path, *args, **kwargs):
        if (failure == "coverage" and path.name == "coverage.csv"
                or failure == "manifest_write" and path.name == ".manifest.tmp"):
            raise OSError("synthetic write failure")
        return original_open(path, *args, **kwargs)

    def fail_replace(path, target):
        if Path(target).name == "manifest.json":
            raise OSError("synthetic publish failure")
        return original_replace(path, target)

    with monkeypatch.context() as patch:
        if failure == "parquet":
            patch.setattr(pd.DataFrame, "to_parquet", fail_parquet)
        elif failure == "manifest_publish":
            patch.setattr(Path, "replace", fail_replace)
        else:
            patch.setattr(Path, "open", fail_open)
        with pytest.raises(OSError, match="synthetic"):
            save(research_preview, root, name="未完成")
    assert list(root.glob("*/manifest.json")) == []
    assert len(list(root.iterdir())) == 1  # 保留半成品供检查，不删除其他目录。
    scan = api.list_research_sets(root)
    assert scan.records == () and scan.issues == ()


def test_save_uses_deep_copy_if_caller_mutates_during_write(api, tmp_path, research_preview, monkeypatch):
    original_hash = research_preview.preview_sha256
    original_to_parquet = pd.DataFrame.to_parquet

    def mutate_caller(frame, *args, **kwargs):
        research_preview.alignment.data.loc[lambda d: d.has_observation, "close"] = 3.15
        research_preview.coverage.loc[0, "version_id"] = "changed-mid-save"
        research_preview.sources[0]["version_id"] = "changed-mid-save"
        return original_to_parquet(frame, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", mutate_caller)
    root = tmp_path / "sets"
    saved = api.save_research_set(research_preview, root, name="隔离快照")
    restored = api.load_research_set(root, saved.set_id).preview
    assert restored.preview_sha256 == original_hash
    assert restored.alignment.data.loc[lambda d: d.has_observation, "close"].eq(3.1).all()
    assert restored.sources[0]["version_id"] != "changed-mid-save"
    assert restored.coverage.version_id.ne("changed-mid-save").all()


@pytest.mark.parametrize("filename", ["aligned.parquet", "coverage.csv"])
def test_tampered_file_is_rejected_and_isolated(api, tmp_path, research_preview, filename):
    root = tmp_path / "sets"
    bad = api.save_research_set(research_preview, root, name="损坏")
    good = api.save_research_set(research_preview, root, name="健康")
    (bad.directory / filename).write_bytes(b"altered")
    with pytest.raises(ValueError, match="指纹|损坏"):
        api.load_research_set(root, bad.set_id)
    scan = api.list_research_sets(root)
    assert scan.records == (good,)
    assert len(scan.issues) == 1 and bad.set_id in scan.issues[0]


@pytest.mark.parametrize("mutation", [
    "malformed", "not_object", "extra_secret", "rule", "id", "empty_name", "naive_time", "invalid_time",
    "extra_environment", "missing_source", "different_basis", "calendar_date", "calendar_coverage",
    "common_dates", "warnings", "alignment_warnings", "alignment_hash", "preview_hash", "path_in_files",
    "range", "adjustment", "duplicate_key", "nan", "missing_file", "missing_field",
])
def test_bad_manifest_rejected_and_isolated(api, tmp_path, research_preview, mutation):
    root = tmp_path / "sets"
    saved = api.save_research_set(research_preview, root, name="损坏说明")
    path = saved.directory / "manifest.json"
    content = manifest(saved)
    if mutation == "malformed":
        path.write_text("{broken")
    elif mutation == "not_object":
        path.write_text("[]")
    elif mutation == "duplicate_key":
        path.write_text(path.read_text().replace('"name":', '"name": "隐藏旧值", "name":', 1))
    elif mutation == "nan":
        content["environment"]["python"] = float("nan")
        write_manifest(saved, content)
    else:
        if mutation == "extra_secret":
            content["secret"] = "unrecognized field"
        elif mutation == "rule":
            content["rule_version"] = "future-unsupported-version"
        elif mutation == "id":
            content["set_id"] = str(uuid4())
        elif mutation == "empty_name":
            content["name"] = " "
        elif mutation == "naive_time":
            content["created_at"] = "2025-01-01T00:00:00"
        elif mutation == "invalid_time":
            content["created_at"] = "not-a-date"
        elif mutation == "extra_environment":
            content["environment"]["private"] = "not-a-package-version"
        elif mutation == "missing_source":
            content["sources"].pop()
        elif mutation == "different_basis":
            content["sources"][0]["adjustment_basis"] = "provider"
        elif mutation == "calendar_date":
            content["calendar"]["dates"][0] = "2025-01-02T09:30:00"
        elif mutation == "calendar_coverage":
            content["calendar"]["coverage_end"] = "2025-01-03"
        elif mutation == "common_dates":
            content["common_dates"] = ["2025-01-02"]
        elif mutation == "warnings":
            content["warnings"] = []
        elif mutation == "alignment_warnings":
            content["alignment_warnings"] = []
        elif mutation == "alignment_hash":
            content["alignment_content_sha256"] = "0" * 64
        elif mutation == "preview_hash":
            content["preview_sha256"] = "0" * 64
        elif mutation == "path_in_files":
            content["files"]["aligned.parquet"] = "/private/not-a-fingerprint"
        elif mutation == "range":
            content["start"] = "2025-01-03"
        elif mutation == "adjustment":
            content["adjustment"] = "qfq"
        elif mutation == "missing_file":
            content["files"].pop("coverage.csv")
        else:
            content.pop("calendar")
        write_manifest(saved, content)
    with pytest.raises(ValueError):
        api.load_research_set(root, saved.set_id)
    scan = api.list_research_sets(root)
    assert scan.records == () and len(scan.issues) == 1


@pytest.mark.parametrize("mutation", ["price", "fill", "grid", "coverage", "dtype", "corrupt_parquet", "corrupt_csv"])
def test_updating_byte_hash_does_not_bypass_semantic_checks(api, tmp_path, research_preview, mutation):
    root = tmp_path / "sets"
    saved = api.save_research_set(research_preview, root, name="内容校验")
    filename = "coverage.csv" if mutation in ("coverage", "corrupt_csv") else "aligned.parquet"
    path = saved.directory / filename
    if mutation == "coverage":
        frame = pd.read_csv(path, dtype={"symbol": str})
        frame.loc[0, "missing_count"] = 0
        frame.to_csv(path, index=False)
    elif mutation.startswith("corrupt_"):
        path.write_bytes(b"not valid data")
    else:
        frame = pd.read_parquet(path)
        if mutation == "price":
            frame.loc[frame.has_observation, "close"] = 3.15
        elif mutation == "fill":
            frame.loc[~frame.has_observation, "volume"] = 0.0
        elif mutation == "grid":
            frame = frame.iloc[1:]
        else:
            frame["volume"] = frame.volume.astype("Float32")
        frame.to_parquet(path, index=False)
    update_file_hash(saved, filename)
    with pytest.raises(ValueError):
        api.load_research_set(root, saved.set_id)


def test_source_paths_remain_inert_local_provenance(api, tmp_path, research_preview):
    root = tmp_path / "sets"
    saved = api.save_research_set(research_preview, root, name="不跟随路径")
    content = manifest(saved)
    for source in content["sources"]:
        source["processed_path"] = str(tmp_path / "missing-private-file.parquet")
    write_manifest(saved, content)
    loaded = api.load_research_set(root, saved.set_id)
    assert loaded.preview.preview_sha256 == research_preview.preview_sha256
    assert len(loaded.preview.alignment.data) == 8


@pytest.mark.parametrize("set_id", ["../escape", "/absolute", "", None, 42,
    "00000000000040008000000000000001", "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA"])
def test_noncanonical_set_id_rejected(api, tmp_path, set_id):
    with pytest.raises(ValueError, match="ID|UUID"):
        api.load_research_set(tmp_path / "sets", set_id)


@pytest.mark.parametrize("target", ["directory", "manifest.json", "aligned.parquet", "coverage.csv"])
def test_symlink_escape_rejected_without_loading_other_location(api, tmp_path, research_preview, target):
    root = tmp_path / "sets"
    saved = api.save_research_set(research_preview, root, name="拒绝链接")
    path = saved.directory if target == "directory" else saved.directory / target
    moved = tmp_path / f"moved-{target}"
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=target == "directory")
    with pytest.raises(ValueError, match="链接|越界"):
        api.load_research_set(root, saved.set_id)
    scan = api.list_research_sets(root)
    assert scan.records == () and len(scan.issues) == 1


def test_missing_root_is_empty_and_unpublished_folder_is_not_a_set(api, tmp_path):
    root = tmp_path / "sets"
    assert api.list_research_sets(root).records == ()
    assert not root.exists()
    partial = root / str(uuid4())
    partial.mkdir(parents=True)
    (partial / ".manifest.tmp").write_text("{half-written")
    assert api.list_research_sets(root).records == ()
    assert api.list_research_sets(root).issues == ()
    with pytest.raises(ValueError, match="未完成|缺少"):
        api.load_research_set(root, partial.name)


def test_loaded_objects_are_independent_of_later_loads(api, tmp_path, research_preview):
    root = tmp_path / "sets"
    saved = api.save_research_set(research_preview, root, name="独立重读")
    first = api.load_research_set(root, saved.set_id)
    first.preview.alignment.data.loc[lambda d: d.has_observation, "close"] = 3.15
    first.preview.sources[0]["provider"] = "changed"
    second = api.load_research_set(root, saved.set_id)
    assert second.preview.alignment.data.loc[lambda d: d.has_observation, "close"].eq(3.1).all()
    assert second.preview.sources[0]["provider"] == "csv"


@pytest.mark.parametrize("mode", ["union", "supplied", "all_missing_asset"])
def test_roundtrip_preserves_calendar_empty_dates_and_fraction_precision(api, tmp_path, imported_pair, mode):
    raw, processed, _ = imported_pair
    selected = tuple(DatasetSelection(r, "none") for r in scan_market_catalog(processed, raw).records)
    start, end, calendar = date(2025, 1, 2), date(2025, 1, 7), None
    if mode == "all_missing_asset":
        end = start
    elif mode == "union":
        end = date(2025, 1, 6)  # 三个轴日期中缺一条：1/3 必须精确回读。
    else:
        end = date(2025, 1, 8)
        calendar = parse_calendar_csv(
            b"date\n2025-01-02\n2025-01-03\n2025-01-06\n2025-01-07\n2025-01-08\n",
            source="合成日历", coverage_start=date(2025, 1, 1), coverage_end=date(2025, 1, 10),
        )
    original = build_research_set(selected, start=start, end=end, calendar=calendar)
    root = tmp_path / "sets"
    saved = api.save_research_set(original, root, name="类型往返")
    restored = api.load_research_set(root, saved.set_id).preview
    pd.testing.assert_frame_equal(restored.coverage, original.coverage, check_exact=True)
    assert restored.alignment.calendar == original.alignment.calendar
    assert restored.alignment.common_dates == original.alignment.common_dates
    assert restored.warnings == original.warnings
    assert restored.preview_sha256 == original.preview_sha256


def test_leading_zero_symbols_and_literal_na_source_survive_csv(api, tmp_path, sample_assets):
    # 合成代码仅用于检验字符串，不声称这些代码是真实 ETF。
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    for frame, symbol in zip(sample_assets.values(), ("000001", "000002")):
        frame = frame.assign(symbol=symbol)
        source = tmp_path / f"{symbol}.csv"
        frame.to_csv(source, index=False)
        imported = import_market_csv(source, raw, processed)
        source_format = Path(imported.version.archived_path).parent / "source_format.json"
        source_format.write_text(json.dumps({"detected_source": "NA"}), encoding="utf-8")
    selected = tuple(DatasetSelection(r, "none") for r in scan_market_catalog(processed, raw).records)
    original = build_research_set(selected, start=date(2025, 1, 2), end=date(2025, 1, 7))
    root = tmp_path / "sets"
    saved = api.save_research_set(original, root, name="文本类型")
    restored = api.load_research_set(root, saved.set_id).preview
    assert restored.coverage.symbol.tolist() == ["000001", "000002"]
    assert restored.coverage.provider.tolist() == ["NA", "NA"]
    assert restored.coverage.version_id.tolist() == original.coverage.version_id.tolist()


def test_listing_uses_actual_creation_instant_not_name_or_lexical_timestamp(api, tmp_path, research_preview):
    root = tmp_path / "sets"
    first = api.save_research_set(research_preview, root, name="Z")
    second = api.save_research_set(research_preview, root, name="A")
    early = manifest(first)
    early["created_at"] = "2025-01-01T01:00:00+00:00"
    write_manifest(first, early)
    later = manifest(second)
    later["created_at"] = "2025-01-01T02:00:00+00:00"
    write_manifest(second, later)
    assert [record.set_id for record in api.list_research_sets(root).records] == [second.set_id, first.set_id]
