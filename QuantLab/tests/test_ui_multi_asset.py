"""页面真实读写临时合成版本；仅替换存储根与 AppTest 不支持的上传边界。"""

from dataclasses import replace
from datetime import date, datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from quantlab.data.calendar import parse_calendar_csv
from quantlab.data.catalog import scan_market_catalog
from quantlab.data.importer import import_market_csv
from quantlab.data.research_sets import DatasetSelection
from quantlab.data.research_storage import list_research_sets, load_research_set
from quantlab.ui import runtime
from quantlab.ui import multi_asset


PAGE = "pages/6_多ETF数据.py"
CALENDAR = b"date\n2025-01-02\n2025-01-03\n2025-01-06\n2025-01-07\n2025-01-08\n"


@pytest.fixture
def api():
    return multi_asset


@pytest.fixture
def research_request(imported_pair):
    raw, processed, _ = imported_pair
    return dict(name="合成示例", selections=tuple(DatasetSelection(r, "none") for r in scan_market_catalog(processed, raw).records),
                start=date(2025, 1, 2), end=date(2025, 1, 7), calendar=None)


@pytest.mark.parametrize("field", ["name", "start", "end", "version", "metadata", "declaration", "calendar"])
def test_changed_request_cannot_save_old_preview(api, research_request, field):
    original = api.research_request_sha256(**research_request)
    changed = dict(research_request)
    if field == "name":
        changed["name"] = "另一集合"
    elif field in ("start", "end"):
        changed[field] = date(2025, 1, 3)
    elif field == "calendar":
        changed["calendar"] = parse_calendar_csv(CALENDAR, source="合成", coverage_start=date(2025, 1, 1), coverage_end=date(2025, 1, 10))
    else:
        first, second = research_request["selections"]
        if field == "version":
            first = replace(first, record=replace(first.record, version_id="new-version"))
        elif field == "metadata":
            first = replace(first, record=replace(first.record, metadata_sha256="0" * 64))
        else:
            first = replace(first, declared_adjustment="qfq")
        changed["selections"] = (first, second)
    new = api.research_request_sha256(**changed)
    assert api.can_save_preview(original, original)
    assert not api.can_save_preview(new, original)
    assert not api.can_save_preview(original, None)


def test_request_sorting_is_deterministic_and_calendar_declarations_matter(api, research_request):
    original = api.research_request_sha256(**research_request)
    research_request["selections"] = tuple(reversed(research_request["selections"]))
    assert api.research_request_sha256(**research_request) == original
    cal = parse_calendar_csv(CALENDAR, source="合成", coverage_start=date(2025, 1, 1), coverage_end=date(2025, 1, 10))
    research_request["calendar"] = cal
    original = api.research_request_sha256(**research_request)
    for changed in (replace(cal, source="不同来源"), replace(cal, coverage_end=date(2025, 1, 11)), replace(cal, file_sha256="0" * 64)):
        research_request["calendar"] = changed
        assert api.research_request_sha256(**research_request) != original


@pytest.mark.parametrize(("current", "old"), [("", ""), (None, None), ("wrong", "wrong"), ("0" * 64, None)])
def test_missing_or_invalid_request_never_enables_save(api, current, old):
    assert not api.can_save_preview(current, old)


@pytest.mark.parametrize(("error", "phrase"), [(ValueError("合成冲突"), "校验"), (FileNotFoundError("missing"), "找不到"), (OSError("disk full"), "读写")])
def test_errors_have_user_facing_category_and_optional_detail(api, error, phrase):
    message, detail = api.friendly_research_error(error)
    assert phrase in message
    assert "Traceback" not in message
    assert isinstance(detail, str)


def test_runtime_uses_local_data_root(tmp_path, monkeypatch):
    assert hasattr(runtime, "get_research_storage_dir"), "缺少研究集合目录边界"
    monkeypatch.delenv("QUANTLAB_PUBLIC_MODE", raising=False)
    assert runtime.get_research_storage_dir(tmp_path) == tmp_path / "data" / "research_sets"


def test_public_sessions_reuse_their_root_but_never_share_sets(tmp_path, monkeypatch):
    assert hasattr(runtime, "get_research_storage_dir"), "缺少研究集合目录边界"
    monkeypatch.setenv("QUANTLAB_PUBLIC_MODE", "1")
    first, second = {}, {}
    try:
        monkeypatch.setattr(st, "session_state", first)
        a = runtime.get_research_storage_dir(tmp_path)
        assert runtime.get_research_storage_dir(tmp_path) == a
        a.mkdir(parents=True)
        (a / "only-first-session.txt").write_text("synthetic")
        monkeypatch.setattr(st, "session_state", second)
        b = runtime.get_research_storage_dir(tmp_path)
        assert a != b and not b.is_relative_to(tmp_path)
        assert not (b / "only-first-session.txt").exists()
        assert runtime.get_research_storage_dir(tmp_path) == b
    finally:
        for state in (first, second):
            if "_quantlab_public_temporary_directory" in state:
                state["_quantlab_public_temporary_directory"].cleanup()


@pytest.fixture
def page_root(imported_pair, tmp_path, monkeypatch):
    raw, processed, _ = imported_pair
    root = tmp_path / "sets"
    monkeypatch.delenv("QUANTLAB_ACCESS_PASSWORD", raising=False)
    monkeypatch.delenv("QUANTLAB_PUBLIC_MODE", raising=False)
    monkeypatch.setattr(runtime, "get_market_storage_dirs", lambda project: (raw, processed))
    monkeypatch.setattr(runtime, "get_research_storage_dir", lambda project: root, raising=False)
    return root


def new_page():
    assert Path(PAGE).is_file(), "多 ETF 页面尚未实现"
    page = AppTest.from_file(PAGE, default_timeout=15).run()
    assert not page.exception
    return page


def configured_page():
    page = new_page()
    page.text_input(key="research_name").set_value("合成验收集合")
    page.multiselect(key="research_symbols").set_value(["510300", "159915"]).run()
    for symbol in ("510300", "159915"):
        page.selectbox(key=f"research_adjustment_{symbol}").set_value("none")
    page.date_input(key="research_start").set_value(date(2025, 1, 2))
    page.date_input(key="research_end").set_value(date(2025, 1, 7)).run()
    return page


def generate(page):
    page.button(key="research_preview").click().run()
    assert not page.exception and not page.error
    assert not page.button(key="research_save").disabled
    return page


def test_real_page_preview_save_load_and_repeated_save(page_root):
    page = configured_page()
    assert page.button(key="research_save").disabled
    assert not page_root.exists(), "配置本身不生成研究快照"
    generate(page)
    assert any("用户声明" in w.value for w in page.warning)
    assert any("未经" in w.value for w in page.warning)
    page.button(key="research_save").click().run()
    assert not page.exception and page.success
    scan = list_research_sets(page_root)
    assert len(scan.records) == 1
    page.selectbox(key="research_saved_id").set_value(scan.records[0].set_id)
    page.button(key="research_load").click().run()
    assert not page.exception and page.success
    loaded = page.session_state["research_loaded_set"]
    assert len(loaded.preview.alignment.data) == 8
    page.button(key="research_save").click().run()
    assert len(list_research_sets(page_root).records) == 2


@pytest.mark.parametrize("field", ["name", "date", "adjustment", "symbols", "version", "calendar"])
def test_changed_widgets_disable_and_server_rejects_old_preview(page_root, imported_pair, tmp_path, field):
    if field == "version":
        raw, processed, _ = imported_pair
        import_market_csv(tmp_path / "510300.csv", raw, processed, imported_at=datetime(2025, 2, 1, tzinfo=timezone.utc))
    page = generate(configured_page())
    if field == "name":
        page.text_input(key="research_name").set_value("新名称").run()
    elif field == "date":
        page.date_input(key="research_start").set_value(date(2025, 1, 3)).run()
    elif field == "adjustment":
        page.selectbox(key="research_adjustment_510300").set_value("qfq").run()
    elif field == "symbols":
        page.multiselect(key="research_symbols").set_value(["510300"]).run()
    elif field == "version":
        raw, processed, _ = imported_pair
        options = [r.version_id for r in scan_market_catalog(processed, raw).records if r.symbol == "510300"]
        current = page.selectbox(key="research_version_510300").value
        page.selectbox(key="research_version_510300").set_value(next(v for v in options if v != current)).run()
    else:
        page.radio(key="research_calendar_mode").set_value("supplied").run()
    assert page.button(key="research_save").disabled
    # 即使绕过浏览器禁用按钮发送事件，保存分支也要验证服务端状态。
    page.button(key="research_save").click().run()
    assert not page.exception and not page.success
    assert list_research_sets(page_root).records == ()


def test_supplied_calendar_and_changed_upload_invalidates_preview(page_root, monkeypatch):
    upload = BytesIO(CALENDAR)
    upload.name = "synthetic-calendar.csv"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    page = configured_page()
    page.radio(key="research_calendar_mode").set_value("supplied").run()
    page.text_input(key="research_calendar_source").set_value("合成测试日历")
    page.date_input(key="research_calendar_start").set_value(date(2025, 1, 1))
    page.date_input(key="research_calendar_end").set_value(date(2025, 1, 10))
    page.date_input(key="research_end").set_value(date(2025, 1, 8)).run()
    generate(page)
    preview = page.session_state["research_preview_data"]
    assert len(preview.alignment.data) == 10
    assert not preview.alignment.data.loc[lambda d: d.date.eq("2025-01-08"), "has_observation"].any()
    page.button(key="research_save").click().run()
    assert page.success
    page.button(key="research_load").click().run()
    assert page.session_state["research_loaded_set"].preview.alignment.calendar.mode == "supplied"
    upload = BytesIO(CALENDAR + b"2025-01-09\n")
    upload.name = "synthetic-calendar.csv"
    page.run()
    assert page.button(key="research_save").disabled


def test_missing_declaration_is_actionable_not_exception(page_root):
    page = new_page()
    page.multiselect(key="research_symbols").set_value(["510300", "159915"]).run()
    page.button(key="research_preview").click().run()
    assert not page.exception and page.error
    assert page.button(key="research_save").disabled


def test_no_common_dates_is_warning_not_failure(page_root):
    page = configured_page()
    page.date_input(key="research_end").set_value(date(2025, 1, 2)).run()
    generate(page)
    assert any("共同日期" in w.value for w in page.warning)
    page.button(key="research_save").click().run()
    assert page.success and len(list_research_sets(page_root).records) == 1


def test_save_failure_does_not_claim_success_or_discard_preview(page_root):
    page = generate(configured_page())
    before = page.session_state["research_preview_data"].preview_sha256
    page_root.write_text("not-a-directory")
    page.button(key="research_save").click().run()
    assert not page.exception and page.error and not page.success
    assert page.session_state["research_preview_data"].preview_sha256 == before


def test_corrupt_saved_set_never_leaves_old_loaded_result_visible(page_root):
    page = generate(configured_page())
    page.button(key="research_save").click().run()
    page.button(key="research_load").click().run()
    saved = page.session_state["research_loaded_set"].record
    (saved.directory / "coverage.csv").write_text("broken")
    page.run()
    assert not page.exception and page.warning
    assert page.session_state.filtered_state.get("research_loaded_set") is None
    assert not any("已载入：" in heading.value for heading in page.subheader)


def test_source_changes_after_preview_do_not_replace_snapshot_at_save(page_root, imported_pair):
    page = generate(configured_page())
    raw, processed, _ = imported_pair
    for record in scan_market_catalog(processed, raw).records:
        data = pd.read_parquet(record.processed_path)
        data["close"] = 3.15
        data.to_parquet(record.processed_path, index=False)
    page.button(key="research_save").click().run()
    saved = list_research_sets(page_root).records[0]
    loaded = load_research_set(page_root, saved.set_id)
    assert loaded.preview.alignment.data.loc[lambda d: d.has_observation, "close"].eq(3.1).all()


def test_empty_catalog_guides_to_data_import(page_root, imported_pair):
    _, processed, _ = imported_pair
    processed.rename(processed.with_name("moved-processed"))
    page = new_page()
    assert any("添加资产数据" in item.value for item in page.info)
    assert not page.error


def test_navigation_exposes_multi_etf_page(page_root, imported_pair, monkeypatch):
    assert Path(PAGE).is_file(), "多 ETF 页面尚未实现"
    import quantlab.ui as ui
    raw, processed, _ = imported_pair
    monkeypatch.setattr(ui, "get_market_storage_dirs", lambda project: (raw, processed))
    page = AppTest.from_file("app.py", default_timeout=15).run()
    page.switch_page(PAGE).run()
    assert not page.exception
    assert page.multiselect(key="research_symbols")
