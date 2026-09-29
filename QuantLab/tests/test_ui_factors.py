"""因子页使用真实合成导入和快照，只替换存储目录与故障边界。"""

from copy import deepcopy
from datetime import date, datetime, timezone
from itertools import count
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from quantlab.data.catalog import scan_market_catalog
from quantlab.data.importer import import_market_csv
from quantlab.data.research_sets import DatasetSelection, build_research_set
from quantlab.data.research_storage import load_research_set, save_research_set
from quantlab.factors import run_research_factor
from quantlab.ui import runtime


PAGE = "pages/7_因子研究.py"


@pytest.fixture
def saved_factory(tmp_path, monkeypatch):
    root = tmp_path / "sets"
    sequence = count()
    monkeypatch.delenv("QUANTLAB_PUBLIC_MODE", raising=False)
    monkeypatch.delenv("QUANTLAB_ACCESS_PASSWORD", raising=False)
    monkeypatch.setattr(runtime, "get_research_storage_dir", lambda project: root)

    def make(*, gap=False, length=50, name="合成因子验收", outside=False):
        source_dir = tmp_path / f"source_{next(sequence)}"
        source_dir.mkdir()
        dates = pd.date_range("2025-01-02", periods=length, freq="B")
        for symbol, close in (("510300", 100.0 + np.arange(length) * .5),
                               ("159915", 200.0 - np.arange(length))):
            frame = pd.DataFrame({
                "date": dates + pd.Timedelta(days=365) if outside and symbol == "159915" else dates,
                "symbol": symbol, "open": close, "high": close, "low": close,
                "close": close, "volume": 0.0,
            })
            if gap and symbol == "510300":
                frame = frame.drop(index=25)
            csv = source_dir / f"{symbol}.csv"
            frame.to_csv(csv, index=False)
            import_market_csv(csv, source_dir / "raw", source_dir / "processed",
                              imported_at=datetime(2025, 6, 1, tzinfo=timezone.utc))
        scan = scan_market_catalog(source_dir / "processed", source_dir / "raw")
        preview = build_research_set(
            tuple(DatasetSelection(record, "none") for record in scan.records),
            start=dates[0].date(), end=dates[-1].date(),
        )
        saved = save_research_set(preview, root, name=name)
        return saved, preview, source_dir

    return root, make


def new_page():
    assert Path(PAGE).is_file(), "尚未实现因子研究页面"
    page = AppTest.from_file(PAGE, default_timeout=15).run()
    assert not page.exception
    return page


def calculate(page, set_id):
    page.selectbox(key="factor_set_id").set_value(set_id).run()
    page.button(key="factor_calculate").click().run()
    assert not page.exception and not page.error
    return page


def test_daily_view_keeps_invalid_rows_and_never_uses_previous_value(saved_factory):
    from quantlab.ui.factors import daily_factor_table

    _, make = saved_factory
    _, preview, _ = make(gap=True)
    result = run_research_factor(preview)
    before = deepcopy(result.data)
    dates = preview.alignment.calendar.dates
    table = daily_factor_table(result, dates[20])
    assert table["ETF 代码"].tolist() == ["159915", "510300"]
    assert table["20 期动量"].tolist() == ["-10.00%", "10.00%"]
    gap = daily_factor_table(result, dates[25]).set_index("ETF 代码")
    assert gap.loc["510300", "20 期动量"] == "—"
    assert "缺失" in gap.loc["510300", "状态 / 原因"]
    assert gap.loc["159915", "20 期动量"] != "—"
    warmup = daily_factor_table(result, dates[0])
    assert warmup["20 期动量"].eq("—").all()
    assert warmup["状态 / 原因"].str.contains("历史不足").all()
    with pytest.raises(ValueError, match="日期"):
        daily_factor_table(result, date(2020, 1, 1))
    table.loc[0, "20 期动量"] = "changed"
    pd.testing.assert_frame_equal(result.data, before)


def test_chart_splits_gap_and_keeps_real_dates_and_percent_units(saved_factory):
    from quantlab.ui.factors import factor_history, build_factor_chart

    _, make = saved_factory
    _, preview, _ = make(gap=True)
    result = run_research_factor(preview)
    history = factor_history(result, "510300")
    assert len(history) == 50
    assert history.factor_value.iloc[20] == pytest.approx(.10)
    assert history.factor_value.iloc[25:46].isna().all()
    assert history.segment.iloc[24] != history.segment.iloc[46]
    assert history.segment.iloc[20] == history.segment.iloc[24]
    assert history.segment.iloc[46] == history.segment.iloc[49]
    spec = build_factor_chart(history).to_dict()
    # 图形绑定必须用真实日期和分段，而非把空值删掉后连成一条线。
    encoding = spec["layer"][0]["encoding"]
    assert encoding["x"]["field"] == "date"
    assert encoding["y"]["field"] == "factor_value"
    assert encoding["y"]["axis"]["format"] == ".1%"
    assert encoding["detail"]["field"] == "segment"
    assert any(t.get("format") == ".2%" for t in encoding["tooltip"])
    assert len(spec["data"]["values"]) == 50
    assert spec["data"]["values"][25]["factor_value"] is None
    with pytest.raises(ValueError, match="ETF"):
        factor_history(result, "000000")


def test_page_empty_does_not_create_snapshots(saved_factory):
    root, _ = saved_factory
    page = new_page()
    assert any("多 ETF 数据" in item.value for item in page.info)
    assert not root.exists()
    assert not page.button


def test_page_real_calculation_date_asset_and_provenance(saved_factory):
    _, make = saved_factory
    saved, preview, _ = make(gap=True)
    before = {p.name: p.read_bytes() for p in saved.directory.iterdir()}
    page = new_page()
    assert not page.session_state.filtered_state.get("factor_result")
    calculate(page, saved.set_id)
    dates = preview.alignment.calendar.dates
    assert page.selectbox(key="factor_view_date").value == dates[-1]
    page.selectbox(key="factor_view_date").set_value(dates[20]).run()
    assert page.dataframe[0].value["20 期动量"].tolist() == ["-10.00%", "10.00%"]
    page.selectbox(key="factor_view_date").set_value(dates[25]).run()
    assert page.dataframe[0].value.loc[1, "20 期动量"] == "—"
    page.selectbox(key="factor_view_symbol").set_value("510300").run()
    assert not page.exception
    assert page.selectbox(key="factor_view_date").value == dates[25]
    assert len(page.get("vega_lite_chart")) == 1
    assert any(preview.preview_sha256 in item.value for item in page.code)
    assert any("close[t] / close[t-20] - 1" in item.value for item in page.code)
    assert any("总回报" in item.value for item in list(page.caption) + list(page.warning))
    assert any("前 20" in item.value for item in page.warning)
    sources = next(d.value for d in page.dataframe if "数据版本" in d.value.columns)
    assert sources["数据版本"].tolist() == [s["version_id"] for s in preview.sources]
    assert not any("processed_path" in d.value.columns for d in page.dataframe)
    assert {p.name: p.read_bytes() for p in saved.directory.iterdir()} == before


def test_switching_sets_clears_old_result_and_resets_filters(saved_factory):
    _, make = saved_factory
    first, _, _ = make()
    second, short, _ = make(length=10, name="短区间")
    page = calculate(new_page(), first.set_id)
    page.selectbox(key="factor_set_id").set_value(second.set_id).run()
    assert page.session_state.filtered_state.get("factor_result") is None
    assert not page.dataframe and not page.get("vega_lite_chart")
    calculate(page, second.set_id)
    assert page.selectbox(key="factor_view_date").value == short.alignment.calendar.dates[-1]
    assert page.dataframe[0].value["20 期动量"].eq("—").all()
    assert any("没有可用" in w.value for w in page.warning)
    assert not page.get("vega_lite_chart")


@pytest.mark.parametrize("damage", ["corrupt", "removed", "scope"])
def test_unavailable_snapshot_or_changed_scope_hides_cached_result(saved_factory, monkeypatch, tmp_path, damage):
    _, make = saved_factory
    saved, _, _ = make()
    page = calculate(new_page(), saved.set_id)
    if damage == "corrupt":
        (saved.directory / "coverage.csv").write_text("broken", encoding="utf-8")
    elif damage == "removed":
        saved.directory.rename(saved.directory.with_name("moved-outside-id"))
    else:
        monkeypatch.setattr(runtime, "get_research_storage_dir", lambda project: tmp_path / "new_scope")
    page.run()
    assert not page.exception
    assert page.session_state.filtered_state.get("factor_result") is None
    assert not page.dataframe and not page.get("vega_lite_chart")


def test_failed_recalculation_clears_old_success(saved_factory, monkeypatch):
    import quantlab.factors as factors

    _, make = saved_factory
    saved, _, _ = make()
    page = calculate(new_page(), saved.set_id)

    def fail(*args, **kwargs):
        raise ValueError("synthetic-calculation-failure")

    monkeypatch.setattr(factors, "run_research_factor", fail)
    page.button(key="factor_calculate").click().run()
    assert not page.exception and page.error
    assert page.session_state.filtered_state.get("factor_result") is None
    assert not page.dataframe and not page.get("vega_lite_chart")


def test_same_id_with_new_valid_content_requires_recalculation(saved_factory):
    root, make = saved_factory
    first, _, _ = make()
    other, changed, _ = make(length=10)
    page = calculate(new_page(), first.set_id)
    # 只替换临时合成快照，模拟外部替换文件但集合 ID 保持不变。
    manifest = json.loads((other.directory / "manifest.json").read_text())
    manifest.update(set_id=first.set_id, name=first.name, created_at=first.created_at)
    for filename in ("aligned.parquet", "coverage.csv"):
        (first.directory / filename).write_bytes((other.directory / filename).read_bytes())
    (first.directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert load_research_set(root, first.set_id).preview.preview_sha256 == changed.preview_sha256
    page.run()
    assert not page.exception
    assert page.session_state.filtered_state.get("factor_result") is None
    assert not page.dataframe
    calculate(page, first.set_id)
    assert page.dataframe[0].value["20 期动量"].eq("—").all()


def test_original_sources_not_needed_and_all_missing_asset_is_visible(saved_factory):
    _, make = saved_factory
    saved, _, source = make(outside=True)
    source.rename(source.with_name("moved-synthetic-source"))
    page = calculate(new_page(), saved.set_id)
    assert page.dataframe[0].value["ETF 代码"].tolist() == ["159915", "510300"]
    assert page.dataframe[0].value.loc[0, "20 期动量"] == "—"
    page.selectbox(key="factor_view_symbol").set_value("159915").run()
    assert not page.get("vega_lite_chart")
    assert any("没有可用" in item.value for item in page.info)


def test_password_gate_precedes_snapshot_access(saved_factory, monkeypatch):
    import quantlab.data.research_storage as storage

    monkeypatch.setenv("QUANTLAB_ACCESS_PASSWORD", "synthetic-password")

    def forbid(*args, **kwargs):
        raise AssertionError("认证前不得读取快照")

    monkeypatch.setattr(storage, "list_research_sets", forbid)
    page = new_page()
    assert page.text_input(key="_quantlab_access_password")
    assert not page.selectbox


def test_public_sessions_use_real_isolated_scopes(monkeypatch):
    monkeypatch.setenv("QUANTLAB_PUBLIC_MODE", "1")
    monkeypatch.delenv("QUANTLAB_ACCESS_PASSWORD", raising=False)
    first = new_page()
    second = new_page()
    temporary_key = "_quantlab_public_temporary_directory"
    first_directory = first.session_state[temporary_key]
    second_directory = second.session_state[temporary_key]
    try:
        assert first_directory.name != second_directory.name
        assert first.session_state["factor_storage_scope"] == str(Path(first_directory.name) / "research_sets")
        assert second.session_state["factor_storage_scope"] == str(Path(second_directory.name) / "research_sets")
        first.session_state["factor_result"] = "synthetic-stale-result"
        second.run()
        assert not second.exception
        assert second.session_state.filtered_state.get("factor_result") is None
        assert not second.dataframe and not second.get("vega_lite_chart")
        # 模拟会话的存储范围发生变化，旧状态必须在显示前清除。
        first.session_state["factor_storage_scope"] = "synthetic-old-scope"
        first.run()
        assert not first.exception
        assert first.session_state.filtered_state.get("factor_result") is None
        assert not first.dataframe and not first.get("vega_lite_chart")
    finally:
        first_directory.cleanup()
        second_directory.cleanup()


def test_navigation_opens_factor_page(saved_factory, monkeypatch, tmp_path):
    import quantlab.ui as ui

    monkeypatch.setattr(ui, "get_market_storage_dirs", lambda project: (tmp_path / "raw", tmp_path / "processed"))
    page = AppTest.from_file("app.py", default_timeout=15).run()
    page.switch_page(PAGE).run()
    assert not page.exception
    assert any(t.value == "因子研究" for t in page.title)
