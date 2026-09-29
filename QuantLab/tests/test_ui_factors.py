"""因子页使用真实合成导入和快照，只替换存储目录与故障边界。"""

from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timezone
from itertools import count
from math import sqrt
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

    def make(*, gap=False, length=50, name="合成因子验收", outside=False, oscillating=False):
        source_dir = tmp_path / f"source_{next(sequence)}"
        source_dir.mkdir()
        dates = pd.date_range("2025-01-02", periods=length, freq="B")
        paths = (("510300", 100.0 + np.arange(length) * .5),
                 ("159915", 200.0 - np.arange(length)))
        if oscillating:
            paths = (("510300", np.where(np.arange(length) % 2, 110.0, 100.0)),
                     ("159915", np.full(length, 200.0)))
        for symbol, close in paths:
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


@pytest.mark.parametrize(("periods", "up", "down"), [(60, "30.00%", "-30.00%"), (120, "60.00%", "-60.00%")])
def test_long_period_research_values_labels_and_chart(saved_factory, periods, up, down):
    from quantlab.factors import create_default_registry
    from quantlab.ui.factors import daily_factor_table, factor_history, history_factor_table, build_factor_chart

    _, make = saved_factory
    saved, preview, source = make(length=160)
    name = f"momentum_{periods}"
    assert name in create_default_registry().names()
    before = {p.name: p.read_bytes() for p in saved.directory.iterdir()}
    source.rename(source.with_name("moved-synthetic-source"))
    result = run_research_factor(load_research_set(saved.directory.parent, saved.set_id).preview, name)
    label = f"{periods} 期动量"
    table = daily_factor_table(result, result.calendar.dates[periods])
    assert table[label].tolist() == [down, up]
    history = factor_history(result, "510300")
    detail = history_factor_table(history, result.definition)
    assert detail[label].iloc[:periods].eq("—").all()
    assert detail["状态 / 原因"].iloc[:periods].str.contains(f"前 {periods} ").all()
    chart = build_factor_chart(history, result.definition).to_dict()
    assert chart["layer"][0]["encoding"]["y"]["title"] == label
    assert chart["layer"][0]["encoding"]["tooltip"][1]["title"] == label
    assert len(chart["data"]["values"]) == 160
    assert result.definition.name == name
    assert result.input_preview_sha256 == preview.preview_sha256
    assert any(f"前 {periods} " in w for w in result.warnings)
    assert any(f"{periods} 个自然日" in w for w in result.warnings)
    assert not any("前 20 " in w for w in result.warnings)
    assert {p.name: p.read_bytes() for p in saved.directory.iterdir()} == before


@pytest.mark.parametrize("periods", [20, 60, 120])
def test_history_helpers_infer_period_and_reject_conflicting_or_missing_identity(saved_factory, periods):
    from quantlab.factors import Momentum20Factor, Momentum60Factor
    from quantlab.ui.factors import factor_history, history_factor_table, build_factor_chart

    _, make = saved_factory
    _, preview, _ = make(length=160)
    history = factor_history(run_research_factor(preview, f"momentum_{periods}"), "510300")
    # 沿用原有单参数调用方式，也必须使用这个结果的周期而非静默默认为 20。
    assert f"{periods} 期动量" in history_factor_table(history).columns
    assert build_factor_chart(history).to_dict()["layer"][0]["encoding"]["y"]["title"] == f"{periods} 期动量"
    wrong = Momentum60Factor.definition if periods == 20 else Momentum20Factor.definition
    for display in (history_factor_table, build_factor_chart):
        with pytest.raises(ValueError, match="不一致"):
            display(history, wrong)
        without_identity = history.copy(deep=True)
        without_identity.attrs.clear()
        with pytest.raises(ValueError, match="因子说明"):
            display(without_identity)


def test_page_factor_switch_clears_result_and_updates_all_period_text(saved_factory):
    _, make = saved_factory
    saved, preview, _ = make(length=160)
    page = calculate(new_page(), saved.set_id)
    assert "factor_name" in page.session_state.filtered_state, "页面尚未支持因子选择"
    assert page.selectbox(key="factor_name").value == "momentum_20"
    for periods, expected in [(60, ["-30.00%", "30.00%"]), (120, ["-60.00%", "60.00%"])]:
        page.selectbox(key="factor_name").set_value(f"momentum_{periods}").run()
        assert not page.exception
        assert page.session_state.filtered_state.get("factor_result") is None
        assert not page.dataframe and not page.get("vega_lite_chart")
        assert str(periods) in page.button(key="factor_calculate").label
        calculate(page, saved.set_id)
        page.selectbox(key="factor_view_date").set_value(preview.alignment.calendar.dates[periods]).run()
        assert page.dataframe[0].value[f"{periods} 期动量"].tolist() == expected
        assert any(f"close[t] / close[t-{periods}] - 1" in c.value for c in page.code)
        assert any(f"前 {periods} " in w.value for w in page.warning)
        assert not any("前 20 " in w.value for w in page.warning)
    page.selectbox(key="factor_name").set_value("momentum_20").run()
    assert page.session_state.filtered_state.get("factor_result") is None
    calculate(page, saved.set_id)
    page.selectbox(key="factor_view_date").set_value(preview.alignment.calendar.dates[20]).run()
    assert page.dataframe[0].value["20 期动量"].tolist() == ["-10.00%", "10.00%"]


@pytest.mark.parametrize("periods", [60, 120])
def test_long_period_page_with_short_history_is_empty_not_previous_factor(saved_factory, periods):
    _, make = saved_factory
    saved, _, _ = make(length=50)
    page = calculate(new_page(), saved.set_id)
    assert "factor_name" in page.session_state.filtered_state
    page.selectbox(key="factor_name").set_value(f"momentum_{periods}").run()
    calculate(page, saved.set_id)
    assert page.dataframe[0].value[f"{periods} 期动量"].eq("—").all()
    assert page.dataframe[0].value["状态 / 原因"].str.contains(f"前 {periods} ").all()
    assert not page.get("vega_lite_chart")
    assert any("没有可用" in info.value for info in page.info)


@pytest.mark.parametrize("periods", [20, 60])
def test_volatility_snapshot_calculation_and_presentation_are_read_only(saved_factory, periods):
    from quantlab.factors import create_default_registry, Momentum20Factor
    from quantlab.ui.factors import daily_factor_table, factor_history, history_factor_table, build_factor_chart

    _, make = saved_factory
    saved, preview, source = make(length=100, gap=True, oscillating=True)
    name = f"volatility_{periods}"
    assert name in create_default_registry().names()
    before = {p.name: p.read_bytes() for p in saved.directory.iterdir()}
    source.rename(source.with_name("removed-volatility-synthetic-source"))
    result = run_research_factor(load_research_set(saved.directory.parent, saved.set_id).preview, name)
    assert result.input_preview_sha256 == preview.preview_sha256
    label = f"{periods} 期年化波动率"
    # 100/110 交替价格产生等量的 +10% 和 -1/11 简单收益率。
    expected = (21 / 220) * sqrt(252 * periods / (periods - 1))
    table = daily_factor_table(result, result.calendar.dates[-1])
    assert table[label].tolist() == ["0.00%", f"{expected:.2%}"]
    history = factor_history(result, "510300")
    detail = history_factor_table(history)
    assert detail[label].iloc[:periods].eq("—").all()
    assert detail[label].iloc[max(25, periods):26 + periods].eq("—").all()
    assert "缺失" in detail["状态 / 原因"].iloc[max(25, periods)]
    assert history.factor_value.iloc[26 + periods] == pytest.approx(expected)
    chart = build_factor_chart(history).to_dict()
    assert chart["layer"][0]["encoding"]["y"]["title"] == label
    assert chart["layer"][0]["encoding"]["tooltip"][1]["title"] == label
    assert len(chart["data"]["values"]) == 100
    assert chart["data"]["values"][max(25, periods)]["factor_value"] is None
    if periods == 20:
        assert history.segment.iloc[24] != history.segment.iloc[46]
    for display in (build_factor_chart, history_factor_table):
        with pytest.raises(ValueError, match="不一致"):
            display(history, Momentum20Factor.definition)
    assert any("252" in w and "假设" in w for w in result.warnings)
    assert any("低波动" in w and "安全" in w for w in result.warnings)
    assert any("未复权" in w and "波动" in w for w in result.warnings)
    assert {p.name: p.read_bytes() for p in saved.directory.iterdir()} == before


def test_page_switches_between_momentum_and_volatility_without_stale_semantics(saved_factory):
    _, make = saved_factory
    saved, _, _ = make(length=100, oscillating=True)
    page = calculate(new_page(), saved.set_id)
    assert page.selectbox(key="factor_name").value == "momentum_20"
    assert "20 期年化波动率" in page.selectbox(key="factor_name").options
    for periods in (20, 60):
        page.selectbox(key="factor_name").set_value(f"volatility_{periods}").run()
        assert not page.exception
        assert page.session_state.filtered_state.get("factor_result") is None
        assert not page.dataframe and not page.get("vega_lite_chart")
        assert "波动率" in page.button(key="factor_calculate").label
        calculate(page, saved.set_id)
        label = f"{periods} 期年化波动率"
        assert label in page.dataframe[0].value.columns
        assert page.dataframe[0].value[label].iloc[0] == "0.00%"
        assert len(page.get("vega_lite_chart")) == 1  # 真正的 0% 仍要有曲线。
        assert any(label in s.value for s in page.subheader)
        text = "\n".join(i.value for i in list(page.caption) + list(page.markdown) + list(page.warning))
        assert "涨幅更高" not in text and "期价格动量 =" not in text
        assert "252" in text and "假设" in text and "样本标准差" in text
        assert any("ddof=1" in c.value and "sqrt(252)" in c.value for c in page.code)
        assert any("波动" in w.value and "低波动" in w.value for w in page.warning)
    page.selectbox(key="factor_name").set_value("momentum_20").run()
    assert page.session_state.filtered_state.get("factor_result") is None
    calculate(page, saved.set_id)
    assert "20 期动量" in page.dataframe[0].value.columns
    assert any("涨幅更高" in m.value for m in page.markdown)
    assert not any("低波动" in w.value for w in page.warning)


@pytest.mark.parametrize("periods", [20, 60])
def test_volatility_page_short_history_has_no_zero_or_chart(saved_factory, periods):
    _, make = saved_factory
    saved, _, _ = make(length=10)
    page = new_page()
    assert f"{periods} 期年化波动率" in page.selectbox(key="factor_name").options
    page.selectbox(key="factor_name").set_value(f"volatility_{periods}").run()
    calculate(page, saved.set_id)
    assert page.dataframe[0].value[f"{periods} 期年化波动率"].eq("—").all()
    assert not page.get("vega_lite_chart")
    assert any("没有可用" in info.value for info in page.info)


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


@pytest.mark.parametrize("ascending, codes, ranks", [
    (False, ["510300", "510500", "159915", "512000", "513000"], ["1", "1", "3", "4", "—"]),
    (True, ["512000", "159915", "510300", "510500", "513000"], ["1", "2", "3", "3", "—"]),
])
def test_ranking_uses_numeric_values_competition_ties_and_keeps_missing(saved_factory, ascending, codes, ranks):
    from quantlab.ui import factors as display

    assert hasattr(display, "ranked_factor_table"), "尚未实现当日排名"
    _, make = saved_factory
    _, preview, _ = make()
    result = run_research_factor(preview)
    when = result.calendar.dates[-1]
    # 手工给定因子结果，独立验证展示排名，不借用排名算法计算期望。
    rows = pd.DataFrame({
        "date": pd.to_datetime([when] * 5),
        "symbol": ["513000", "510500", "159915", "512000", "510300"],
        "factor_value": [np.nan, .1, 0.0, -.2, .1],
        "status": ["missing_window", "valid", "valid", "valid", "valid"],
        "is_valid": [False, True, True, True, True],
    }, index=[7, 7, 9, 12, 15])
    result = replace(result, data=rows)
    before = deepcopy(result.data)
    table = display.ranked_factor_table(result, when, ascending=ascending)
    assert table["ETF 代码"].tolist() == codes
    assert table["名次"].tolist() == ranks
    indexed = table.set_index("ETF 代码")
    assert indexed.loc["159915", "20 期动量"] == "0.00%"
    assert indexed.loc["513000", "20 期动量"] == "—"
    assert "缺失" in indexed.loc["513000", "状态 / 原因"]
    table.loc[0, "名次"] = "changed"
    pd.testing.assert_frame_equal(result.data, before)


def test_ranking_does_not_round_values_into_false_ties(saved_factory):
    from quantlab.ui import factors as display

    assert hasattr(display, "ranked_factor_table"), "尚未实现当日排名"
    _, make = saved_factory
    _, preview, _ = make()
    result = run_research_factor(preview)
    when = result.calendar.dates[-1]
    rows = result.data.loc[result.data.date.dt.date.eq(when)].copy()
    rows["factor_value"] = [.100001, .100002]
    result = replace(result, data=rows)
    table = display.ranked_factor_table(result, when)
    assert table["ETF 代码"].tolist() == ["510300", "159915"]
    assert table["名次"].tolist() == ["1", "2"]
    assert table["20 期动量"].tolist() == ["10.00%", "10.00%"]


@pytest.mark.parametrize("name, codes", [
    ("momentum_20", ["510300", "159915"]),
    ("momentum_60", ["510300", "159915"]),
    ("momentum_120", ["510300", "159915"]),
    ("volatility_20", ["159915", "510300"]),
    ("volatility_60", ["159915", "510300"]),
])
def test_ranking_default_direction_is_factor_specific_and_date_exact(saved_factory, name, codes):
    from quantlab.ui import factors as display

    assert hasattr(display, "ranked_factor_table"), "尚未实现当日排名"
    _, make = saved_factory
    _, preview, _ = make(length=160, gap=True, oscillating=True)
    result = run_research_factor(preview, name)
    when = result.calendar.dates[-1]
    if name.startswith("momentum"):
        # 本例完整后缀中动量相等；另给定两个有效值单独检验默认方向。
        result.data.loc[result.data.date.dt.date.eq(when), "factor_value"] = [-.1, .1]
    assert display.ranked_factor_table(result, when)["ETF 代码"].tolist() == codes
    warmup = display.ranked_factor_table(result, result.calendar.dates[0])
    assert warmup["名次"].eq("—").all() and len(warmup) == 2
    gap_date = result.calendar.dates[max(25, result.definition.lookback_periods)]
    gap = display.ranked_factor_table(result, gap_date).set_index("ETF 代码")
    assert gap.loc["510300", "名次"] == "—"
    assert gap.loc["159915", "名次"] == "1"
    with pytest.raises(ValueError, match="日期"):
        display.ranked_factor_table(result, date(2020, 1, 1))


def test_page_ranking_switch_direction_date_and_read_only_snapshot(saved_factory):
    _, make = saved_factory
    saved, preview, _ = make(gap=True)
    before = {p.name: p.read_bytes() for p in saved.directory.iterdir()}
    page = calculate(new_page(), saved.set_id)
    assert "factor_daily_order" in page.session_state.filtered_state, "页面尚未支持排名"
    assert "名次" not in page.dataframe[0].value
    page.radio(key="factor_daily_order").set_value("因子排名").run()
    assert not page.exception
    assert page.selectbox(key="factor_rank_direction").value == "descending"
    assert page.dataframe[0].value["ETF 代码"].tolist() == ["510300", "159915"]
    assert page.dataframe[0].value["名次"].tolist() == ["1", "2"]
    assert any("2 / 2" in c.value for c in page.caption)
    page.selectbox(key="factor_rank_direction").set_value("ascending").run()
    assert page.dataframe[0].value["ETF 代码"].tolist() == ["159915", "510300"]
    dates = preview.alignment.calendar.dates
    page.selectbox(key="factor_view_date").set_value(dates[25]).run()
    assert page.dataframe[0].value["名次"].tolist() == ["1", "—"]
    assert any("1 / 2" in c.value for c in page.caption)
    assert any("不足两只" in i.value for i in page.info)
    page.selectbox(key="factor_view_date").set_value(dates[0]).run()
    assert page.dataframe[0].value["名次"].eq("—").all()
    assert any("0 / 2" in c.value for c in page.caption)
    assert len(page.get("vega_lite_chart")) == 1  # 日期选择不截断历史。
    page.radio(key="factor_daily_order").set_value("ETF 代码").run()
    assert "名次" not in page.dataframe[0].value
    assert page.dataframe[0].value["ETF 代码"].tolist() == ["159915", "510300"]
    assert {p.name: p.read_bytes() for p in saved.directory.iterdir()} == before


def test_page_rank_state_resets_across_factors_and_sets_and_explains_equal_values(saved_factory):
    _, make = saved_factory
    saved, _, _ = make(length=100, oscillating=True)
    other, _, _ = make(length=10)
    page = calculate(new_page(), saved.set_id)
    assert "factor_daily_order" in page.session_state.filtered_state, "页面尚未支持排名"
    page.radio(key="factor_daily_order").set_value("因子排名").run()
    assert page.dataframe[0].value["名次"].tolist() == ["1", "1"]
    assert any("全部相同" in i.value for i in page.info)
    page.selectbox(key="factor_name").set_value("volatility_20").run()
    assert not page.dataframe and not page.get("vega_lite_chart")
    assert "factor_rank_direction" not in page.session_state.filtered_state
    calculate(page, saved.set_id)
    assert page.radio(key="factor_daily_order").value == "ETF 代码"
    page.radio(key="factor_daily_order").set_value("因子排名").run()
    assert page.selectbox(key="factor_rank_direction").value == "ascending"
    assert page.dataframe[0].value["20 期年化波动率"].iloc[0] == "0.00%"
    assert page.dataframe[0].value["名次"].tolist() == ["1", "2"]
    page.selectbox(key="factor_rank_direction").set_value("descending").run()
    assert page.dataframe[0].value["ETF 代码"].tolist() == ["510300", "159915"]
    page.selectbox(key="factor_set_id").set_value(other.set_id).run()
    assert not page.dataframe
    assert "factor_rank_direction" not in page.session_state.filtered_state
