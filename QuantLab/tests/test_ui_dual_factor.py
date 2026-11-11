"""固定双因子预览使用真实快照，独立于单因子观察、Top N 和档案。"""

from datetime import datetime, timezone
import json

import pandas as pd
import pytest

from quantlab.data.catalog import scan_market_catalog
from quantlab.data.importer import import_market_csv
from quantlab.data.research_sets import DatasetSelection, build_research_set
from quantlab.data.research_storage import load_research_set, save_research_set
from quantlab.ui import runtime
from test_ui_factors import calculate, new_page, saved_factory


COLUMNS = [
    "综合名次", "ETF 代码", "20 期动量", "动量状态", "20 期年化波动率", "波动率状态",
    "动量评分", "低波动评分", "综合评分", "共同样本 / 原因",
]
SCORES = ["动量评分", "低波动评分", "综合评分"]


@pytest.fixture(autouse=True)
def isolated_portfolio_storage(tmp_path, monkeypatch):
    # 页面同时渲染组合与历史区域，必须也只访问本次测试的临时目录。
    monkeypatch.setattr(runtime, "get_portfolio_storage_dir", lambda project: tmp_path / "archives")


def require_pair_controls(page):
    assert any(button.key == "factor_pair_calculate" for button in page.button), "尚未实现独立双因子预览计算按钮"
    assert any(box.key == "factor_pair_date" for box in page.selectbox), "尚未实现双因子独立日期选择"


def pair_tables(page):
    return [item.value for item in page.dataframe if "综合评分" in item.value.columns]


def pair_table(page):
    assert not page.exception
    tables = pair_tables(page)
    assert len(tables) == 1, "双因子计算后应显示且仅显示一张当日预览表"
    assert list(tables[0].columns) == COLUMNS
    return tables[0]


def pair_calculate(page, when=None):
    require_pair_controls(page)
    if when is not None:
        page.selectbox(key="factor_pair_date").set_value(when).run()
    page.button(key="factor_pair_calculate").click().run()
    assert not page.exception and not page.error
    return pair_table(page)


def notice_text(page):
    return "\n".join(item.value for item in (
        list(page.caption) + list(page.info) + list(page.warning) + list(page.markdown) + list(page.error)
    ))


def snapshot_bytes(saved):
    return {path.name: path.read_bytes() for path in saved.directory.iterdir()}


def test_pair_requires_explicit_click_and_defaults_to_exact_last_axis_date(saved_factory):
    _, make = saved_factory
    saved, preview, _ = make()
    before = snapshot_bytes(saved)
    page = new_page()
    require_pair_controls(page)
    assert [tab.label for tab in page.tabs if tab.label in ("因子观察", "组合回测", "双因子预览")] == [
        "因子观察", "组合回测", "双因子预览",
    ]
    axis = preview.alignment.calendar.dates
    assert page.selectbox(key="factor_pair_date").value == axis[-1]
    assert page.selectbox(key="factor_pair_date").options == [day.isoformat() for day in axis]
    assert not pair_tables(page)
    assert page.session_state.filtered_state.get("factor_result") is None
    table = pair_calculate(page)
    assert table["ETF 代码"].tolist() == ["510300", "159915"]
    assert table["综合名次"].tolist() == ["1", "2"]
    assert table["动量评分"].tolist() == ["100.00", "0.00"]
    assert table["低波动评分"].tolist() == ["100.00", "0.00"]
    assert table["综合评分"].tolist() == ["100.00", "0.00"]
    assert table["20 期动量"].tolist() == ["8.73%", "-11.70%"]
    assert table["动量状态"].tolist() == ["可计算", "可计算"]
    assert table["波动率状态"].tolist() == ["可计算", "可计算"]
    assert snapshot_bytes(saved) == before


def test_pair_uses_snapshot_after_original_sources_move_and_never_writes_archives(saved_factory, tmp_path):
    root, make = saved_factory
    saved, _, source = make()
    before = snapshot_bytes(saved)
    source.rename(source.with_name("moved-dual-synthetic-source"))
    page = new_page()
    require_pair_controls(page)
    pair_calculate(page)
    pair_calculate(page)
    assert snapshot_bytes(saved) == before
    assert [path.name for path in root.iterdir()] == [saved.set_id]
    assert not (tmp_path / "archives").exists()


def test_pair_all_equal_momentum_still_uses_distinct_low_volatility_scores(saved_factory):
    _, make = saved_factory
    saved, _, _ = make(oscillating=True)
    page = new_page()
    require_pair_controls(page)
    table = pair_calculate(page)
    assert table["ETF 代码"].tolist() == ["159915", "510300"]
    assert table["20 期动量"].tolist() == ["0.00%", "0.00%"]
    assert table["20 期年化波动率"].tolist() == ["0.00%", "155.47%"]
    assert table["动量评分"].tolist() == ["50.00", "50.00"]
    assert table["低波动评分"].tolist() == ["100.00", "0.00"]
    assert table["综合评分"].tolist() == ["75.00", "25.00"]
    assert table["综合名次"].tolist() == ["1", "2"]
    text = notice_text(page)
    assert "50" in text and ("相同" in text or "并列" in text)


def test_pair_exact_combined_ties_have_competition_ranks_and_code_tiebreak(saved_factory):
    root, make = saved_factory
    _, original, source = make()
    # 复制上涨行情为第三只真实导入 ETF；两只上涨 ETF 的两个原值均完全相同。
    frame = pd.read_csv(source / "510300.csv", dtype={"symbol": str})
    frame["symbol"] = "510500"
    csv = source / "510500.csv"
    frame.to_csv(csv, index=False)
    import_market_csv(csv, source / "raw", source / "processed",
                      imported_at=datetime(2025, 6, 1, tzinfo=timezone.utc))
    scan = scan_market_catalog(source / "processed", source / "raw")
    preview = build_research_set(tuple(DatasetSelection(record, "none") for record in scan.records),
                                 start=original.start, end=original.end)
    saved = save_research_set(preview, root, name="真实三 ETF 并列预览")
    page = new_page()
    require_pair_controls(page)
    page.selectbox(key="factor_set_id").set_value(saved.set_id).run()
    table = pair_calculate(page)
    assert table["ETF 代码"].tolist() == ["510300", "510500", "159915"]
    assert table["综合名次"].tolist() == ["1", "1", "3"]
    assert table["动量评分"].tolist() == ["75.00", "75.00", "0.00"]
    assert table["低波动评分"].tolist() == ["75.00", "75.00", "0.00"]
    assert table["综合评分"].tolist() == ["75.00", "75.00", "0.00"]


@pytest.mark.parametrize("day_index, common", [(0, 0), (25, 1)])
def test_pair_insufficient_common_sample_has_no_scores_but_retains_raw_values(saved_factory, day_index, common):
    _, make = saved_factory
    _, preview, _ = make(gap=True, oscillating=True)
    page = new_page()
    require_pair_controls(page)
    table = pair_calculate(page, preview.alignment.calendar.dates[day_index])
    assert len(table) == 2
    assert table[SCORES].eq("—").all().all()
    assert table["综合名次"].eq("—").all()
    indexed = table.set_index("ETF 代码")
    if common == 0:
        assert table["20 期动量"].eq("—").all()
        assert table["20 期年化波动率"].eq("—").all()
        assert table["动量状态"].str.contains("历史不足").all()
        assert table["波动率状态"].str.contains("历史不足").all()
    else:
        assert indexed.loc["159915", "20 期动量"] == "0.00%"
        assert indexed.loc["159915", "20 期年化波动率"] == "0.00%"
        assert indexed.loc["159915", "动量状态"] == "可计算"
        assert indexed.loc["159915", "波动率状态"] == "可计算"
        assert indexed.loc["510300", "20 期动量"] == "—"
        assert indexed.loc["510300", "20 期年化波动率"] == "—"
        assert "缺失" in indexed.loc["510300", "动量状态"]
        assert "缺失" in indexed.loc["510300", "波动率状态"]
        assert "缺失" in indexed.loc["510300", "共同样本 / 原因"]
    text = notice_text(page)
    assert f"{common} / 2" in text
    assert "评分" in text and ("不足" in text or "少于" in text)


def test_pair_date_change_clears_only_pair_until_recalculated(saved_factory):
    _, make = saved_factory
    saved, preview, _ = make(gap=True)
    before = snapshot_bytes(saved)
    page = calculate(new_page(), saved.set_id)
    require_pair_controls(page)
    page.radio(key="factor_daily_order").set_value("因子排名").run()
    page.number_input(key="factor_selection_n").set_value(1).run()
    single_before = page.dataframe[0].value.copy(deep=True)
    top_before = next(item.value.copy(deep=True) for item in page.dataframe if "目标权重" in item.value)
    pair_calculate(page)
    pd.testing.assert_frame_equal(page.dataframe[0].value, single_before)
    pd.testing.assert_frame_equal(next(item.value for item in page.dataframe if "目标权重" in item.value), top_before)
    old_single_date = page.selectbox(key="factor_view_date").value
    page.selectbox(key="factor_pair_date").set_value(preview.alignment.calendar.dates[25]).run()
    assert not pair_tables(page)
    assert page.selectbox(key="factor_view_date").value == old_single_date
    pd.testing.assert_frame_equal(page.dataframe[0].value, single_before)
    pd.testing.assert_frame_equal(next(item.value for item in page.dataframe if "目标权重" in item.value), top_before)
    text = notice_text(page)
    assert "日期" in text and ("重新计算" in text or "清除" in text)
    current = pair_calculate(page)
    assert current[SCORES].eq("—").all().all()
    assert snapshot_bytes(saved) == before


def test_pair_fixed_definition_does_not_inherit_single_factor_or_reverse_controls(saved_factory):
    _, make = saved_factory
    saved, _, _ = make(oscillating=True)
    page = new_page()
    require_pair_controls(page)
    page.selectbox(key="factor_name").set_value("volatility_20").run()
    calculate(page, saved.set_id)
    page.radio(key="factor_daily_order").set_value("因子排名").run()
    page.selectbox(key="factor_rank_direction").set_value("descending").run()
    assert page.dataframe[0].value["ETF 代码"].tolist() == ["510300", "159915"]
    expected = pair_calculate(page).copy(deep=True)
    assert expected["ETF 代码"].tolist() == ["159915", "510300"]
    assert expected["综合评分"].tolist() == ["75.00", "25.00"]
    page.selectbox(key="factor_name").set_value("momentum_60").run()
    assert not page.exception
    assert page.session_state.filtered_state.get("factor_result") is None
    pd.testing.assert_frame_equal(pair_table(page), expected)


def test_pair_set_change_discards_cached_result_and_resets_date(saved_factory):
    _, make = saved_factory
    first, _, _ = make()
    second, short, _ = make(length=10, name="双因子短历史")
    page = new_page()
    require_pair_controls(page)
    page.selectbox(key="factor_set_id").set_value(first.set_id).run()
    pair_calculate(page)
    page.selectbox(key="factor_set_id").set_value(second.set_id).run()
    assert not page.exception and not pair_tables(page)
    assert page.selectbox(key="factor_pair_date").value == short.alignment.calendar.dates[-1]
    table = pair_calculate(page)
    assert table[SCORES].eq("—").all().all()
    assert table["动量状态"].str.contains("历史不足").all()


def test_pair_same_id_and_dates_with_new_valid_fingerprint_clears_old_scores(saved_factory):
    root, make = saved_factory
    first, original, _ = make()
    other, changed, _ = make(oscillating=True)
    assert original.alignment.calendar.dates == changed.alignment.calendar.dates
    assert original.preview_sha256 != changed.preview_sha256
    page = new_page()
    page.selectbox(key="factor_set_id").set_value(first.set_id).run()
    assert pair_calculate(page)["综合评分"].tolist() == ["100.00", "0.00"]
    old_date = page.selectbox(key="factor_pair_date").value
    # 只替换临时合成快照；保持 ID 和日期不变，确保由指纹变化触发失效。
    manifest = json.loads((other.directory / "manifest.json").read_text())
    manifest.update(set_id=first.set_id, name=first.name, created_at=first.created_at)
    for filename in ("aligned.parquet", "coverage.csv"):
        (first.directory / filename).write_bytes((other.directory / filename).read_bytes())
    (first.directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert load_research_set(root, first.set_id).preview.preview_sha256 == changed.preview_sha256
    before = snapshot_bytes(first)
    page.run()
    assert not page.exception and not pair_tables(page)
    assert page.selectbox(key="factor_set_id").value == first.set_id
    assert page.selectbox(key="factor_pair_date").value == old_date
    assert page.session_state.filtered_state.get("factor_pair_result") is None
    assert page.session_state.filtered_state.get("factor_pair_identity") is None
    assert pair_calculate(page)["综合评分"].tolist() == ["75.00", "25.00"]
    assert page.session_state["factor_pair_result"].input_preview_sha256 == changed.preview_sha256
    assert snapshot_bytes(first) == before


@pytest.mark.parametrize("damage", ["corrupt", "removed", "scope"])
def test_pair_unavailable_snapshot_or_changed_scope_cannot_show_old_success(saved_factory, monkeypatch, tmp_path, damage):
    _, make = saved_factory
    saved, _, _ = make()
    page = new_page()
    require_pair_controls(page)
    pair_calculate(page)
    if damage == "corrupt":
        (saved.directory / "coverage.csv").write_text("broken", encoding="utf-8")
    elif damage == "removed":
        saved.directory.rename(saved.directory.with_name("moved-outside-synthetic-id"))
    else:
        monkeypatch.setattr(runtime, "get_research_storage_dir", lambda project: tmp_path / "new-pair-scope")
    page.run()
    assert not page.exception and not pair_tables(page)
    assert page.session_state.filtered_state.get("factor_pair_result") is None


def test_pair_failed_recalculation_hides_old_success_without_leaking_exception(saved_factory, monkeypatch):
    import quantlab.factors as factors

    _, make = saved_factory
    _, _, _ = make()
    page = new_page()
    require_pair_controls(page)
    pair_calculate(page)

    def fail(*args, **kwargs):
        raise ValueError("synthetic-private-path-do-not-display")

    # 仅替换已经真实成功过的故障边界，不替换任何正常因子或评分计算。
    monkeypatch.setattr(factors, "run_dual_factor_preview", fail)
    page.button(key="factor_pair_calculate").click().run()
    assert not page.exception and page.error
    assert not pair_tables(page)
    assert page.session_state.filtered_state.get("factor_pair_result") is None
    assert "synthetic-private-path-do-not-display" not in notice_text(page)
