"""单因子相对评分只增补当日排名，不渗入配置、历史或已保存快照。"""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quantlab.factors import create_default_registry, run_research_factor
from quantlab.ui.factors import (
    build_factor_chart, daily_factor_table, factor_history, factor_label,
    history_factor_table, ranked_factor_table, top_n_factor_preview,
)
from test_ui_factors import calculate, new_page, saved_factory


SCORE = "相对评分（0–100）"


@pytest.fixture
def sample_result(saved_factory):
    _, make = saved_factory
    _, preview, _ = make(length=160)
    return run_research_factor(preview)


def with_values(result, values, valid=None, symbols=None):
    """手写原始截面；不借用评分代码生成期望。"""
    when = result.calendar.dates[-1]
    if valid is None:
        valid = [np.isfinite(value) for value in values]
    if symbols is None:
        symbols = [f"{510000 + i:06}" for i in range(len(values))]
    rows = pd.DataFrame({
        "date": pd.to_datetime([when] * len(values)),
        "symbol": symbols,
        "factor_value": np.asarray(values, dtype="float64"),
        "is_valid": valid,
        "status": ["valid" if available else "missing_window" for available in valid],
    }, index=[7] * len(values))
    return replace(result, data=rows)


def scored_table(result, when=None, **kwargs):
    table = ranked_factor_table(result, when or result.calendar.dates[-1], **kwargs)
    assert SCORE in table.columns, "当日因子排名尚未增加相对评分列"
    assert set(table.columns) == {"名次", "ETF 代码", factor_label(result.definition), SCORE, "状态 / 原因"}
    assert len(table.columns) == 5, "相对评分应当只增加一列"
    return table


@pytest.mark.parametrize("ascending,codes,ranks,scores", [
    (False, ["510300", "510500", "159915", "512000", "513000"],
     ["1", "1", "3", "4", "—"], ["83.33", "83.33", "33.33", "0.00", "—"]),
    (True, ["512000", "159915", "510300", "510500", "513000"],
     ["1", "2", "3", "3", "—"], ["100.00", "66.67", "16.67", "16.67", "—"]),
])
def test_average_tie_scores_preserve_competition_ranks(sample_result, ascending, codes, ranks, scores):
    result = with_values(sample_result, [np.nan, .1, 0, -.2, .1],
                         symbols=["513000", "510500", "159915", "512000", "510300"])
    before = deepcopy(result.data)
    table = scored_table(result, ascending=ascending)
    assert table["ETF 代码"].tolist() == codes
    assert table["名次"].tolist() == ranks
    assert table[SCORE].tolist() == scores
    assert table.set_index("ETF 代码").loc["159915", "20 期动量"] == "0.00%"
    table.loc[0, SCORE] = "changed"
    pd.testing.assert_frame_equal(result.data, before)


def test_display_only_ties_use_unrounded_values_and_reverse_scores(sample_result):
    result = with_values(sample_result, [.100001, .100002, .100003],
                         symbols=["510300", "159915", "510500"])
    descending = scored_table(result)
    ascending = scored_table(result, ascending=True)
    assert descending["20 期动量"].tolist() == ["10.00%"] * 3
    assert descending["名次"].tolist() == ["1", "2", "3"]
    assert descending["ETF 代码"].tolist() == ["510500", "159915", "510300"]
    assert descending[SCORE].tolist() == ["100.00", "50.00", "0.00"]
    assert ascending["ETF 代码"].tolist() == ["510300", "159915", "510500"]
    assert ascending[SCORE].tolist() == ["100.00", "50.00", "0.00"]
    assert ascending.set_index("ETF 代码")[SCORE].to_dict() == {
        "510300": "100.00", "159915": "50.00", "510500": "0.00",
    }


@pytest.mark.parametrize("values,valid,want", [
    ([np.nan, np.nan], [False, False], ["—", "—"]),
    ([0, np.nan], [True, False], ["—", "—"]),
    ([-.2, np.nan], [True, False], ["—", "—"]),
    ([0, 0], [True, True], ["50.00", "50.00"]),
    ([-.2, -.2, -.2, np.nan], [True, True, True, False], ["50.00", "50.00", "50.00", "—"]),
])
def test_small_and_all_equal_samples_do_not_invent_discrimination(sample_result, values, valid, want):
    result = with_values(sample_result, values, valid)
    table = scored_table(result)
    assert table[SCORE].tolist() == want
    if sum(valid) == 1:
        assert table.loc[0, "名次"] == "1"
        assert table.loc[0, "状态 / 原因"] == "可计算"
        assert table.loc[0, "20 期动量"] != "—"
    elif sum(valid) >= 2:
        assert table.loc[table["状态 / 原因"].eq("可计算"), "名次"].eq("1").all()


def test_unavailable_rows_do_not_change_valid_sample_denominator(sample_result):
    result = with_values(sample_result, [0, -.2, .4, 999, np.nan, np.inf, -np.inf],
                         [True, True, True, False, False, False, False])
    table = scored_table(result).set_index("ETF 代码")
    assert table[SCORE].to_dict() == {
        "510002": "100.00", "510000": "50.00", "510001": "0.00",
        "510003": "—", "510004": "—", "510005": "—", "510006": "—",
    }


@pytest.mark.parametrize("name,winner", [
    ("momentum_20", "510000"), ("momentum_60", "510000"),
    ("momentum_120", "510000"), ("volatility_20", "510001"),
    ("volatility_60", "510001"), ("bias_20", "510000"),
    ("max_drawdown_20", "510000"), ("volume_change_20", "510000"),
    ("trend_efficiency_20", "510000"),
])
def test_scores_follow_existing_default_direction_for_every_factor(sample_result, name, winner):
    definition = create_default_registry().get(name).definition
    result = with_values(replace(sample_result, definition=definition), [0, -.2])
    table = scored_table(result)
    assert table["ETF 代码"].iloc[0] == winner
    assert table[SCORE].tolist() == ["100.00", "0.00"]


def test_score_scope_is_exact_date_and_current_pool(sample_result):
    result = with_values(sample_result, [3, 2, 1])
    previous = result.data.copy(deep=True)
    previous["date"] = pd.to_datetime(result.calendar.dates[-2])
    previous["factor_value"] = [1000, -1000, 0]
    result = replace(result, data=pd.concat([previous, result.data]))
    table = scored_table(result)
    assert table[SCORE].tolist() == ["100.00", "50.00", "0.00"]
    previous_table = scored_table(result, result.calendar.dates[-2])
    assert previous_table["ETF 代码"].tolist() == ["510000", "510002", "510001"]
    assert previous_table[SCORE].tolist() == ["100.00", "50.00", "0.00"]
    narrowed = replace(result, data=result.data.loc[result.data.symbol.ne("510002")].copy())
    assert scored_table(narrowed)[SCORE].tolist() == ["100.00", "0.00"]


def test_scoring_does_not_change_top_n_ties_weights_daily_or_history(sample_result):
    when = sample_result.calendar.dates[-1]
    result = with_values(sample_result, [.1, .1, 0, -.2],
                         symbols=["510500", "510300", "159915", "512000"])
    before = deepcopy(result.data)
    history = factor_history(result, "510300")
    history_before = history_factor_table(history)
    chart_before = build_factor_chart(history).to_dict()
    scored_table(result)
    preview = top_n_factor_preview(result, when, 1)
    assert preview.valid_count == 4 and preview.cutoff_tie
    assert list(preview.table.columns) == ["名次", "ETF 代码", "20 期动量", "目标权重"]
    assert preview.table.to_dict("records") == [{
        "名次": "1", "ETF 代码": "510300", "20 期动量": "10.00%", "目标权重": 1.0,
    }]
    three = top_n_factor_preview(result, when, 3)
    assert three.table["ETF 代码"].tolist() == ["510300", "510500", "159915"]
    assert three.table["名次"].tolist() == ["1", "1", "3"]
    assert three.table["目标权重"].tolist() == [1 / 3] * 3
    assert not three.cutoff_tie and SCORE not in three.table
    assert SCORE not in daily_factor_table(result, when)
    assert SCORE not in history_before
    pd.testing.assert_frame_equal(history_factor_table(factor_history(result, "510300")), history_before)
    assert build_factor_chart(factor_history(result, "510300")).to_dict() == chart_before
    pd.testing.assert_frame_equal(result.data, before)


def page_rank_table(page):
    assert not page.exception
    table = page.dataframe[0].value
    assert SCORE in table.columns, "因子排名页面尚未展示相对评分"
    return table


def page_text(page):
    return [item.value for item in list(page.caption) + list(page.info) + list(page.markdown)]


def test_page_scores_switch_direction_date_and_leave_snapshot_and_other_views_unchanged(saved_factory):
    _, make = saved_factory
    saved, preview, _ = make(gap=True)
    snapshot_before = {p.name: p.read_bytes() for p in saved.directory.iterdir()}
    page = calculate(new_page(), saved.set_id)
    daily_before = page.dataframe[0].value.copy(deep=True)
    history_before = next(d.value.copy(deep=True) for d in page.dataframe if "日期" in d.value.columns)
    assert SCORE not in daily_before
    page.radio(key="factor_daily_order").set_value("因子排名").run()
    table = page_rank_table(page)
    assert table["ETF 代码"].tolist() == ["510300", "159915"]
    assert table[SCORE].tolist() == ["100.00", "0.00"]
    targets = next(d.value for d in page.dataframe if "目标权重" in d.value.columns)
    assert targets["ETF 代码"].tolist() == ["510300", "159915"]
    assert targets["目标权重"].tolist() == [.5, .5] and SCORE not in targets
    page.selectbox(key="factor_rank_direction").set_value("ascending").run()
    assert page_rank_table(page)["ETF 代码"].tolist() == ["159915", "510300"]
    assert page_rank_table(page)[SCORE].tolist() == ["100.00", "0.00"]
    page.selectbox(key="factor_view_date").set_value(preview.alignment.calendar.dates[25]).run()
    singleton = page_rank_table(page).set_index("ETF 代码")
    assert singleton.loc["159915", "名次"] == "1"
    assert singleton.loc["159915", "20 期动量"] != "—"
    assert singleton.loc["159915", "状态 / 原因"] == "可计算"
    assert singleton.loc["159915", SCORE] == "—"
    assert singleton.loc["510300", SCORE] == "—"
    assert "缺失" in singleton.loc["510300", "状态 / 原因"]
    assert any("评分" in t and ("不足" in t or "少于" in t) for t in page_text(page))
    page.selectbox(key="factor_view_date").set_value(preview.alignment.calendar.dates[0]).run()
    assert page_rank_table(page)[SCORE].eq("—").all()
    assert any("0 / 2" in c.value for c in page.caption)
    assert len(page.get("vega_lite_chart")) == 1
    history_after = next(d.value for d in page.dataframe if "日期" in d.value.columns)
    pd.testing.assert_frame_equal(history_after, history_before)
    page.selectbox(key="factor_view_date").set_value(preview.alignment.calendar.dates[-1]).run()
    page.radio(key="factor_daily_order").set_value("ETF 代码").run()
    pd.testing.assert_frame_equal(page.dataframe[0].value, daily_before)
    assert {p.name: p.read_bytes() for p in saved.directory.iterdir()} == snapshot_before


def test_page_all_tied_scores_are_neutral_and_factor_switch_recomputes_direction(saved_factory):
    _, make = saved_factory
    saved, _, _ = make(length=100, oscillating=True)
    snapshot_before = {p.name: p.read_bytes() for p in saved.directory.iterdir()}
    page = calculate(new_page(), saved.set_id)
    page.radio(key="factor_daily_order").set_value("因子排名").run()
    assert page_rank_table(page)["名次"].tolist() == ["1", "1"]
    assert page_rank_table(page)[SCORE].tolist() == ["50.00", "50.00"]
    assert any("全部相同" in t for t in page_text(page))
    assert any("评分" in t and "50" in t for t in page_text(page))
    page.selectbox(key="factor_rank_direction").set_value("ascending").run()
    assert page_rank_table(page)[SCORE].tolist() == ["50.00", "50.00"]
    page.selectbox(key="factor_name").set_value("volatility_20").run()
    assert not page.exception and not page.dataframe
    assert page.session_state.filtered_state.get("factor_result") is None
    assert "factor_rank_direction" not in page.session_state.filtered_state
    calculate(page, saved.set_id)
    assert SCORE not in page.dataframe[0].value
    page.radio(key="factor_daily_order").set_value("因子排名").run()
    assert page.selectbox(key="factor_rank_direction").value == "ascending"
    table = page_rank_table(page)
    assert table["ETF 代码"].tolist() == ["159915", "510300"]
    assert table[SCORE].tolist() == ["100.00", "0.00"]
    assert "20 期年化波动率" in table and "20 期动量" not in table
    page.selectbox(key="factor_rank_direction").set_value("descending").run()
    assert page_rank_table(page)["ETF 代码"].tolist() == ["510300", "159915"]
    assert page_rank_table(page)[SCORE].tolist() == ["100.00", "0.00"]
    assert {p.name: p.read_bytes() for p in saved.directory.iterdir()} == snapshot_before
