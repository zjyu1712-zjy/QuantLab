"""趋势效率显示数字而非收益百分比，保留旧因子的展示契约。"""

import pytest

from quantlab.factors import create_default_registry, run_research_factor
from quantlab.ui import runtime
from quantlab.ui.factors import (
    build_factor_chart, daily_factor_table, factor_history, history_factor_table,
    ranked_factor_table, top_n_factor_preview,
)
from test_trend_factor_portfolio import trend_preview_factory, trend_factor
from test_ui_factors import calculate, new_page


@pytest.fixture
def trend_page_factory(trend_preview_factory, monkeypatch):
    monkeypatch.delenv("QUANTLAB_PUBLIC_MODE", raising=False)
    monkeypatch.delenv("QUANTLAB_ACCESS_PASSWORD", raising=False)

    def make(**kwargs):
        preview, saved = trend_preview_factory(**kwargs)
        monkeypatch.setattr(runtime, "get_research_storage_dir", lambda project: saved.directory.parent)
        return preview, saved

    return make


def test_numeric_tables_chart_and_flat_zero_are_distinct_from_missing(trend_preview_factory):
    preview, _ = trend_preview_factory(downside=True, gap=True)
    result = trend_factor(preview)
    day = result.calendar.dates[20]
    table = daily_factor_table(result, day).set_index("ETF 代码")
    assert table.loc["159915", "20 期趋势效率"] == "—"
    assert table.loc["510300", "20 期趋势效率"] == "0.00"
    assert ranked_factor_table(result, day)["20 期趋势效率"].tolist() == ["0.00", "—"]
    assert top_n_factor_preview(result, day, 1).table["20 期趋势效率"].tolist() == ["0.00"]
    history = factor_history(result, "510300")
    assert history_factor_table(history)["20 期趋势效率"].iloc[20] == "0.00"
    spec = build_factor_chart(history).to_dict()["layer"][0]["encoding"]
    assert spec["y"]["axis"]["format"] == ".2f"
    assert spec["tooltip"][1]["format"] == ".2f"


def test_all_eight_legacy_factor_tables_and_charts_keep_percentage(trend_preview_factory):
    preview, _ = trend_preview_factory(periods=125)
    names = create_default_registry().names()
    assert "trend_efficiency_20" in names
    for name in names:
        if name == "trend_efficiency_20":
            continue
        result = run_research_factor(preview, name)
        history = factor_history(result, "510300")
        label = result.definition.description.replace("价格动量", "动量")
        table = daily_factor_table(result, result.calendar.dates[-1])
        assert table[label].str.endswith("%").all()
        spec = build_factor_chart(history).to_dict()["layer"][0]["encoding"]
        assert spec["y"]["axis"]["format"] == ".1%"
        assert spec["tooltip"][1]["format"] == ".2%"


def test_page_clears_price_result_and_resets_trend_directions(trend_page_factory):
    preview, saved = trend_page_factory()
    trend_factor(preview)
    before = {p.name: p.read_bytes() for p in saved.directory.iterdir()}
    app = calculate(new_page(), saved.set_id)
    assert app.selectbox(key="factor_name").value == "momentum_20"
    assert "20 期趋势效率" in app.selectbox(key="factor_name").options
    app.radio(key="factor_daily_order").set_value("因子排名").run()
    app.selectbox(key="factor_rank_direction").set_value("ascending").run()
    app.selectbox(key="portfolio_direction").set_value("ascending").run()
    app.selectbox(key="factor_name").set_value("trend_efficiency_20").run()
    assert not app.exception
    assert app.session_state.filtered_state.get("factor_result") is None
    assert not app.dataframe and not app.get("vega_lite_chart")
    assert "factor_rank_direction" not in app.session_state.filtered_state
    assert app.selectbox(key="portfolio_direction").value == "descending"
    calculate(app, saved.set_id)
    text = "\n".join(item.value for item in list(app.caption) + list(app.markdown) + list(app.warning))
    assert "前 20" in text and "21" in text
    assert "不是 0%" not in text and "涨幅更高" not in text
    assert "收益" in text and "趋势" in text
    assert "20 期趋势效率" in app.dataframe[0].value.columns
    assert not app.dataframe[0].value["20 期趋势效率"].str.contains("%").any()
    app.radio(key="factor_daily_order").set_value("因子排名").run()
    assert app.selectbox(key="factor_rank_direction").value == "descending"
    app.selectbox(key="factor_name").set_value("volume_change_20").run()
    assert app.session_state.filtered_state.get("factor_result") is None
    assert not app.dataframe
    assert {p.name: p.read_bytes() for p in saved.directory.iterdir()} == before


@pytest.mark.parametrize("periods", [20, 21])
def test_page_flat_zero_is_valid_after_warmup_not_unavailable(trend_page_factory, periods):
    preview, saved = trend_page_factory(periods=periods, downside=True)
    trend_factor(preview)
    app = new_page()
    app.selectbox(key="factor_name").set_value("trend_efficiency_20").run()
    calculate(app, saved.set_id)
    assert not app.exception
    table = app.dataframe[0].value.set_index("ETF 代码")
    assert table.loc["510300", "20 期趋势效率"] == ("—" if periods == 20 else "0.00")
    assert table.loc["159915", "20 期趋势效率"] == ("—" if periods == 20 else "-1.00")
