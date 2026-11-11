"""滚动价格回撤的展示、默认方向与结果隔离，不混为组合绩效。"""

import pytest

from quantlab.factors import create_default_registry, run_research_factor
from quantlab.ui.factors import (
    build_factor_chart, daily_factor_table, factor_explanation, factor_history,
    ranked_factor_table, top_n_factor_preview,
)
from test_ui_factors import calculate, new_page, saved_factory


def require_factor():
    assert "max_drawdown_20" in create_default_registry().names(), "滚动价格最大回撤尚未注册"


def test_price_drawdown_tables_label_zero_and_negative_values(saved_factory):
    require_factor()
    _, make = saved_factory
    _, preview, _ = make(length=50, gap=True)
    result = run_research_factor(preview, "max_drawdown_20")
    day = result.calendar.dates[19]
    assert daily_factor_table(result, day)["20 期价格最大回撤"].tolist() == ["-9.50%", "0.00%"]
    assert ranked_factor_table(result, day)["ETF 代码"].tolist() == ["510300", "159915"]
    assert top_n_factor_preview(result, day, 1).table["ETF 代码"].tolist() == ["510300"]
    assert top_n_factor_preview(result, day, 1, ascending=True).table["ETF 代码"].tolist() == ["159915"]
    history = factor_history(result, "510300")
    assert history.factor_value.iloc[25:45].isna().all()
    assert history.factor_value.iloc[45:].eq(0).all()
    assert history.segment.iloc[24] != history.segment.iloc[45]
    spec = build_factor_chart(history, result.definition).to_dict()
    assert spec["layer"][0]["encoding"]["y"]["title"] == "20 期价格最大回撤"
    explanation = factor_explanation(result.definition)
    assert "回撤" in explanation and "接近 0" in explanation
    assert "涨幅" not in explanation
    assert any("未复权" in w and "回撤" in w for w in result.warnings)
    assert any("组合" in w and "全区间" in w for w in result.warnings)
    assert not any("价格动量" in w for w in result.warnings)


def test_page_switches_semantics_and_preserves_momentum_default(saved_factory):
    require_factor()
    _, make = saved_factory
    saved, _, _ = make(length=45)
    before = {p.name: p.read_bytes() for p in saved.directory.iterdir()}
    app = calculate(new_page(), saved.set_id)
    assert app.selectbox(key="factor_name").value == "momentum_20"
    assert "20 期价格最大回撤" in app.selectbox(key="factor_name").options
    app.selectbox(key="factor_name").set_value("max_drawdown_20").run()
    assert not app.exception
    assert app.session_state.filtered_state.get("factor_result") is None
    assert not app.dataframe and not app.get("vega_lite_chart")
    assert app.selectbox(key="portfolio_direction").value == "descending"
    calculate(app, saved.set_id)
    text = "\n".join(item.value for item in list(app.caption) + list(app.markdown) + list(app.warning))
    assert "窗口内" in text and "20 条" in text and "前 19" in text
    assert "期价格动量 =" not in text and "涨幅更高" not in text
    assert "20 期价格最大回撤" in app.dataframe[0].value.columns
    app.radio(key="factor_daily_order").set_value("因子排名").run()
    assert app.selectbox(key="factor_rank_direction").value == "descending"
    app.selectbox(key="factor_rank_direction").set_value("ascending").run()
    app.selectbox(key="portfolio_direction").set_value("ascending").run()
    app.selectbox(key="factor_name").set_value("bias_20").run()
    assert app.session_state.filtered_state.get("factor_result") is None
    assert "factor_rank_direction" not in app.session_state.filtered_state
    assert app.selectbox(key="portfolio_direction").value == "descending"
    assert {p.name: p.read_bytes() for p in saved.directory.iterdir()} == before


@pytest.mark.parametrize("length", [19, 20])
def test_page_distinguishes_warmup_from_valid_zero(saved_factory, length):
    require_factor()
    _, make = saved_factory
    saved, _, _ = make(length=length)
    app = new_page()
    app.selectbox(key="factor_name").set_value("max_drawdown_20").run()
    calculate(app, saved.set_id)
    assert app.dataframe[0].value["20 期价格最大回撤"].eq("—").all() == (length == 19)
    assert len(app.get("vega_lite_chart")) == (0 if length == 19 else 1)
    assert not app.exception
