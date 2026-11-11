"""均线偏离页面的公式口径、默认方向与旧结果隔离。"""

import pandas as pd
import pytest

from quantlab.factors import create_default_registry, run_research_factor
from quantlab.ui.factors import (
    build_factor_chart, daily_factor_table, factor_explanation, factor_history,
    ranked_factor_table, top_n_factor_preview,
)
from test_ui_factors import calculate, new_page, saved_factory


def require_bias():
    assert "bias_20" in create_default_registry().names(), "均线偏离尚未注册"


def test_bias_tables_chart_and_explanation_do_not_claim_return(saved_factory):
    require_bias()
    _, make = saved_factory
    _, preview, _ = make(length=45, gap=True)
    result = run_research_factor(preview, "bias_20")
    day = result.calendar.dates[19]
    daily = daily_factor_table(result, day)
    assert daily["20 期均线偏离"].tolist() == ["-4.99%", "4.53%"]
    assert ranked_factor_table(result, day)["ETF 代码"].tolist() == ["510300", "159915"]
    assert top_n_factor_preview(result, day, 1).table["ETF 代码"].tolist() == ["510300"]
    assert top_n_factor_preview(result, day, 1, ascending=True).table["ETF 代码"].tolist() == ["159915"]
    warmup = daily_factor_table(result, result.calendar.dates[18])
    assert warmup["状态 / 原因"].str.contains("前 19 ").all()
    assert warmup["20 期均线偏离"].eq("—").all()
    history = factor_history(result, "510300")
    assert history.factor_value.iloc[25:45].isna().all()
    spec = build_factor_chart(history, result.definition).to_dict()
    assert spec["layer"][0]["encoding"]["y"]["title"] == "20 期均线偏离"
    text = factor_explanation(result.definition)
    assert "均线" in text and "涨幅" not in text
    assert any("未复权" in w and "均线偏离" in w for w in result.warnings)
    assert not any("价格动量" in w for w in result.warnings)


def test_switching_bias_keeps_momentum_default_and_clears_old_results(saved_factory):
    require_bias()
    _, make = saved_factory
    saved, _, _ = make(length=45)
    original = {p.name: p.read_bytes() for p in saved.directory.iterdir()}
    app = calculate(new_page(), saved.set_id)
    assert app.selectbox(key="factor_name").value == "momentum_20"
    assert "20 期均线偏离" in app.selectbox(key="factor_name").options
    app.selectbox(key="factor_name").set_value("bias_20").run()
    assert not app.exception
    assert app.session_state.filtered_state.get("factor_result") is None
    assert not app.dataframe and not app.get("vega_lite_chart")
    assert app.button(key="factor_calculate").label == "计算 20 期均线偏离"
    assert app.selectbox(key="portfolio_direction").value == "descending"
    calculate(app, saved.set_id)
    text = "\n".join(i.value for i in list(app.caption) + list(app.markdown) + list(app.warning))
    assert "含当前期" in text and "20 条" in text
    assert "期价格动量 =" not in text and "涨幅更高" not in text
    assert "20 期均线偏离" in app.dataframe[0].value.columns
    app.radio(key="factor_daily_order").set_value("因子排名").run()
    assert app.selectbox(key="factor_rank_direction").value == "descending"
    app.selectbox(key="factor_rank_direction").set_value("ascending").run()
    app.selectbox(key="portfolio_direction").set_value("ascending").run()
    app.selectbox(key="factor_name").set_value("momentum_20").run()
    assert app.session_state.filtered_state.get("factor_result") is None
    assert "factor_rank_direction" not in app.session_state.filtered_state
    assert app.selectbox(key="portfolio_direction").value == "descending"
    assert {p.name: p.read_bytes() for p in saved.directory.iterdir()} == original


@pytest.mark.parametrize("length", [19, 20])
def test_page_twentieth_observation_boundary(saved_factory, length):
    require_bias()
    _, make = saved_factory
    saved, _, _ = make(length=length)
    app = new_page()
    app.selectbox(key="factor_name").set_value("bias_20").run()
    calculate(app, saved.set_id)
    table = app.dataframe[0].value
    assert table["20 期均线偏离"].eq("—").all() == (length == 19)
    assert len(app.get("vega_lite_chart")) == (0 if length == 19 else 1)
    assert not app.exception
