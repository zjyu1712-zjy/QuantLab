"""成交量研究展示及切换隔离，使用真实保存的独立成交量样本。"""

import pytest

from quantlab.ui import runtime
from quantlab.ui.factors import daily_factor_table, factor_explanation
from test_ui_factors import calculate, new_page
from test_volume_factor_portfolio import volume_preview_factory, volume_factor


@pytest.fixture
def volume_page_factory(volume_preview_factory, monkeypatch):
    monkeypatch.delenv("QUANTLAB_PUBLIC_MODE", raising=False)
    monkeypatch.delenv("QUANTLAB_ACCESS_PASSWORD", raising=False)

    def make(**kwargs):
        preview, saved = volume_preview_factory(**kwargs)
        monkeypatch.setattr(runtime, "get_research_storage_dir", lambda project: saved.directory.parent)
        return preview, saved

    return make


def test_tables_keep_minus_one_valid_and_zero_denominator_unavailable(volume_preview_factory):
    preview, _ = volume_preview_factory(zero_baseline=True, current_zero=True)
    result = volume_factor(preview)
    table = daily_factor_table(result, result.calendar.dates[20]).set_index("ETF 代码")
    assert table.loc["159915", "20 期成交量变化"] == "—"
    assert table.loc["510300", "20 期成交量变化"] == "-100.00%"
    explanation = factor_explanation(result.definition)
    assert "成交量" in explanation and "收益" in explanation


def test_page_switch_from_price_clears_stale_result_and_resets_directions(volume_page_factory):
    preview, saved = volume_page_factory()
    volume_factor(preview)
    before = {p.name: p.read_bytes() for p in saved.directory.iterdir()}
    app = calculate(new_page(), saved.set_id)
    assert app.selectbox(key="factor_name").value == "momentum_20"
    app.radio(key="factor_daily_order").set_value("因子排名").run()
    app.selectbox(key="factor_rank_direction").set_value("ascending").run()
    app.selectbox(key="portfolio_direction").set_value("ascending").run()
    app.selectbox(key="factor_name").set_value("volume_change_20").run()
    assert not app.exception
    assert app.session_state.filtered_state.get("factor_result") is None
    assert not app.dataframe and not app.get("vega_lite_chart")
    assert "factor_rank_direction" not in app.session_state.filtered_state
    assert app.selectbox(key="portfolio_direction").value == "descending"
    calculate(app, saved.set_id)
    text = "\n".join(item.value for item in list(app.caption) + list(app.markdown) + list(app.warning))
    assert "前 20" in text and "21" in text and "成交量" in text
    assert "分母" in text and "0" in text
    assert "期价格动量 =" not in text and "涨幅更高" not in text
    assert "20 期成交量变化" in app.dataframe[0].value.columns
    app.radio(key="factor_daily_order").set_value("因子排名").run()
    assert app.selectbox(key="factor_rank_direction").value == "descending"
    app.selectbox(key="factor_name").set_value("bias_20").run()
    assert app.session_state.filtered_state.get("factor_result") is None
    assert not app.dataframe
    assert {p.name: p.read_bytes() for p in saved.directory.iterdir()} == before


@pytest.mark.parametrize("periods", [20, 21])
def test_page_distinguishes_warmup_from_first_complete_volume_window(volume_page_factory, periods):
    preview, saved = volume_page_factory(periods=periods)
    volume_factor(preview)
    app = new_page()
    app.selectbox(key="factor_name").set_value("volume_change_20").run()
    calculate(app, saved.set_id)
    assert not app.exception
    assert app.dataframe[0].value["20 期成交量变化"].eq("—").all() == (periods == 20)
    assert len(app.get("vega_lite_chart")) == (0 if periods == 20 else 1)
