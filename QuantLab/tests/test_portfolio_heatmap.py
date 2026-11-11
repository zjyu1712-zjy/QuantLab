"""展示层保留原参数网格、真实零值、失败与未定义，不能重算绩效。"""

from copy import deepcopy
import importlib
import json

import numpy as np
import pandas as pd
import pytest


def ui():
    assert importlib.util.find_spec("quantlab.ui.portfolio_heatmap") is not None, "参数热力图尚未实现"
    return importlib.import_module("quantlab.ui.portfolio_heatmap")


def results():
    return pd.DataFrame([
        dict(top_n=n, rebalance_interval=k, status="complete", cumulative_return=value,
             annualized_return=value, annualized_volatility=abs(value), max_drawdown=-abs(value),
             sharpe_ratio=value, trade_count=0 if value == 0 else 3,
             failure_reason="", unavailable_reasons={})
        for (n, k), value in zip([(n, k) for n in (1, 2) for k in (5, 10, 20)],
                                 [0., .12345, -.2, .5, .04, -.1])])


def rows(spec):
    data = spec["data"]
    return data.get("values", spec.get("datasets", {}).get(data.get("name")))


@pytest.mark.parametrize("metric,want,number_format", [
    ("cumulative_return", "12.35%", ".2%"), ("annualized_return", "12.35%", ".2%"),
    ("annualized_volatility", "12.35%", ".2%"), ("max_drawdown", "-12.35%", ".2%"),
    ("sharpe_ratio", "0.12", ".2f"),
])
def test_chart_uses_raw_values_discrete_order_and_consistent_units(metric, want, number_format):
    source = results().iloc[::-1].reset_index(drop=True)
    before = deepcopy(source.to_dict("records"))
    spec = ui().build_parameter_heatmap(source, metric=metric).to_dict()
    cells = rows(spec)
    assert [(c["top_n"], c["rebalance_interval"]) for c in cells] == [
        (1, 5), (1, 10), (1, 20), (2, 5), (2, 10), (2, 20)]
    assert cells[1]["value"] == (-.12345 if metric == "max_drawdown" else .12345)
    assert cells[1]["label"] == want
    assert cells[0]["value"] == 0 and cells[0]["state"] == "完成（零成交）"
    assert source.to_dict("records") == before
    assert spec["width"] == "container"
    rect = spec["layer"][0]["encoding"]
    assert rect["x"]["type"] == rect["y"]["type"] == "ordinal"
    assert rect["x"]["sort"] == [5, 10, 20] and rect["y"]["sort"] == [1, 2]
    assert rect["color"]["condition"]["legend"]["format"] == number_format
    assert rect["tooltip"][2]["field"] == "label" and rect["tooltip"][2]["type"] == "nominal"
    assert spec["layer"][1]["encoding"]["text"]["field"] == "label"


def test_failed_metric_is_never_colored_and_undefined_keeps_specific_reason():
    source = results()
    source.loc[1, "status"] = "failed"
    source.loc[1, "failure_reason"] = "zero_volume"
    source.loc[2, "sharpe_ratio"] = np.nan
    source.at[2, "unavailable_reasons"] = {"sharpe_ratio": "收益率标准差为零"}
    spec = ui().build_parameter_heatmap(source, metric="sharpe_ratio").to_dict()
    cells = rows(spec)
    assert cells[1]["value"] is None and cells[1]["label"] == "失败"
    assert "成交量" in cells[1]["reason"]
    assert cells[2]["value"] is None and cells[2]["label"] == "不可用"
    assert cells[2]["reason"] == "收益率标准差为零"
    assert cells[0]["value"] == 0 and cells[0]["label"] == "0.00"
    assert spec["layer"][0]["encoding"]["color"]["value"] == "#e5e7eb"
    assert spec["layer"][0]["mark"]["invalid"] == "show", "灰格不能被图表编译器过滤"
    assert spec.get("config", {}).get("scale", {}).get("invalid", {}).get("color") == {"value": "#e5e7eb"}, "缺失值不得回退为零值色阶"
    assert spec["layer"][0]["encoding"]["tooltip"][2]["field"] == "label", "null 数字格式化会伪装为零"


@pytest.mark.parametrize("mode", ["all_failed", "all_nan", "all_inf", "zero", "same_positive", "same_negative"])
def test_all_missing_or_constant_values_keep_all_cells_and_finite_color_domain(mode):
    source = results()
    if mode == "all_failed":
        source["status"] = "failed"
    else:
        source["cumulative_return"] = {"all_nan": np.nan, "all_inf": np.inf, "zero": 0.,
                                       "same_positive": .12, "same_negative": -.12}[mode]
    spec = json.loads(ui().build_parameter_heatmap(source, metric="cumulative_return").to_json())
    assert len(rows(spec)) == 6
    domain = spec["layer"][0]["encoding"]["color"]["condition"]["scale"]["domain"]
    assert np.isfinite(domain).all() and domain[0] < domain[1]
    if mode in {"all_failed", "all_nan", "all_inf"}:
        assert all(c["value"] is None for c in rows(spec))
    else:
        assert len({c["value"] for c in rows(spec)}) == 1


def test_nine_cells_preserved_without_mutating_nested_reasons():
    source = results()
    third = source.loc[source.top_n.eq(2)].copy(deep=True)
    third["top_n"] = 3
    source = pd.concat([source, third], ignore_index=True)
    before = deepcopy(source.to_dict("records"))
    chart = ui().build_parameter_heatmap(source, metric="max_drawdown")
    assert len(rows(chart.to_dict())) == 9
    assert source.to_dict("records") == before


def test_cell_labels_remain_readable_across_light_and_dark_fills():
    source = results()
    source["cumulative_return"] = [0., .3, .55, .69, .7, 1.]
    spec = ui().build_parameter_heatmap(source, metric="cumulative_return").to_dict()
    scale = spec["layer"][0]["encoding"]["color"]["condition"]["scale"]
    def rgb(color):
        return [int(color[i:i+2], 16) for i in (1, 3, 5)]
    def luminance(channels):
        values = [c / 255 for c in channels]
        linear = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in values]
        return sum(c * w for c, w in zip(linear, [.2126, .7152, .0722]))
    start, end = map(rgb, scale["range"])
    for cell in rows(spec):
        fill = [round(a + (b - a) * cell["value"]) for a, b in zip(start, end)]
        a, b = sorted([luminance(fill), luminance(rgb(cell["text_color"]))])
        assert (b + .05) / (a + .05) >= 4.5


def test_literal_text_colors_do_not_merge_with_numeric_fill_scale():
    spec = ui().build_parameter_heatmap(results(), metric="cumulative_return").to_dict()
    # Vega-Lite 不能将矩形的定量色阶与文字的 literal/scale=None 合并。
    assert spec.get("resolve", {}).get("scale", {}).get("color") == "independent"


def test_chart_preserves_plot_height_when_streamlit_fits_width():
    spec = ui().build_parameter_heatmap(results(), metric="cumulative_return").to_dict()
    assert spec.get("autosize", {}).get("type") == "fit-x"
    assert spec["height"] >= 140


def test_large_finite_percent_keeps_finite_label_and_raw_value():
    source = results()
    source["annualized_return"] = 1e307
    cells = rows(ui().build_parameter_heatmap(source, metric="annualized_return").to_dict())
    assert all(c["value"] == 1e307 and c["valid"] for c in cells)
    assert all(c["label"] == "1.00E+309%" for c in cells)


@pytest.mark.parametrize("problem", ["empty", "duplicate", "missing_cell", "bad_n", "bad_k", "status", "columns", "metric"])
def test_malformed_plot_input_is_rejected_not_silently_aggregated(problem):
    source = results()
    if problem == "empty": source = source.iloc[:0]
    elif problem == "duplicate": source = pd.concat([source, source.iloc[:1]])
    elif problem == "missing_cell": source = source.iloc[1:]
    elif problem == "bad_n": source.loc[0, "top_n"] = 4
    elif problem == "bad_k": source.loc[0, "rebalance_interval"] = 7
    elif problem == "status": source.loc[0, "status"] = "unknown"
    elif problem == "columns": source = source.drop(columns="status")
    with pytest.raises(ValueError):
        ui().build_parameter_heatmap(source, metric="unknown" if problem == "metric" else "cumulative_return")
