"""20 条收盘价均线偏离：独立手算、完整窗口与浮点边界。"""

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from quantlab.factors import create_default_registry


def prices(values):
    return pd.DataFrame({
        "date": pd.date_range("2025-01-02", periods=len(values), freq="B").astype("datetime64[ns]"),
        "symbol": "510300", "close": np.asarray(values, dtype="float64"),
        "has_observation": True,
    }, index=pd.Index(np.arange(len(values)) * 3 + 7, name="source_row"))


def factor():
    registry = create_default_registry()
    assert "bias_20" in registry.names(), "20 期均线偏离尚未注册"
    return registry.get("bias_20")()


def test_formula_window_metadata_public_export_and_identity():
    import quantlab.factors as factors

    instance = factor()
    assert factors.Bias20Factor is type(instance)
    frame = prices([100.0] * 19 + [110.0, 90.0])
    before = frame.copy(deep=True)
    out = instance.evaluate(frame)
    assert out.status.iloc[:19].eq("warmup").all()
    assert out.values.iloc[:19].isna().all()
    assert out.status.iloc[19:].eq("valid").all()
    assert out.values.iloc[19] == pytest.approx(19 / 201)  # 均价 100.5
    assert out.values.iloc[20] == pytest.approx(-0.1)  # 均价 100；最早一条已离窗
    assert out.values.index.identical(frame.index)
    assert out.status.index.identical(frame.index)
    assert out.values.dtype == np.dtype("float64")
    assert isinstance(out.status.dtype, pd.StringDtype)
    assert out.values.name == "bias_20" and out.status.name == "status"
    pd.testing.assert_series_equal(instance.compute(frame), out.values)
    pd.testing.assert_frame_equal(frame, before)
    assert asdict(instance.definition) == {
        "name": "bias_20", "description": "20 期均线偏离",
        "formula": "close[t] / mean(close[t-19:t+1]) - 1", "version": "1.0",
        "required_fields": ("close", "has_observation"),
        "lookback_periods": 19, "window_observations": 20,
        "direction": "higher_is_more_above_ma", "missing_policy": "strict_window",
        "availability": "after_close",
    }


@pytest.mark.parametrize("values,expected", [
    ([100.0] * 20, 0.0),
    (list(range(1, 21)), 19 / 21),  # 均价 10.5，当期 20
    (list(range(20, 0, -1)), -19 / 21),
    ([1e308] * 20, 0.0),  # 原始求和溢出，但结果可表示
    ([1e307] * 19 + [1e308], 171 / 29),
    ([1e308] * 19 + [5e307], -19 / 39),
    ([np.nextafter(0.0, 1.0)] * 20, 0.0),
    ([1e308] + [np.nextafter(0.0, 1.0)] * 18 + [1e307], 9 / 11),
    ([np.nextafter(0.0, 1.0)] * 19 + [1e308], 19.0),
])
def test_hand_values_and_extreme_finite_prices(values, expected):
    with np.errstate(all="raise"):
        settings = np.geterr().copy()
        out = factor().evaluate(prices(values))
        assert np.geterr() == settings
    assert out.status.iloc[-1] == "valid"
    assert out.values.iloc[-1] == pytest.approx(expected)
    if expected == 0:
        assert out.values.iloc[-1] == 0


@pytest.mark.parametrize("last", [1e290, np.nextafter(0.0, 1.0)])
def test_rounded_negative_one_is_unavailable_and_next_window_can_recover(last):
    with np.errstate(all="raise"):
        out = factor().evaluate(prices([1e308] * 19 + [last, 1e308]))
    assert out.status.iloc[19] == "numeric_range"
    assert pd.isna(out.values.iloc[19])
    assert out.status.iloc[20] == "valid"
    assert out.values.iloc[20] == pytest.approx(1 / 19)


@pytest.mark.parametrize("gap", [0, 10, 19, 25])
def test_missing_input_invalidates_whole_window_until_gap_leaves(gap):
    frame = prices([100.0] * 50)
    frame.loc[frame.index[gap], ["close", "has_observation"]] = [np.nan, False]
    before = frame.copy(deep=True)
    out = factor().evaluate(frame)
    assert out.status.iloc[:19].eq("warmup").all()
    assert out.status.iloc[max(19, gap):gap + 20].eq("missing_window").all()
    assert out.values.iloc[max(19, gap):gap + 20].isna().all()
    assert out.status.iloc[gap + 20:].eq("valid").all()
    assert out.values.iloc[gap + 20:].eq(0).all()
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("length", [1, 19, 20])
def test_short_history_and_absent_observations(length):
    frame = prices([100.0] * length)
    out = factor().evaluate(frame)
    assert out.status.eq("valid").sum() == max(length - 19, 0)
    absent = factor().evaluate(frame.assign(close=np.nan, has_observation=False))
    assert absent.values.isna().all()
    assert absent.status.iloc[:19].eq("warmup").all()
    assert absent.status.iloc[19:].eq("missing_window").all()


def test_future_independence_scale_invariance_and_output_copies():
    frame = prices(100.0 + np.arange(45))
    frame.index = pd.MultiIndex.from_arrays([["input"] * 45, frame.index], names=["origin", "row"])
    before = frame.copy(deep=True)
    instance = factor()
    original = instance.evaluate(frame)
    changed = frame.copy(deep=True)
    changed.loc[changed.index[30:], "close"] *= 2
    later = instance.evaluate(changed)
    pd.testing.assert_series_equal(original.values.iloc[:30], later.values.iloc[:30])
    assert original.values.iloc[30] != later.values.iloc[30]
    assert original.values.index.identical(frame.index)
    scaled = instance.compute(frame.assign(close=frame.close * 1000))
    np.testing.assert_allclose(original.values, scaled, atol=1e-15, equal_nan=True)
    pd.testing.assert_series_equal(original.values, instance.compute(frame.assign(volume=0.0)))
    original.values.iloc[-1] = 999
    original.status.iloc[-1] = "numeric_range"
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("mutate", [
    lambda f: f.iloc[:0], lambda f: f.drop(columns="close"),
    lambda f: f.set_axis([0] * len(f)), lambda f: f.iloc[::-1],
    lambda f: f.assign(date=f.date.astype(str)), lambda f: f.assign(symbol=510300),
    lambda f: f.assign(close=f.close.astype("float32")), lambda f: f.assign(close=0.0),
    lambda f: f.assign(close=np.inf), lambda f: f.assign(close=np.nan),
    lambda f: f.assign(has_observation=1), lambda f: f.assign(has_observation=False),
])
def test_invalid_input_is_rejected_not_repaired(mutate):
    instance = factor()
    with pytest.raises(ValueError):
        instance.evaluate(mutate(prices([100.0] * 20)))
