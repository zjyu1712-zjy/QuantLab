"""带方向的20期趋势效率：路径效率不是价格涨幅，恒价与缺失不能混淆。"""

from fractions import Fraction

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
    assert "trend_efficiency_20" in registry.names(), "趋势效率因子尚未注册"
    return registry.get("trend_efficiency_20")()


def exact_efficiency(values):
    """独立有理数参考：逐对计算距离，不调用生产辅助函数。"""
    points = [Fraction(float(value)) for value in values]
    distance = sum(abs(right - left) for left, right in zip(points, points[1:]))
    return float((points[-1] - points[0]) / distance) if distance else 0.0


def test_signed_path_efficiency_metadata_and_unchanged_input():
    import quantlab.factors as factors

    frame = prices([100.0] * 18 + [110.0, 105.0, 115.0])
    before = frame.copy(deep=True)
    instance = factor()
    result = instance.evaluate(frame)
    assert result.values.iloc[20] == pytest.approx(.6)  # 净变化15 / 总路程25
    assert result.status.iloc[:20].eq("warmup").all()
    assert result.values.iloc[:20].isna().all()
    assert result.status.iloc[20] == "valid"
    assert result.values.index.identical(frame.index)
    assert result.status.index.identical(frame.index)
    assert result.values.dtype == np.dtype("float64")
    assert isinstance(result.status.dtype, pd.StringDtype)
    assert result.values.name == "trend_efficiency_20" and result.status.name == "status"
    pd.testing.assert_series_equal(instance.compute(frame), result.values)
    pd.testing.assert_frame_equal(frame, before)
    assert factors.TrendEfficiency20Factor is type(instance)
    definition = instance.definition
    assert definition.required_fields == ("close", "has_observation")
    assert (definition.lookback_periods, definition.window_observations) == (20, 21)
    assert definition.direction == "higher_is_more_upward_efficient"
    assert definition.availability == "after_close" and definition.missing_policy == "strict_window"
    assert definition.version == "1.0"


@pytest.mark.parametrize("values,expected", [
    (list(range(100, 121)), 1.0),
    (list(range(120, 99, -1)), -1.0),
    ([100.0] * 21, 0.0),
    ([100.0] * 20 + [101.0], 1.0),
    ([101.0] * 20 + [100.0], -1.0),
    ([100.0, 110.0] * 10 + [100.0], 0.0),
    ([100.0, 110.0] * 10 + [110.0], 1 / 19),
    ([115.0] * 18 + [105.0, 110.0, 100.0], -.6),
    ([1e308] * 21, 0.0),
    ([np.nextafter(0., 1.)] * 21, 0.0),
    ([np.nextafter(0., 1.)] * 10 + [1e308] * 11, 1.0),
    ([1e308] * 10 + [np.nextafter(0., 1.)] * 11, -1.0),
    ([1e-308, 1e308] * 10 + [1e-308], 0.0),  # 总路程溢出也不能误丢真实零
])
def test_monotonic_flat_and_oscillating_hand_calculations(values, expected):
    with np.errstate(all="raise"):
        settings = np.geterr().copy()
        result = factor().evaluate(prices(values))
        assert np.geterr() == settings
    assert result.status.iloc[-1] == "valid"
    assert result.values.iloc[-1] == pytest.approx(expected)
    assert -1 <= result.values.iloc[-1] <= 1
    if expected == 0:
        assert result.values.iloc[-1] == 0


def test_small_nonzero_moves_near_large_price_do_not_get_flattened_by_price_scaling():
    first = 1e308
    up = np.nextafter(first, np.inf)
    second_up = np.nextafter(up, np.inf)
    result = factor().evaluate(prices([first] * 18 + [up, first, second_up]))
    assert result.status.iloc[-1] == "valid"
    assert result.values.iloc[-1] == pytest.approx(.5, abs=1e-15)


@pytest.mark.parametrize("values", [
    [1.0] * 19 + [1e308, 2.0],
    [2.0] * 19 + [1e308, 1.0],
    [1e-15] * 19 + [1e308, 2e-15],
    [2**-54, float(2**1022)] + [5 * 2**-54] * 19,
    [5 * 2**-54, float(2**1022)] + [2**-54] * 19,
])
def test_representable_subnormal_efficiency_not_misreported_as_zero(values):
    expected = exact_efficiency(values)
    assert expected != 0
    with np.errstate(all="raise"):
        result = factor().evaluate(prices(values))
    assert result.status.iloc[-1] == "valid"
    assert result.values.iloc[-1] == expected


@pytest.mark.parametrize("first,last", [(2**-54, 5 * 2**-54), (5 * 2**-54, 2**-54)])
def test_net_smaller_than_representable_efficiency_is_not_a_flat_path(first, last):
    window = [first, float(2**1023)] + [last] * 19
    assert first != last and exact_efficiency(window) == 0.0
    result = factor().evaluate(prices(window))
    assert result.status.iloc[-1] == "numeric_range"
    assert pd.isna(result.values.iloc[-1])


def test_unrepresentable_nonzero_net_path_is_unavailable_then_recovers():
    tiny = np.nextafter(0., 1.)
    frame = prices([tiny] * 19 + [1e308, tiny * 2] + [tiny * 2] * 21)
    with np.errstate(all="raise"):
        result = factor().evaluate(frame)
    assert result.status.iloc[20] == "numeric_range"
    assert pd.isna(result.values.iloc[20])
    assert result.status.iloc[39] == "valid" and result.values.iloc[39] == -1.0
    assert result.status.iloc[40] == "valid" and result.values.iloc[40] == 0.0


def test_rolling_window_uses_new_start_not_full_history():
    result = factor().evaluate(prices([100.0, 120.0] + [110.0] * 20))
    assert result.values.iloc[20] == pytest.approx(1 / 3)
    assert result.values.iloc[21] == -1.0


@pytest.mark.parametrize("gap", [0, 10, 20, 25])
def test_missing_window_recovers_without_compressing_dates(gap):
    frame = prices([100.0] * 52)
    frame.loc[frame.index[gap], ["close", "has_observation"]] = [np.nan, False]
    result = factor().evaluate(frame)
    assert result.status.iloc[:20].eq("warmup").all()
    assert result.status.iloc[max(20, gap):gap + 21].eq("missing_window").all()
    assert result.values.iloc[max(20, gap):gap + 21].isna().all()
    assert result.status.iloc[gap + 21:].eq("valid").all()
    assert result.values.iloc[gap + 21:].eq(0.0).all()


@pytest.mark.parametrize("length", [1, 20, 21, 22])
def test_history_and_all_missing_boundaries(length):
    frame = prices([100.0] * length)
    result = factor().evaluate(frame)
    assert result.status.eq("valid").sum() == max(length - 20, 0)
    absent = factor().evaluate(frame.assign(close=np.nan, has_observation=False))
    assert absent.values.isna().all()
    assert absent.status.iloc[:20].eq("warmup").all()
    assert absent.status.iloc[20:].eq("missing_window").all()


def test_future_isolation_scale_invariance_and_independent_outputs():
    frame = prices([100.0] * 45)
    frame.index = pd.MultiIndex.from_arrays([["input"] * 45, frame.index], names=["origin", "row"])
    before = frame.copy(deep=True)
    changed = frame.copy(deep=True)
    changed.loc[changed.index[30:], "close"] = 110.0
    instance = factor()
    early, late = instance.evaluate(frame), instance.evaluate(changed)
    pd.testing.assert_series_equal(early.values.iloc[:30], late.values.iloc[:30])
    assert late.values.iloc[30] == 1.0 and early.values.iloc[30] == 0
    assert late.values.index.identical(frame.index)
    np.testing.assert_allclose(late.values, instance.compute(changed.assign(close=changed.close * 100)), equal_nan=True)
    pd.testing.assert_series_equal(early.values, instance.compute(frame.assign(volume=0.0)))
    pd.testing.assert_series_equal(late.values.iloc[:32], instance.compute(changed.iloc[:32]))
    early.values.iloc[-1] = -.8
    early.status.iloc[-1] = "numeric_range"
    pd.testing.assert_frame_equal(frame, before)
    assert instance.compute(frame).iloc[-1] == 0


@pytest.mark.parametrize("mutate", [
    lambda f: f.iloc[:0], lambda f: f.drop(columns="close"),
    lambda f: f.set_axis([0] * len(f)), lambda f: f.iloc[::-1],
    lambda f: f.assign(date=f.date.astype(str)), lambda f: f.assign(symbol=510300),
    lambda f: f.assign(close=f.close.astype("float32")), lambda f: f.assign(close=0.0),
    lambda f: f.assign(close=np.inf), lambda f: f.assign(close=np.nan),
    lambda f: f.assign(has_observation=1), lambda f: f.assign(has_observation=False),
])
def test_invalid_price_input_not_repaired(mutate):
    instance = factor()
    with pytest.raises(ValueError):
        instance.evaluate(mutate(prices([100.0] * 21)))
