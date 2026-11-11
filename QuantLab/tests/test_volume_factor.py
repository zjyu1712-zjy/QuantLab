"""成交量相对此前20期均量：排除当期、零值语义、严格窗口及独立输入。"""

import numpy as np
import pandas as pd
import pytest

from quantlab.factors import create_default_registry


def volumes(values):
    return pd.DataFrame({
        "date": pd.date_range("2025-01-02", periods=len(values), freq="B").astype("datetime64[ns]"),
        "symbol": "510300", "volume": np.asarray(values, dtype="float64"),
        "has_observation": True,
    }, index=pd.Index(np.arange(len(values)) * 3 + 7, name="source_row"))


def factor():
    registry = create_default_registry()
    assert "volume_change_20" in registry.names(), "成交量变化因子尚未注册"
    return registry.get("volume_change_20")()


def test_current_volume_excluded_from_mean_and_metadata_preserved():
    import quantlab.factors as factors

    frame = volumes([100.0] * 20 + [150.0])
    before = frame.copy(deep=True)
    instance = factor()
    result = instance.evaluate(frame)
    assert result.values.iloc[-1] == pytest.approx(.5)
    assert result.status.iloc[:20].eq("warmup").all()
    assert result.values.iloc[:20].isna().all()
    assert result.status.iloc[20] == "valid"
    assert result.values.index.identical(frame.index)
    assert result.status.index.identical(frame.index)
    assert result.values.dtype == np.dtype("float64")
    assert isinstance(result.status.dtype, pd.StringDtype)
    assert result.values.name == "volume_change_20" and result.status.name == "status"
    pd.testing.assert_series_equal(instance.compute(frame), result.values)
    pd.testing.assert_frame_equal(frame, before)
    assert factors.VolumeChange20Factor is type(instance)
    definition = instance.definition
    assert definition.required_fields == ("volume", "has_observation")
    assert (definition.lookback_periods, definition.window_observations) == (20, 21)
    assert definition.formula == "volume[t] / mean(volume[t-20:t]) - 1"
    assert definition.direction == "higher_is_more_volume"
    assert definition.availability == "after_close" and definition.missing_policy == "strict_window"
    assert definition.version == "1.0"


@pytest.mark.parametrize("values,expected", [
    ([100.0] * 21, 0.0),
    ([100.0] * 20 + [50.0], -.5),
    ([100.0] * 20 + [0.0], -1.0),  # 真零不是缺失，也不能被-100%过滤规则删除
    ([0.0] * 19 + [100.0, 10.0], 1.0),  # 分母为100/20，不能丢弃历史零值
    ([1e308] * 21, 0.0),  # 直接求和会溢出，但均量与比值可以表示
    ([np.nextafter(0., 1.)] * 21, 0.0),
    ([0.0] * 19 + [np.nextafter(0., 1.)] * 2, 19.0),  # 均量低于最小浮点数，比值仍可表示
    ([1e308] * 20 + [0.0], -1.0),
])
def test_hand_calculations_and_extreme_but_representable_windows(values, expected):
    with np.errstate(all="raise"):
        settings = np.geterr().copy()
        result = factor().evaluate(volumes(values))
        assert np.geterr() == settings
    assert result.status.iloc[-1] == "valid"
    assert result.values.iloc[-1] == pytest.approx(expected)
    if expected == 0:
        assert result.values.iloc[-1] == 0


@pytest.mark.parametrize("current", [0.0, 100.0])
def test_zero_historical_mean_is_unavailable_and_recovers(current):
    result = factor().evaluate(volumes([0.0] * 20 + [current, 100.0, 100.0]))
    assert result.status.iloc[20] == "numeric_range"
    assert pd.isna(result.values.iloc[20])
    if current > 0:
        assert result.status.iloc[21] == "valid"
        assert result.values.iloc[21] == pytest.approx(19.0)
    else:
        assert result.status.iloc[21] == "numeric_range"
    assert result.status.iloc[22] == "valid"


@pytest.mark.parametrize("prior,current", [
    (np.nextafter(0., 1.), 1e308),  # 正向溢出
    (1e308, np.nextafter(0., 1.)),  # 正量不能伪装成真实零量的-100%
    (1e308, 1e290),
])
def test_unrepresentable_ratios_stay_unavailable(prior, current):
    with np.errstate(all="raise"):
        result = factor().evaluate(volumes([prior] * 20 + [current]))
    assert result.status.iloc[-1] == "numeric_range"
    assert pd.isna(result.values.iloc[-1])


def test_rolling_history_drops_oldest_and_does_not_use_fixed_initial_average():
    # 第21条：200/((19*100+200)/20)-1 = 19/21；第22条旧的200移出，均量仍为105。
    result = factor().evaluate(volumes([200.0] + [100.0] * 19 + [200.0, 105.0]))
    assert result.values.iloc[20] == pytest.approx(19 / 21)
    assert result.values.iloc[21] == pytest.approx(0.0)


@pytest.mark.parametrize("gap", [0, 10, 20, 25])
def test_gap_is_not_zero_and_recovers_only_after_entire_window_leaves(gap):
    frame = volumes([100.0] * 52)
    frame.loc[frame.index[gap], ["volume", "has_observation"]] = [np.nan, False]
    result = factor().evaluate(frame)
    assert result.status.iloc[:20].eq("warmup").all()
    assert result.status.iloc[max(20, gap):gap + 21].eq("missing_window").all()
    assert result.values.iloc[max(20, gap):gap + 21].isna().all()
    assert result.status.iloc[gap + 21:].eq("valid").all()
    assert result.values.iloc[gap + 21:].eq(0.0).all()


@pytest.mark.parametrize("length", [1, 20, 21, 22])
def test_warmup_boundaries_and_all_absent(length):
    frame = volumes([100.0] * length)
    result = factor().evaluate(frame)
    assert result.status.eq("valid").sum() == max(length - 20, 0)
    absent = factor().evaluate(frame.assign(volume=np.nan, has_observation=False))
    assert absent.values.isna().all()
    assert absent.status.iloc[:20].eq("warmup").all()
    assert absent.status.iloc[20:].eq("missing_window").all()


def test_future_isolation_unit_invariance_and_no_price_dependency():
    frame = volumes([100.0] * 45)
    frame.index = pd.MultiIndex.from_arrays([["input"] * 45, frame.index], names=["origin", "row"])
    before = frame.copy(deep=True)
    changed = frame.copy(deep=True)
    changed.loc[changed.index[30:], "volume"] = 150.0
    instance = factor()
    early, late = instance.evaluate(frame), instance.evaluate(changed)
    pd.testing.assert_series_equal(early.values.iloc[:30], late.values.iloc[:30])
    assert late.values.iloc[30] == pytest.approx(.5)
    assert early.values.iloc[30] == 0
    assert late.values.index.identical(frame.index)
    np.testing.assert_allclose(late.values, instance.compute(changed.assign(volume=changed.volume * 100)), equal_nan=True)
    pd.testing.assert_series_equal(early.values, instance.compute(frame.assign(close=-99.0)))
    early.values.iloc[-1] = -.8
    early.status.iloc[-1] = "numeric_range"
    pd.testing.assert_frame_equal(frame, before)
    assert instance.compute(frame).iloc[-1] == 0


@pytest.mark.parametrize("mutate", [
    lambda f: f.iloc[:0], lambda f: f.drop(columns="volume"),
    lambda f: f.set_axis([0] * len(f)), lambda f: f.iloc[::-1],
    lambda f: f.assign(date=f.date.astype(str)), lambda f: f.assign(symbol=510300),
    lambda f: f.assign(volume=f.volume.astype("float32")),
    lambda f: f.assign(volume=f.volume.astype("int64")), lambda f: f.assign(volume=True),
    lambda f: f.assign(volume=-1.0), lambda f: f.assign(volume=np.inf),
    lambda f: f.assign(volume=np.nan), lambda f: f.assign(has_observation=1),
    lambda f: f.assign(has_observation=False),
    lambda f: pd.concat([f, f[["volume"]]], axis=1),
])
def test_invalid_volume_input_rejected_without_conversion(mutate):
    instance = factor()
    with pytest.raises(ValueError):
        instance.evaluate(mutate(volumes([100.0] * 21)))
