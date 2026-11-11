"""滚动价格最大回撤：每个20观测窗口重新起峰，而非当前或全历史回撤。"""

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
    assert "max_drawdown_20" in registry.names(), "20期价格最大回撤尚未注册"
    return registry.get("max_drawdown_20")()


def test_recovery_retains_worst_prior_peak_to_trough_in_window():
    instance = factor()
    frame = prices([100.0] * 17 + [120.0, 90.0, 115.0])
    before = frame.copy(deep=True)
    result = instance.evaluate(frame)
    assert result.values.iloc[-1] == pytest.approx(-.25)  # 120 → 90，不是末价115的回撤
    assert result.status.iloc[:19].eq("warmup").all()
    assert result.values.iloc[:19].isna().all()
    assert result.status.iloc[19] == "valid"
    assert result.values.index.identical(frame.index)
    assert result.status.index.identical(frame.index)
    assert result.values.dtype == np.dtype("float64")
    assert isinstance(result.status.dtype, pd.StringDtype)
    assert result.values.name == "max_drawdown_20" and result.status.name == "status"
    pd.testing.assert_series_equal(instance.compute(frame), result.values)
    pd.testing.assert_frame_equal(frame, before)


def test_recorded_definition_and_public_export_match_window_contract():
    import quantlab.factors as factors

    instance = factor()
    assert factors.MaxDrawdown20Factor is type(instance)
    assert asdict(instance.definition) == {
        "name": "max_drawdown_20", "description": "20 期价格最大回撤",
        "formula": "min(close[t-19:t+1] / cummax(close[t-19:t+1]) - 1)",
        "version": "1.0", "required_fields": ("close", "has_observation"),
        "lookback_periods": 19, "window_observations": 20,
        "direction": "higher_is_less_drawdown", "missing_policy": "strict_window",
        "availability": "after_close",
    }


@pytest.mark.parametrize("values,expected", [
    ([100.0] * 20, 0.0),
    (list(range(1, 21)), 0.0),  # 不能拿后面的高点衡量前面的低点
    (list(range(20, 0, -1)), -.95),
    ([100.0] * 16 + [120.0, 90.0, 140.0, 98.0], -.30),
    ([1e308] * 20, 0.0),
    ([1e308] + [5e307] * 19, -.5),
    ([np.nextafter(0.0, 1.0)] * 20, 0.0),
    ([np.nextafter(0.0, 1.0) * 2] + [np.nextafter(0.0, 1.0)] * 19, -.5),
    ([np.nextafter(0.0, 1.0)] * 10 + [1e308] * 10, 0.0),
])
def test_hand_calculation_and_extreme_prices(values, expected):
    with np.errstate(all="raise"):
        settings = np.geterr().copy()
        result = factor().evaluate(prices(values))
        assert np.geterr() == settings
    assert result.status.iloc[-1] == "valid"
    assert result.values.iloc[-1] == pytest.approx(expected)
    if expected == 0:
        assert result.values.iloc[-1] == 0


@pytest.mark.parametrize("values,first,next_value", [
    ([200.0] + [100.0] * 20, -.5, 0.0),  # 旧高点移出，不沿用全历史峰值
    ([100.0, 80.0] + [100.0] * 19, -.2, 0.0),  # 新窗起点是谷值，后续只有上涨
    ([100.0, 90.0] + [100.0] * 18 + [80.0], -.1, -.2),
])
def test_moving_window_restarts_peak(values, first, next_value):
    result = factor().evaluate(prices(values))
    assert result.values.iloc[19] == pytest.approx(first)
    assert result.values.iloc[20] == pytest.approx(next_value)


@pytest.mark.parametrize("low", [1e290, np.nextafter(0.0, 1.0)])
def test_unrepresentable_past_trough_is_not_hidden_by_recovery(low):
    with np.errstate(all="raise"):
        result = factor().evaluate(prices([1e308, low] + [1e308] * 19))
    assert result.status.iloc[19] == "numeric_range"
    assert pd.isna(result.values.iloc[19])
    # 峰值退出，新窗从低价起步；巨大上涨并非回撤，也不该整体缩放为0/0。
    assert result.status.iloc[20] == "valid"
    assert result.values.iloc[20] == 0


@pytest.mark.parametrize("gap", [0, 10, 19, 25])
def test_full_window_gap_propagates_and_recovers(gap):
    frame = prices([100.0] * 50)
    frame.loc[frame.index[gap], ["close", "has_observation"]] = [np.nan, False]
    result = factor().evaluate(frame)
    assert result.status.iloc[:19].eq("warmup").all()
    assert result.status.iloc[max(19, gap):gap + 20].eq("missing_window").all()
    assert result.values.iloc[max(19, gap):gap + 20].isna().all()
    assert result.status.iloc[gap + 20:].eq("valid").all()
    assert result.values.iloc[gap + 20:].eq(0.0).all()


@pytest.mark.parametrize("length", [1, 19, 20])
def test_history_boundary_and_all_absent(length):
    frame = prices([100.0] * length)
    result = factor().evaluate(frame)
    assert result.status.eq("valid").sum() == max(length - 19, 0)
    absent = factor().evaluate(frame.assign(close=np.nan, has_observation=False))
    assert absent.values.isna().all()
    assert absent.status.iloc[:19].eq("warmup").all()
    assert absent.status.iloc[19:].eq("missing_window").all()


def test_future_isolation_scale_invariance_and_independent_outputs():
    frame = prices([100.0] * 45)
    frame.index = pd.MultiIndex.from_arrays([["input"] * 45, frame.index], names=["origin", "row"])
    before = frame.copy(deep=True)
    changed = frame.copy(deep=True)
    changed.loc[changed.index[30:], "close"] = 80.0
    instance = factor()
    early, late = instance.evaluate(frame), instance.evaluate(changed)
    pd.testing.assert_series_equal(early.values.iloc[:30], late.values.iloc[:30])
    assert late.values.iloc[30] == pytest.approx(-.2)
    assert early.values.iloc[30] == 0
    assert late.values.index.identical(frame.index)
    np.testing.assert_allclose(late.values, instance.compute(changed.assign(close=changed.close * 1000)), equal_nan=True)
    pd.testing.assert_series_equal(early.values, instance.compute(frame.assign(volume=0.0)))
    early.values.iloc[-1] = -.8
    early.status.iloc[-1] = "numeric_range"
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("mutate", [
    lambda f: f.iloc[:0], lambda f: f.drop(columns="close"),
    lambda f: f.set_axis([0] * len(f)), lambda f: f.iloc[::-1],
    lambda f: f.assign(date=f.date.astype(str)), lambda f: f.assign(symbol=510300),
    lambda f: f.assign(close=f.close.astype("float32")), lambda f: f.assign(close=0.0),
    lambda f: f.assign(close=np.inf), lambda f: f.assign(close=np.nan),
    lambda f: f.assign(has_observation=1), lambda f: f.assign(has_observation=False),
])
def test_invalid_input_not_repaired(mutate):
    instance = factor()
    with pytest.raises(ValueError):
        instance.evaluate(mutate(prices([100.0] * 20)))
