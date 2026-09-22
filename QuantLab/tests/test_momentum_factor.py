"""20 期动量契约：合成价格手算、完整窗口、严格输入及无未来行依赖。"""

from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from quantlab.factors import FactorRegistry, Momentum20Factor, create_default_registry


def prices(n=45):
    """工作日仅用来制造自然日间隔，不冒充官方交易日历。"""
    return pd.DataFrame({
        "date": pd.date_range("2025-01-02", periods=n, freq="B").astype("datetime64[ns]"),
        "symbol": "510300",
        "close": 100.0 + np.arange(n) * 0.5,
        "has_observation": True,
    }, index=pd.Index(np.arange(n) * 3 + 7, name="source_row"))


def test_hand_calculation_metadata_and_default_registry():
    factor = Momentum20Factor()
    frame = prices(22)
    out = factor.evaluate(frame)
    assert out.values.iloc[:20].isna().all()
    assert out.status.iloc[:20].eq("warmup").all()
    assert out.values.iloc[20] == pytest.approx(0.10)
    assert out.values.iloc[21] == pytest.approx(110.5 / 100.5 - 1)
    assert out.status.iloc[20:].eq("valid").all()
    assert out.values.index.identical(frame.index)
    assert out.status.index.identical(frame.index)
    assert out.values.dtype == np.dtype("float64")
    assert isinstance(out.status.dtype, pd.StringDtype)
    assert out.values.name == "momentum_20" and out.status.name == "status"
    pd.testing.assert_series_equal(factor.compute(frame), out.values)
    assert asdict(factor.definition) == {
        "name": "momentum_20", "description": "20 期价格动量",
        "formula": "close[t] / close[t-20] - 1", "version": "1.0",
        "required_fields": ("close", "has_observation"),
        "lookback_periods": 20, "window_observations": 21,
        "direction": "higher_is_stronger", "missing_policy": "strict_window",
        "availability": "after_close",
    }
    left, right = create_default_registry(), create_default_registry()
    assert left is not right
    assert left.names() == right.names() == ("momentum_20",)
    assert left.get("momentum_20") is Momentum20Factor
    assert FactorRegistry().names() == ()


def test_default_registries_do_not_share_mutable_registrations():
    class TestOnlyFactor(Momentum20Factor):
        name = "test_only"
        definition = replace(Momentum20Factor.definition, name=name)

    left, right = create_default_registry(), create_default_registry()
    left.register(TestOnlyFactor)
    assert left.names() == ("momentum_20", "test_only")
    assert right.names() == create_default_registry().names() == ("momentum_20",)


@pytest.mark.parametrize(("change", "expected"), [(0.0, 0.0), (-0.5, -0.10)])
def test_constant_and_falling_prices(change, expected):
    frame = prices(21)
    frame["close"] = 100.0 + np.arange(21) * change
    actual = Momentum20Factor().compute(frame)
    assert actual.iloc[20] == pytest.approx(expected)


@pytest.mark.parametrize("gap", [0, 5, 20])
def test_full_window_missing_and_recovery(gap):
    frame = prices()
    frame.loc[frame.index[gap], ["close", "has_observation"]] = [np.nan, False]
    out = Momentum20Factor().evaluate(frame)
    assert out.status.iloc[:20].eq("warmup").all()
    assert out.status.iloc[20:gap + 21].eq("missing_window").all()
    assert out.values.iloc[20:gap + 21].isna().all()
    assert out.status.iloc[gap + 21:].eq("valid").all()
    # 从缺失行离开窗口的第一个位置恢复，不能提前一行。
    assert out.values.iloc[gap + 21] == pytest.approx(
        (100.0 + (gap + 21) * 0.5) / (100.0 + (gap + 1) * 0.5) - 1,
    )


def test_short_all_missing_and_zero_volume():
    for length in (1, 20):
        short = Momentum20Factor().evaluate(prices(length))
        assert short.values.isna().all() and short.status.eq("warmup").all()
    absent = prices().assign(close=np.nan, has_observation=False)
    out = Momentum20Factor().evaluate(absent)
    assert out.values.isna().all()
    assert out.status.iloc[:20].eq("warmup").all()
    assert out.status.iloc[20:].eq("missing_window").all()
    pd.testing.assert_series_equal(
        Momentum20Factor().compute(prices().assign(volume=0.0)),
        Momentum20Factor().compute(prices()),
    )


@pytest.mark.parametrize(("first", "last"), [
    (1e-308, 1e308), (1e308, 1e-308), (1e20, 1.0),
])
def test_unrepresentable_ratio_is_not_a_return(first, last):
    frame = prices(21)
    frame.loc[frame.index[[0, 20]], "close"] = [first, last]
    with np.errstate(all="raise"):
        settings = np.geterr().copy()
        out = Momentum20Factor().evaluate(frame)
        assert np.geterr() == settings
    assert out.status.iloc[20] == "numeric_range"
    assert pd.isna(out.values.iloc[20])
    assert not np.isinf(out.values.to_numpy()).any()


def test_missing_window_takes_precedence_over_extreme_ratio():
    frame = prices(21)
    frame.loc[frame.index[[0, 20]], "close"] = [1e-308, 1e308]
    frame.loc[frame.index[5], ["close", "has_observation"]] = [np.nan, False]
    out = Momentum20Factor().evaluate(frame)
    assert out.status.iloc[20] == "missing_window"
    assert pd.isna(out.values.iloc[20])


def test_only_past_rows_are_used_and_input_is_untouched():
    frame = prices()
    before = frame.copy(deep=True)
    early = Momentum20Factor().evaluate(frame)
    changed = frame.copy(deep=True)
    changed.loc[changed.index[30:], "close"] *= 2
    later = Momentum20Factor().evaluate(changed)
    pd.testing.assert_series_equal(early.values.iloc[:30], later.values.iloc[:30])
    pd.testing.assert_series_equal(early.status.iloc[:30], later.status.iloc[:30])
    # 防止一个“总是返回相同结果”的错误实现碰巧通过隔离断言。
    assert later.values.iloc[30] != early.values.iloc[30]
    early.values.iloc[-1] = 999.0
    early.status.iloc[-1] = "numeric_range"
    pd.testing.assert_frame_equal(frame, before)


def test_unique_multiindex_labels_and_calendar_gaps_are_preserved():
    frame = prices(21)
    frame.index = pd.MultiIndex.from_arrays(
        [["source"] * len(frame), np.arange(len(frame)) * 5], names=["origin", "row"],
    )
    out = Momentum20Factor().evaluate(frame)
    assert out.values.index.identical(frame.index)
    assert out.status.index.identical(frame.index)
    assert (frame.date.iloc[-1] - frame.date.iloc[0]).days > 20
    assert out.values.iloc[-1] == pytest.approx(0.10)


@pytest.mark.parametrize("field", ["date", "symbol", "close", "has_observation"])
def test_missing_required_column_is_rejected(field):
    with pytest.raises(ValueError, match="字段"):
        Momentum20Factor().evaluate(prices().drop(columns=field))


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda f: f.iloc[:0], id="empty"),
    pytest.param(lambda f: f.drop(columns="close"), id="missing-close"),
    pytest.param(lambda f: pd.concat([f, f[["close"]]], axis=1), id="duplicate-columns"),
    pytest.param(lambda f: f.set_axis([0] * len(f)), id="duplicate-index"),
    pytest.param(lambda f: f.iloc[::-1], id="unsorted-dates"),
    pytest.param(lambda f: f.assign(date=f.date.iloc[0]), id="duplicate-dates"),
    pytest.param(lambda f: f.assign(date=f.date.astype(str)), id="string-dates"),
    pytest.param(lambda f: f.assign(date=f.date.astype("datetime64[us]")), id="microsecond-dates"),
    pytest.param(lambda f: f.assign(date=f.date.dt.tz_localize("UTC")), id="timezone-dates"),
    pytest.param(lambda f: f.assign(date=f.date + pd.Timedelta(hours=1)), id="intraday-dates"),
    pytest.param(lambda f: f.assign(date=f.date.mask(f.index == f.index[0])), id="missing-date"),
    pytest.param(lambda f: f.assign(symbol=510300), id="numeric-symbol"),
    pytest.param(lambda f: f.assign(symbol="51030"), id="short-symbol"),
    pytest.param(lambda f: f.assign(symbol=["159915"] + ["510300"] * (len(f) - 1)), id="multiple-symbols"),
    pytest.param(lambda f: f.assign(symbol=None), id="missing-symbol"),
    pytest.param(lambda f: f.assign(close=f.close.astype("float32")), id="float32-price"),
    pytest.param(lambda f: f.assign(close=f.close.astype("Float64")), id="nullable-price"),
    pytest.param(lambda f: f.assign(close=f.close.astype(str)), id="string-price"),
    pytest.param(lambda f: f.assign(close=True), id="bool-price"),
    pytest.param(lambda f: f.assign(close=0.0), id="zero-price"),
    pytest.param(lambda f: f.assign(close=-1.0), id="negative-price"),
    pytest.param(lambda f: f.assign(close=np.inf), id="infinite-price"),
    pytest.param(lambda f: f.assign(close=np.nan), id="missing-observed-price"),
    pytest.param(lambda f: f.assign(has_observation=1), id="numeric-observation-flag"),
    pytest.param(lambda f: f.assign(has_observation=False), id="filled-missing-price"),
    pytest.param(lambda f: f.assign(has_observation=pd.Series(pd.NA, index=f.index, dtype="boolean")), id="missing-observation-flag"),
])
def test_invalid_input_is_not_silently_normalized(mutate):
    with pytest.raises(ValueError):
        Momentum20Factor().evaluate(mutate(prices()))


def test_non_dataframe_rejected_and_nullable_bools_without_na_allowed():
    with pytest.raises(ValueError):
        Momentum20Factor().evaluate(None)
    frame = prices()
    expected = Momentum20Factor().compute(frame)
    frame["has_observation"] = frame.has_observation.astype("boolean")
    pd.testing.assert_series_equal(Momentum20Factor().compute(frame), expected)
