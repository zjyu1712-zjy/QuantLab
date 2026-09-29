"""滚动波动率：独立手算、严格窗口、有限数值与因果边界。"""

from math import sqrt

import numpy as np
import pandas as pd
import pytest

from quantlab.factors import create_default_registry


def prices(close):
    return pd.DataFrame({
        "date": pd.date_range("2025-01-02", periods=len(close), freq="B").astype("datetime64[ns]"),
        "symbol": "510300", "close": np.asarray(close, dtype="float64"),
        "has_observation": True,
    }, index=pd.Index(np.arange(len(close)) * 3 + 7, name="source_row"))


def factor(periods):
    name = f"volatility_{periods}"
    registry = create_default_registry()
    assert name in registry.names(), "波动率因子尚未注册"
    return registry.get(name)()


@pytest.mark.parametrize("periods", [20, 60])
def test_one_nonzero_return_matches_sample_standard_deviation(periods):
    import quantlab.factors as factors

    selected = factor(periods)
    assert getattr(factors, f"Volatility{periods}Factor") is type(selected)
    # N 个收益率中一个为 10%、其余为 0；样本标准差 = 0.1 / sqrt(N)。
    frame = prices([100.0] + [110.0] * (periods + 1))
    before = frame.copy(deep=True)
    out = selected.evaluate(frame)
    assert out.values.iloc[:periods].isna().all()
    assert out.status.iloc[:periods].eq("warmup").all()
    assert out.values.iloc[periods] == pytest.approx(.1 * sqrt(252 / periods))
    assert out.values.iloc[periods + 1] == 0.0
    assert out.status.iloc[periods:].eq("valid").all()
    assert out.values.name == f"volatility_{periods}"
    assert out.values.index.identical(frame.index)
    assert out.status.index.identical(frame.index)
    assert out.values.dtype == np.dtype("float64")
    assert selected.definition.window_observations == periods + 1
    assert selected.definition.direction == "higher_is_more_volatile"
    assert "ddof=1" in selected.definition.formula and "sqrt(252)" in selected.definition.formula
    pd.testing.assert_series_equal(selected.compute(frame), out.values)
    out.values.iloc[-1] = 999
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("periods", [20, 60])
@pytest.mark.parametrize("ratio", [1.0, 2.0, .5])
def test_constant_returns_are_zero_even_when_price_moves(periods, ratio):
    frame = prices(100 * ratio ** np.arange(periods + 2))
    out = factor(periods).evaluate(frame)
    assert out.status.iloc[periods:].eq("valid").all()
    assert out.values.iloc[periods:].eq(0.0).all()


@pytest.mark.parametrize("periods", [20, 60])
@pytest.mark.parametrize("gap_kind", ["first", "middle", "last"])
def test_missing_price_invalidates_whole_window_and_then_recovers(periods, gap_kind):
    frame = prices(100.0 + np.arange(2 * periods + 2))
    gap = {"first": 0, "middle": periods // 2, "last": periods}[gap_kind]
    frame.loc[frame.index[gap], ["close", "has_observation"]] = [np.nan, False]
    out = factor(periods).evaluate(frame)
    assert out.status.iloc[:periods].eq("warmup").all()
    assert out.status.iloc[periods:gap + periods + 1].eq("missing_window").all()
    assert out.values.iloc[periods:gap + periods + 1].isna().all()
    assert out.status.iloc[gap + periods + 1:].eq("valid").all()


@pytest.mark.parametrize("periods", [20, 60])
def test_short_and_all_missing_history_are_not_zero(periods):
    selected = factor(periods)
    short = selected.evaluate(prices([100.0] * periods))
    assert short.values.isna().all() and short.status.eq("warmup").all()
    absent = selected.evaluate(prices([100.0] * (periods + 2)).assign(close=np.nan, has_observation=False))
    assert absent.values.isna().all()
    assert absent.status.iloc[periods:].eq("missing_window").all()


@pytest.mark.parametrize("periods", [20, 60])
def test_future_rows_do_not_change_earlier_volatility(periods):
    frame = prices(100.0 + np.arange(periods + 5))
    changed = frame.copy(deep=True)
    changed.loc[changed.index[-1], "close"] *= 2
    early = factor(periods).evaluate(frame)
    later = factor(periods).evaluate(changed)
    pd.testing.assert_series_equal(early.values.iloc[:-1], later.values.iloc[:-1])
    pd.testing.assert_series_equal(early.status.iloc[:-1], later.status.iloc[:-1])
    assert early.values.iloc[-1] != later.values.iloc[-1]


@pytest.mark.parametrize("periods", [20, 60])
def test_representable_large_volatility_does_not_overflow_variance(periods):
    # 平方会溢出，但标准差本身可表示；不能因此丢失有效值。
    frame = prices([1e-200] + [1e-40] * periods)
    with np.errstate(all="raise"):
        out = factor(periods).evaluate(frame)
    assert out.status.iloc[-1] == "valid"
    assert out.values.iloc[-1] == pytest.approx(1e160 * sqrt(252 / periods))


@pytest.mark.parametrize("periods", [20, 60])
@pytest.mark.parametrize(("first", "rest"), [(1e-308, 1e308), (1e308, 1.0), (1e-308, 1.0)])
def test_unrepresentable_returns_or_annualized_result_are_unavailable_then_recover(periods, first, rest):
    frame = prices([first] + [rest] * (periods + 1))
    with np.errstate(all="raise"):
        out = factor(periods).evaluate(frame)
    assert out.status.iloc[periods] == "numeric_range"
    assert pd.isna(out.values.iloc[periods])
    assert out.status.iloc[periods + 1] == "valid"
    assert out.values.iloc[periods + 1] == 0.0
    frame.loc[frame.index[3], ["close", "has_observation"]] = [np.nan, False]
    assert factor(periods).evaluate(frame).status.iloc[periods] == "missing_window"


@pytest.mark.parametrize("mutate", [
    lambda f: f.iloc[:0], lambda f: f.assign(close=0.0),
    lambda f: f.assign(close=np.inf), lambda f: f.assign(close=np.nan),
    lambda f: f.assign(has_observation=False), lambda f: f.assign(has_observation=1),
    lambda f: f.drop(columns="date"), lambda f: f.iloc[::-1],
    lambda f: f.assign(symbol=["159915"] + ["510300"] * 20),
])
def test_invalid_input_is_not_repaired(mutate):
    selected = factor(20)
    with pytest.raises(ValueError):
        selected.evaluate(mutate(prices([100.0] * 21)))
