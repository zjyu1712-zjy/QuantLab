"""年度片段独立手算：跨年不漏区间，局部回撤不沿用全程峰值。"""

from copy import deepcopy
import importlib
import math

import numpy as np
import pandas as pd
import pytest

from quantlab.portfolio import PortfolioBacktestError


def yearly(curve):
    assert importlib.util.find_spec("quantlab.portfolio.yearly") is not None, "年度统计尚未实现"
    return importlib.import_module("quantlab.portfolio.yearly").yearly_equity_statistics(curve)


def curve(dates, values):
    return pd.DataFrame({"date": pd.to_datetime(dates).astype("datetime64[ns]"),
                         "equity": np.asarray(values, dtype="float64")})


def test_year_boundary_uses_previous_observation_and_local_peak():
    source = curve(["2024-12-27", "2024-12-30", "2024-12-31",
                    "2025-01-02", "2025-01-03", "2025-12-31", "2026-01-02"],
                   [100, 200, 150, 120, 180, 144, 158.4])
    before = deepcopy(source)
    result = yearly(source)
    assert result.year.tolist() == [2024, 2025, 2026]
    assert result.period_return.tolist() == pytest.approx([.5, -.04, .1])
    # 2025 starts from 150, not the 2024 all-time high 200, and includes Jan 2's loss.
    assert result.max_drawdown.tolist() == pytest.approx([-.25, -.2, 0])
    assert result.observation_count.tolist() == [3, 3, 1]
    assert result.return_count.tolist() == [2, 3, 1]
    assert result.base_date.tolist() == pd.to_datetime(["2024-12-27", "2024-12-31", "2025-12-31"]).tolist()
    assert result.first_date.tolist() == pd.to_datetime(["2024-12-27", "2025-01-02", "2026-01-02"]).tolist()
    assert math.prod(1 + result.period_return) - 1 == pytest.approx(.584)
    pd.testing.assert_frame_equal(source, before)
    result.loc[0, "period_return"] = 999
    pd.testing.assert_frame_equal(source, before)


def test_year_start_loss_and_cash_warmup_are_not_dropped():
    result = yearly(curve(["2024-12-30", "2024-12-31", "2025-01-02"], [100, 100, 99]))
    assert result.period_return.tolist() == pytest.approx([0, -.01])
    assert result.max_drawdown.tolist() == pytest.approx([0, -.01])


def test_one_observation_is_unavailable_not_zero_return():
    result = yearly(curve(["2025-06-12"], [100])).iloc[0]
    assert result.observation_count == 1 and result.return_count == 0
    assert pd.isna(result.period_return) and pd.isna(result.max_drawdown)
    assert result.status == "no_return_interval"


def test_missing_whole_year_is_explicit_and_bridge_is_not_called_one_year():
    result = yearly(curve(["2023-12-29", "2025-01-02", "2025-01-03"], [100, 90, 99]))
    assert result.year.tolist() == [2023, 2024, 2025]
    missing = result.iloc[1]
    assert missing.observation_count == missing.return_count == 0
    assert pd.isna(missing.base_date) and pd.isna(missing.first_date) and pd.isna(missing.last_date)
    assert pd.isna(missing.period_return) and pd.isna(missing.max_drawdown)
    assert missing.status == "no_observations"
    bridge = result.iloc[2]
    assert bridge.status == "gap_spanning_years"
    assert bridge.base_date == pd.Timestamp("2023-12-29")
    assert bridge.period_return == pytest.approx(-.01)
    assert bridge.max_drawdown == pytest.approx(-.1)


def test_constant_path_has_real_zero_even_for_extreme_equity():
    for value in (1e-300, 1e308):
        result = yearly(curve(["2024-12-31", "2025-01-02", "2025-01-03"], [value] * 3))
        assert result.iloc[1].period_return == result.iloc[1].max_drawdown == 0


def test_row_labels_do_not_define_chronology_and_extra_columns_are_untouched():
    source = curve(["2024-02-29", "2024-07-01", "2025-02-03"], [80, 100, 110])
    source.index = pd.Index([8, 8, 1], name="date")
    source["cash"] = [80, 100, 110]
    before = deepcopy(source)
    result = yearly(source)
    assert result.period_return.tolist() == pytest.approx([.25, .1])
    pd.testing.assert_frame_equal(source, before)


@pytest.mark.parametrize("bad", [
    "none", "empty", "missing_column", "duplicate_column", "reversed", "duplicate_date",
    "missing_date", "string_date", "timezone", "intraday", "nan", "inf", "zero", "negative",
    "bool", "string_equity", "complex",
])
def test_rejects_invalid_curve_without_sorting_filling_or_coercing(bad):
    source = curve(["2024-12-31", "2025-01-02"], [100, 110])
    if bad == "none":
        source = None
    elif bad == "empty":
        source = source.iloc[:0]
    elif bad == "missing_column":
        source = source.drop(columns="equity")
    elif bad == "duplicate_column":
        source = pd.concat([source, source[["equity"]]], axis=1)
    elif bad == "reversed":
        source = source.iloc[::-1]
    elif bad == "duplicate_date":
        source.loc[1, "date"] = source.loc[0, "date"]
    elif bad == "missing_date":
        source.loc[0, "date"] = pd.NaT
    elif bad == "string_date":
        source["date"] = source.date.astype(str)
    elif bad == "timezone":
        source["date"] = source.date.dt.tz_localize("UTC")
    elif bad == "intraday":
        source.loc[0, "date"] += pd.Timedelta(hours=1)
    elif bad in ("nan", "inf", "zero", "negative"):
        source.loc[1, "equity"] = {"nan": np.nan, "inf": np.inf, "zero": 0, "negative": -1}[bad]
    elif bad == "bool":
        source["equity"] = True
    elif bad == "string_equity":
        source["equity"] = source.equity.astype(str)
    elif bad == "complex":
        source["equity"] = source.equity.astype(complex)
    with pytest.raises(PortfolioBacktestError) as caught:
        yearly(source)
    assert caught.value.reason == "invalid_portfolio_result"


@pytest.mark.parametrize("values", [[1e-300, 1e300], [1e300, 1e-300], [1e308, 1, 1e308]])
def test_nonrepresentable_returns_or_drawdown_are_not_infinity_or_minus_one(values):
    with pytest.raises(PortfolioBacktestError) as caught:
        yearly(curve(pd.date_range("2025-01-01", periods=len(values)), values))
    assert caught.value.reason == "numeric_range"
