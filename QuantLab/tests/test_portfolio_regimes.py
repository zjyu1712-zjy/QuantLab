"""基准同期涨跌分组的独立手算；现金/建仓不能充当市场涨跌。"""

from copy import deepcopy
import importlib

import numpy as np
import pandas as pd
import pytest

from quantlab.portfolio import PortfolioBacktestError


def calculate(strategy, benchmark):
    assert importlib.util.find_spec("quantlab.portfolio.regimes") is not None, "基准涨跌分组尚未实现"
    return importlib.import_module("quantlab.portfolio.regimes").benchmark_regime_statistics(strategy, benchmark)


def curve(values, positions=None, dates=None):
    result = pd.DataFrame({"date": pd.to_datetime(dates if dates is not None else
        pd.date_range("2025-01-01", periods=len(values))).astype("datetime64[ns]"),
        "equity": np.array(values, dtype="float64")})
    if positions is not None:
        result["position_value"] = np.array(positions, dtype="float64")
    return result


def test_sign_groups_exclude_cash_and_entry_and_reconcile_all_intervals():
    strategy = curve([100, 110, 110, 121, 114.95, 114.95])
    benchmark = curve([100, 100, 99, 108.9, 98.01, 98.01], [0, 0, 50, 55, 45, 45])
    before = deepcopy((strategy, benchmark))
    result = calculate(strategy, benchmark).set_index("group")
    assert result.interval_count.to_dict() == {"up": 1, "down": 1, "flat": 1, "cash": 1, "entry": 1}
    assert result.interval_count.sum() == len(strategy) - 1
    assert result.loc["up", "strategy_mean_return"] == pytest.approx(.1)
    assert result.loc["down", "strategy_mean_return"] == pytest.approx(-.05)
    assert result.loc["down", "benchmark_mean_return"] == pytest.approx(-.1)
    assert result.loc["down", "mean_difference_pp"] == pytest.approx(5)
    assert result.loc["flat", "strategy_mean_return"] == 0
    assert result.loc[["cash", "entry"], ["strategy_mean_return", "benchmark_mean_return", "mean_difference_pp"]].isna().all().all()
    pd.testing.assert_frame_equal(strategy, before[0])
    pd.testing.assert_frame_equal(benchmark, before[1])


def test_non_contiguous_groups_use_arithmetic_mean_not_compound_return():
    strategy = curve([100, 100, 110, 110, 104.5])
    benchmark = curve([100, 100, 102, 100, 104], [0, 50, 51, 50, 52])
    result = calculate(strategy, benchmark).set_index("group")
    assert result.loc["up", "interval_count"] == 2
    assert result.loc["up", "strategy_mean_return"] == pytest.approx(.025)
    assert result.loc["up", "benchmark_mean_return"] == pytest.approx(.03)
    assert result.loc["up", "mean_difference_pp"] == pytest.approx(-.5)


@pytest.mark.parametrize("entry_equity", [90, 100, 110])
def test_entry_classification_does_not_depend_on_entry_return(entry_equity):
    result = calculate(curve([100,100]), curve([100,entry_equity],[0,50])).set_index("group")
    assert result.loc["entry", "interval_count"] == 1
    assert result.loc[["up","down","flat"], "interval_count"].sum() == 0


@pytest.mark.parametrize("values,positions,cash,entry", [([100],[0],0,0),
    ([100,100,100],[0,0,0],2,0), ([100,100,99],[0,0,50],1,1)])
def test_no_eligible_intervals_have_missing_means_not_zero(values, positions, cash, entry):
    result = calculate(curve(values), curve(values,positions)).set_index("group")
    assert result.loc["cash", "interval_count"] == cash
    assert result.loc["entry", "interval_count"] == entry
    assert result.loc[["up","down","flat"], "interval_count"].sum() == 0
    assert result[["strategy_mean_return","benchmark_mean_return","mean_difference_pp"]].isna().all().all()


def test_tiny_nonzero_return_is_not_flat_and_dates_are_not_filled():
    values = [1., 1., np.nextafter(1., 2.), 1.]
    dates = ["2020-12-31","2022-01-03","2025-01-01","2026-01-01"]
    strategy, benchmark = curve(values,dates=dates), curve(values,[0,.5,.5,.5],dates)
    strategy.index = [9,9,2,3]
    benchmark.index = [4,2,7,7]
    result = calculate(strategy,benchmark).set_index("group")
    assert result.loc["up","interval_count"] == result.loc["down","interval_count"] == 1
    assert result.loc["flat","interval_count"] == 0 and result.interval_count.sum() == 3


@pytest.mark.parametrize("bad", ["none", "empty", "missing_column", "duplicate_column", "date_text",
    "date_tz", "intraday", "missing_date", "duplicate_date", "reorder", "mismatch", "equity_text",
    "equity_bool", "nan", "inf", "zero", "negative", "position_missing", "position_nan",
    "position_inf", "position_negative", "position_text", "position_bool", "position_above_equity",
    "initial_holding", "exit", "second_entry"])
def test_invalid_curves_are_not_repaired_or_partially_grouped(bad):
    strategy = curve([100,100,110,110])
    benchmark = curve([100,100,110,110],[0,50,55,55])
    if bad == "none": benchmark = None
    elif bad == "empty": benchmark = benchmark.iloc[:0]
    elif bad == "missing_column": benchmark = benchmark.drop(columns="equity")
    elif bad == "duplicate_column": benchmark = pd.concat([benchmark,benchmark[["equity"]]],axis=1)
    elif bad == "date_text": benchmark["date"] = benchmark.date.astype(str)
    elif bad == "date_tz": benchmark["date"] = benchmark.date.dt.tz_localize("UTC")
    elif bad == "intraday": benchmark.loc[1,"date"] += pd.Timedelta(hours=1)
    elif bad == "missing_date": benchmark.loc[1,"date"] = pd.NaT
    elif bad == "duplicate_date": benchmark.loc[1,"date"] = benchmark.loc[0,"date"]
    elif bad == "reorder": benchmark = benchmark.iloc[::-1]
    elif bad == "mismatch": strategy = strategy.iloc[:-1]
    elif bad == "equity_text": strategy["equity"] = strategy.equity.astype(str)
    elif bad == "equity_bool": strategy["equity"] = True
    elif bad in ("nan","inf","zero","negative"):
        strategy.loc[2,"equity"] = {"nan":np.nan,"inf":np.inf,"zero":0.,"negative":-1.}[bad]
    elif bad == "position_missing": benchmark = benchmark.drop(columns="position_value")
    elif bad in ("position_nan","position_inf","position_negative","position_above_equity"):
        benchmark.loc[1,"position_value"] = {"position_nan":np.nan,"position_inf":np.inf,
                                             "position_negative":-1.,"position_above_equity":200.}[bad]
    elif bad == "position_text": benchmark["position_value"] = benchmark.position_value.astype(str)
    elif bad == "position_bool": benchmark["position_value"] = False
    elif bad == "initial_holding": benchmark.loc[0,"position_value"] = 50.
    elif bad == "exit": benchmark.loc[3,"position_value"] = 0.
    elif bad == "second_entry": benchmark.loc[2,"position_value"] = 0.
    with pytest.raises(PortfolioBacktestError) as caught:
        calculate(strategy, benchmark)
    assert caught.value.reason == "invalid_comparison"


@pytest.mark.parametrize("values", [[1.,1.,1e308], [1e308,1e308,1.], [1e-308,1e-308,1e308]])
def test_unreliable_return_or_percentage_points_fail_safely(values):
    strategy = curve(values)
    benchmark = curve([1.,1.,2.],[0.,.5,1.])
    with pytest.raises(PortfolioBacktestError) as caught:
        calculate(strategy,benchmark)
    assert caught.value.reason == "numeric_range"
