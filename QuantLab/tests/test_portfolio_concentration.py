"""集中度的独立手算；分母是全部正资产增量，而不是净利润。"""

from copy import deepcopy
import importlib

import numpy as np
import pandas as pd
import pytest

from quantlab.portfolio import PortfolioBacktestError


def calculate(source):
    assert importlib.util.find_spec("quantlab.portfolio.concentration") is not None, "收益集中度尚未实现"
    return importlib.import_module("quantlab.portfolio.concentration").equity_gain_concentration(source)


def curve(values, dates=None):
    return pd.DataFrame({"date": pd.to_datetime(dates if dates is not None else
                         pd.date_range("2025-01-01", periods=len(values))).astype("datetime64[ns]"),
                         "equity": np.asarray(values, dtype="float64")})


def test_top_five_uses_all_positive_changes_not_net_profit():
    source = curve([100, 101, 103, 106, 110, 115, 121, 90])
    before = deepcopy(source)
    result = calculate(source)
    assert result.interval_count == 7 and result.positive_count == 6 and result.selected_count == 5
    assert result.gross_positive_gain == 21 and result.top_gain == 20
    assert result.top5_share == pytest.approx(20 / 21) and result.status == "available"
    assert result.intervals.asset_gain.tolist() == [6, 5, 4, 3, 2]
    assert result.intervals.positive_gain_share.tolist() == pytest.approx([6/21, 5/21, 4/21, 3/21, 2/21])
    assert result.intervals.iloc[0].start_date == pd.Timestamp("2025-01-06")
    assert result.intervals.iloc[0].end_date == pd.Timestamp("2025-01-07")
    pd.testing.assert_frame_equal(source, before)
    result.intervals.loc[0, "asset_gain"] = 999
    pd.testing.assert_frame_equal(source, before)


@pytest.mark.parametrize("values,gross,count", [([100,110,105,125,123],30,2),
    ([100,110,80],10,1), ([100,101,102,103,104,105],5,5)])
def test_fewer_or_exactly_five_positive_intervals_report_actual_count(values, gross, count):
    result = calculate(curve(values))
    assert result.gross_positive_gain == result.top_gain == gross
    assert result.positive_count == result.selected_count == count
    assert result.top5_share == 1 and len(result.intervals) == count


@pytest.mark.parametrize("values,status", [([100],"no_return_interval"),
    ([100,100,100],"no_positive_gain"), ([100,90,80],"no_positive_gain")])
def test_missing_concentration_is_not_zero(values, status):
    result = calculate(curve(values))
    assert result.status == status and result.top5_share is None
    assert result.positive_count == result.selected_count == 0
    assert result.gross_positive_gain == result.top_gain == 0
    assert result.interval_count == len(values)-1 and result.intervals.empty


def test_ties_use_earliest_end_date_without_expanding_top_five():
    source = curve(range(100, 108))
    source.index = [9, 9, 1, 3, 2, 4, 7, 0]
    result = calculate(source)
    assert result.selected_count == 5 and result.top5_share == pytest.approx(5/7)
    assert result.intervals.end_date.tolist() == pd.date_range("2025-01-02", periods=5).tolist()


def test_date_gaps_remain_observation_intervals_not_daily_returns():
    source = curve([100,110,130], ["2020-12-31", "2022-01-03", "2026-01-01"])
    result = calculate(source)
    assert result.interval_count == 2
    assert result.intervals.start_date.tolist() == pd.to_datetime(["2022-01-03", "2020-12-31"]).tolist()
    assert result.intervals.end_date.tolist() == pd.to_datetime(["2026-01-01", "2022-01-03"]).tolist()


def test_tiny_positive_changes_are_not_discarded_as_zero():
    result = calculate(curve([1e-300, 2e-300, 1.5e-300]))
    assert result.positive_count == 1 and result.top5_share == 1
    assert result.gross_positive_gain == 1e-300


@pytest.mark.parametrize("bad", ["none", "empty", "column", "duplicate_column", "date_order",
    "date_duplicate", "date_missing", "date_text", "date_tz", "intraday", "nan", "inf",
    "zero", "negative", "equity_text", "equity_bool", "equity_complex"])
def test_invalid_input_is_rejected_not_sorted_filled_or_coerced(bad):
    source = curve([100,110])
    if bad == "none": source = None
    elif bad == "empty": source = source.iloc[:0]
    elif bad == "column": source = source.drop(columns="equity")
    elif bad == "duplicate_column": source = pd.concat([source, source[["equity"]]], axis=1)
    elif bad == "date_order": source = source.iloc[::-1]
    elif bad == "date_duplicate": source.loc[1,"date"] = source.loc[0,"date"]
    elif bad == "date_missing": source.loc[1,"date"] = pd.NaT
    elif bad == "date_text": source["date"] = source.date.astype(str)
    elif bad == "date_tz": source["date"] = source.date.dt.tz_localize("UTC")
    elif bad == "intraday": source.loc[1,"date"] += pd.Timedelta(hours=1)
    elif bad in ("nan", "inf", "zero", "negative"):
        source.loc[1,"equity"] = {"nan": np.nan, "inf": np.inf, "zero": 0, "negative": -1}[bad]
    elif bad == "equity_text": source["equity"] = source.equity.astype(str)
    elif bad == "equity_bool": source["equity"] = True
    elif bad == "equity_complex": source["equity"] = source.equity.astype(complex)
    with pytest.raises(PortfolioBacktestError) as caught:
        calculate(source)
    assert caught.value.reason == "invalid_portfolio_result"


@pytest.mark.parametrize("values", [[1,1e308,1,1e308], [1e-308,2e-308,1e308]])
def test_overflow_or_share_underflow_does_not_create_infinity_or_fake_zero(values):
    with pytest.raises(PortfolioBacktestError) as caught:
        calculate(curve(values))
    assert caught.value.reason == "numeric_range"
