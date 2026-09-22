"""用可手算的合成行情验证对齐、缺失语义及内容指纹。"""

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantlab.data import alignment as alignment_module
from quantlab.data.calendar import parse_calendar_csv


def align(datasets, **kwargs):
    options = {"start": date(2025, 1, 2), "end": date(2025, 1, 7)}
    options.update(kwargs)
    return alignment_module.align_market_data(datasets, **options)


def supplied_calendar(content=b"date\n2025-01-02\n2025-01-03\n2025-01-06\n2025-01-07\n"):
    return parse_calendar_csv(content, source="synthetic fixture",
                              coverage_start=date(2025, 1, 1), coverage_end=date(2025, 1, 10))


def test_alignment_keeps_every_date_symbol_pair_gaps_and_original_values(sample_assets):
    before = {s: d.copy(deep=True) for s, d in sample_assets.items()}
    result = align(sample_assets)
    assert result.data.columns.tolist() == [
        "date", "symbol", "open", "high", "low", "close", "volume", "has_observation",
    ]
    assert list(zip(result.data.date.dt.strftime("%Y-%m-%d"), result.data.symbol)) == [
        ("2025-01-02", "159915"), ("2025-01-02", "510300"),
        ("2025-01-03", "159915"), ("2025-01-03", "510300"),
        ("2025-01-06", "159915"), ("2025-01-06", "510300"),
        ("2025-01-07", "159915"), ("2025-01-07", "510300"),
    ]
    assert result.data.has_observation.tolist() == [False, True, True, True, True, False, True, True]
    assert result.data.loc[~result.data.has_observation, ["open", "high", "low", "close", "volume"]].isna().all().all()
    assert result.common_dates == (date(2025, 1, 3), date(2025, 1, 7))
    for symbol, source in before.items():
        actual = result.data.loc[result.data.symbol.eq(symbol) & result.data.has_observation]
        pd.testing.assert_frame_equal(
            actual[source.columns].reset_index(drop=True),
            source.assign(date=source.date.astype("datetime64[ns]")).reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(sample_assets[symbol], source)
    assert any("并集" in w and "无法" in w for w in result.warnings)


def test_coverage_is_hand_calculable_and_common_dates_are_not_a_continuous_range(sample_assets):
    result = align(sample_assets)
    assert result.coverage.columns.tolist() == [
        "symbol", "first_observed", "last_observed", "calendar_count", "observation_count",
        "missing_count", "missing_ratio", "longest_missing_run", "zero_volume_count",
    ]
    rows = result.coverage.set_index("symbol")
    for symbol in ("159915", "510300"):
        assert rows.loc[symbol, ["calendar_count", "observation_count", "missing_count",
                                 "missing_ratio", "longest_missing_run", "zero_volume_count"]].tolist() == [4, 3, 1, 0.25, 1, 0]
    assert rows.loc["159915", "first_observed"] == date(2025, 1, 3)
    assert rows.loc["510300", "last_observed"] == date(2025, 1, 7)
    coverage, common = alignment_module.summarize_alignment(result.data)
    pd.testing.assert_frame_equal(coverage, result.coverage)
    assert common == (date(2025, 1, 3), date(2025, 1, 7))


def test_normalizer_copies_converts_and_orders_only_required_columns(sample_assets):
    source = sample_assets["510300"].iloc[::-1].copy()
    source["date"] = source.date.dt.strftime("%Y/%m/%d")
    source["close"] = source.close.astype(str)
    source["volume"] = source.volume.astype("int64")
    source["adjusted_close"] = 999.0
    before = source.copy(deep=True)
    result = alignment_module.normalize_market_frame(source, "510300")
    assert result.columns.tolist() == ["date", "symbol", "open", "high", "low", "close", "volume"]
    assert result.date.dt.strftime("%Y-%m-%d").tolist() == ["2025-01-02", "2025-01-03", "2025-01-07"]
    assert result.close.tolist() == [3.1, 3.1, 3.1]
    assert all(result[c].dtype == np.dtype("float64") for c in ["open", "high", "low", "close", "volume"])
    pd.testing.assert_frame_equal(source, before)


@pytest.mark.parametrize("value", [9007199254740993, "9007199254740993", "9.007199254740993e15"])
def test_normalizer_rejects_integer_value_that_float64_would_silently_change(sample_assets, value):
    source = sample_assets["510300"].copy(deep=True)
    source["volume"] = source.volume.astype(object)
    source.loc[0, "volume"] = value
    with pytest.raises(ValueError, match="精度|表示"):
        alignment_module.normalize_market_frame(source, "510300")


@pytest.mark.parametrize("symbol", [510300, "12345", "510300.SH", "５１０３００", "", None])
def test_normalizer_rejects_invalid_symbol_identity(sample_assets, symbol):
    with pytest.raises(ValueError):
        alignment_module.normalize_market_frame(sample_assets["510300"], symbol)


@pytest.mark.parametrize("failure", [
    "empty", "missing_column", "duplicate_columns", "null", "non_numeric", "positive_inf",
    "negative_inf", "zero_price", "negative_volume", "high", "low", "mixed_symbol", "numeric_symbol",
    "duplicate_date", "normalized_duplicate", "intraday", "timezone", "bool", "complex", "numeric_datetime",
])
def test_normalizer_rejects_bad_full_input_before_any_clipping(sample_assets, failure):
    data = sample_assets["510300"].copy(deep=True)
    if failure == "empty":
        data = data.iloc[:0]
    elif failure == "missing_column":
        data = data.drop(columns="volume")
    elif failure == "duplicate_columns":
        data = pd.concat([data, data[["close"]]], axis=1)
    elif failure in ("duplicate_date", "normalized_duplicate"):
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
        if failure == "normalized_duplicate":
            data["date"] = data.date.astype(object)
            data.loc[3, "date"] = "20250102"
    elif failure in ("intraday", "timezone"):
        data["date"] = data.date.astype(object)
        data.loc[0, "date"] = "2025-01-02 12:00:00" if failure == "intraday" else "2025-01-02T00:00:00Z"
    elif failure in ("mixed_symbol", "numeric_symbol"):
        data["symbol"] = data.symbol.astype(object)
        data.loc[0, "symbol"] = "159915" if failure == "mixed_symbol" else 510300
    else:
        column, value = {
            "null": ("close", np.nan), "non_numeric": ("close", "bad"),
            "positive_inf": ("high", np.inf), "negative_inf": ("low", -np.inf),
            "zero_price": ("low", 0.0), "negative_volume": ("volume", -1.0),
            "high": ("high", 3.05), "low": ("low", 3.05),
            "bool": ("volume", True), "complex": ("close", 3.1 + 2j),
            "numeric_datetime": ("volume", pd.Timestamp("2025-01-02")),
        }[failure]
        data[column] = data[column].astype(object)
        data.loc[0, column] = value
    with pytest.raises(ValueError):
        alignment_module.normalize_market_frame(data, "510300")


@pytest.mark.parametrize("datasets", [{}, {"510300": None}, None])
def test_alignment_requires_at_least_two_assets(datasets):
    with pytest.raises(ValueError):
        align(datasets)


def test_leading_zero_symbol_is_not_lost(sample_assets):
    assets = {"000001": sample_assets["510300"].assign(symbol="000001"),
              "159915": sample_assets["159915"]}
    # 仅验证代码格式保留，不声称 000001 是 ETF。
    assert align(assets).coverage.symbol.tolist() == ["000001", "159915"]


def test_invalid_history_outside_research_range_is_not_hidden(sample_assets):
    sample_assets["510300"].loc[0, "low"] = -1.0
    with pytest.raises(ValueError, match="510300"):
        align(sample_assets, start=date(2025, 1, 3))


def test_shuffled_assets_rows_and_columns_have_identical_results(sample_assets):
    first = align(sample_assets)
    shuffled = {s: d.iloc[::-1, ::-1].copy() for s, d in reversed(list(sample_assets.items()))}
    second = align(shuffled)
    pd.testing.assert_frame_equal(first.data, second.data)
    pd.testing.assert_frame_equal(first.coverage, second.coverage)
    assert first.content_sha256 == second.content_sha256
    assert len(first.content_sha256) == 64


def test_zero_volume_is_an_observation_not_a_gap(sample_assets):
    sample_assets["510300"].loc[1, "volume"] = 0.0
    result = align(sample_assets)
    row = result.data.query("symbol == '510300'").set_index("date").loc["2025-01-03"]
    assert bool(row.has_observation) and row.volume == 0.0 and row.close == 3.1
    assert result.coverage.set_index("symbol").loc["510300", "zero_volume_count"] == 1
    assert any("510300" in w and "零成交量" in w for w in result.warnings)


def test_entire_asset_outside_range_remains_with_full_missingness(sample_assets):
    sample_assets["510300"]["date"] -= pd.Timedelta(days=40)
    result = align(sample_assets)
    row = result.coverage.set_index("symbol").loc["510300"]
    assert row.calendar_count == 3 and row.missing_count == 3
    assert row.observation_count == 0 and row.missing_ratio == 1.0 and row.longest_missing_run == 3
    assert pd.isna(row.first_observed) and pd.isna(row.last_observed)
    assert result.data.loc[result.data.symbol.eq("510300"), "close"].isna().all()
    assert any("510300" in w and "无行情" in w for w in result.warnings)
    assert result.common_dates == ()


@pytest.mark.parametrize("supplied", [False, True])
def test_all_assets_outside_range_are_rejected(sample_assets, supplied):
    for data in sample_assets.values():
        data["date"] -= pd.Timedelta(days=40)
    with pytest.raises(ValueError, match="无行情|没有观测"):
        align(sample_assets, calendar=supplied_calendar() if supplied else None)


def test_disjoint_histories_are_preserved_with_no_common_dates_warning(sample_assets):
    assets = {"510300": sample_assets["510300"].iloc[[0]], "159915": sample_assets["159915"].iloc[[1]]}
    result = align(assets)
    assert len(result.data) == 4 and result.common_dates == ()
    assert any("共同日期" in w for w in result.warnings)


def test_missing_run_counts_axis_positions_not_calendar_days(sample_assets):
    assets = {"510300": sample_assets["510300"].iloc[[0]], "159915": sample_assets["159915"]}
    result = align(assets)
    assert result.coverage.set_index("symbol").loc["510300", "longest_missing_run"] == 3


def test_supplied_calendar_adds_whole_missing_days_and_keeps_before_after_gaps(sample_assets):
    calendar = supplied_calendar(b"date\n2025-01-01\n2025-01-02\n2025-01-03\n2025-01-06\n2025-01-07\n2025-01-08\n")
    result = align(sample_assets, start=date(2025, 1, 1), end=date(2025, 1, 8), calendar=calendar)
    assert len(result.data) == 12
    assert not result.data.loc[result.data.date.isin(pd.to_datetime(["2025-01-01", "2025-01-08"])), "has_observation"].any()
    assert result.coverage.missing_count.tolist() == [3, 3]
    assert result.coverage.longest_missing_run.tolist() == [2, 1]
    assert any("用户提供" in w for w in result.warnings)


def test_calendar_conflicts_are_propagated(sample_assets):
    calendar = supplied_calendar(b"date\n2025-01-02\n2025-01-03\n2025-01-07\n")
    with pytest.raises(ValueError, match="2025-01-06"):
        align(sample_assets, calendar=calendar)


@pytest.mark.parametrize("failure", [
    "missing_row", "extra_row", "row_order", "column_order", "extra_column", "date_string", "intraday",
    "numeric_symbol", "flag_int", "flag_null", "missing_filled", "observed_null", "infinity", "invalid_ohlc",
    "all_missing", "union_unobserved_day",
])
def test_validator_rejects_corrupted_aligned_tables(sample_assets, failure):
    result = align(sample_assets)
    data = result.data.copy(deep=True)
    if failure == "missing_row":
        data = data.iloc[1:]
    elif failure == "extra_row":
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    elif failure == "row_order":
        data = data.iloc[::-1]
    elif failure == "column_order":
        data = data.iloc[:, ::-1]
    elif failure == "extra_column":
        data["tradable"] = True
    elif failure == "date_string":
        data["date"] = data.date.dt.strftime("%Y-%m-%d")
    elif failure == "intraday":
        data.loc[0, "date"] += pd.Timedelta(hours=1)
    elif failure == "numeric_symbol":
        data["symbol"] = data.symbol.astype(int)
    elif failure == "flag_int":
        data["has_observation"] = data.has_observation.astype(int)
    elif failure == "flag_null":
        data["has_observation"] = data.has_observation.astype("boolean")
        data.loc[0, "has_observation"] = pd.NA
    elif failure in ("missing_filled", "observed_null", "infinity", "invalid_ohlc"):
        idx, column, value = {
            "missing_filled": (0, "volume", 0.0), "observed_null": (1, "close", np.nan),
            "infinity": (1, "high", np.inf), "invalid_ohlc": (1, "high", 2.0),
        }[failure]
        data.loc[idx, column] = value
    else:
        indices = data.index if failure == "all_missing" else data.index[data.date.eq(pd.Timestamp("2025-01-03"))]
        data.loc[indices, "has_observation"] = False
        data.loc[indices, ["open", "high", "low", "close", "volume"]] = np.nan
    with pytest.raises(ValueError):
        alignment_module.validate_aligned_data(data, result.calendar, ("159915", "510300"))
    with pytest.raises(ValueError):
        alignment_module.alignment_content_sha256(data, result.calendar)


@pytest.mark.parametrize("changes", [
    {"dates": ()}, {"dates": (date(2025, 1, 3), date(2025, 1, 2))},
    {"dates": (date(2025, 1, 2), date(2025, 1, 2))}, {"mode": "weekdays"},
    {"coverage_start": date(2025, 1, 3)}, {"source": ""},
    {"input_dates_sha256": "0" * 64}, {"file_sha256": "1" * 64},
])
def test_validator_rechecks_fabricated_calendar_axis(sample_assets, changes):
    result = align(sample_assets)
    with pytest.raises(ValueError):
        alignment_module.validate_aligned_data(result.data, replace(result.calendar, **changes), ("159915", "510300"))


@pytest.mark.parametrize("dtype", ["int64", "float32"])
def test_aligned_numeric_types_must_remain_canonical_for_hashing(sample_assets, dtype):
    result = align(sample_assets, start=date(2025, 1, 3), end=date(2025, 1, 3))
    changed = result.data.copy(deep=True)
    changed["volume"] = changed.volume.astype(dtype)
    if dtype == "int64":
        changed.loc[0, "volume"] = 9007199254740993
    with pytest.raises(ValueError, match="类型|float64|精度"):
        alignment_module.alignment_content_sha256(changed, result.calendar)


def test_content_hash_tracks_real_values_and_calendar_declarations_not_csv_order(sample_assets):
    first = align(sample_assets, calendar=supplied_calendar())
    other_bytes = supplied_calendar(b"date\r\n2025-01-07\r\n2025-01-06\r\n2025-01-03\r\n2025-01-02\r\n")
    same = align(sample_assets, calendar=other_bytes)
    assert first.content_sha256 == same.content_sha256
    changed_source = align(sample_assets, calendar=replace(supplied_calendar(), source="another declaration"))
    assert first.content_sha256 != changed_source.content_sha256
    sample_assets["510300"].loc[0, "close"] += 0.00000000001
    changed_price = align(sample_assets, calendar=supplied_calendar())
    assert first.content_sha256 != changed_price.content_sha256
    assert first.content_sha256 == alignment_module.alignment_content_sha256(first.data, first.calendar)
