"""日历测试只使用可手算的日期和内存 CSV，不访问联网日历或私人数据。"""

from dataclasses import replace
from datetime import date, datetime, timezone
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from quantlab.data import calendar as calendar_module


def parse(content=b"date\n2025-01-02\n2025-01-03\n2025-01-06\n", **kwargs):
    options = {
        "source": "synthetic fixture",
        "coverage_start": date(2025, 1, 1),
        "coverage_end": date(2025, 1, 7),
    }
    options.update(kwargs)
    return calendar_module.parse_calendar_csv(content, **options)


def test_normalize_daily_dates_retains_index_order_and_duplicates_without_mutation():
    values = pd.Series([
        "2025-01-06", date(2025, 1, 2), datetime(2025, 1, 3),
        pd.Timestamp("2025-01-03"), np.datetime64("2025-01-02"),
        "2025-01-02 00:00:00", "2025-01-02T00:00:00.000000000",
        "2025/01/02", "20250102",
    ], index=[9, 2, 7, 1, 5, 3, 4, 6, 8], name="trade_date", dtype=object)
    before = values.copy(deep=True)
    result = calendar_module.normalize_daily_dates(values)
    expected = pd.Series(pd.to_datetime([
        "2025-01-06", "2025-01-02", "2025-01-03", "2025-01-03",
        "2025-01-02", "2025-01-02", "2025-01-02", "2025-01-02", "2025-01-02",
    ]), index=values.index, name="trade_date", dtype="datetime64[ns]")
    pd.testing.assert_series_equal(result, expected)
    pd.testing.assert_series_equal(values, before)


@pytest.mark.parametrize("value", [
    None, pd.NaT, pd.NA, np.nan, "", "   ", "not-a-date", "2025-02-30",
    20250102, 1735776000.0, True, np.int64(20250102),
    "2025-01-02 13:00:00", "2025-01-02T00:00:00.000000001",
    datetime(2025, 1, 2, 0, 0, 0, 1), np.datetime64("2025-01-02T00:00:00.000000001"),
    "2025-01-02T00:00:00+08:00", "2025-01-02T00:00:00Z",
    pd.Timestamp("2025-01-02", tz="Asia/Shanghai"), datetime(2025, 1, 2, tzinfo=timezone.utc),
    "today", "2025", "2025-01", "01/02/2025", [],
])
def test_normalize_rejects_non_daily_or_ambiguous_values(value):
    with pytest.raises(ValueError, match="日期|日线|时区|时间"):
        calendar_module.normalize_daily_dates(pd.Series([value], dtype=object))


def test_empty_date_series_has_a_stable_datetime_type():
    values = pd.Series([], dtype=object, name="date")
    result = calendar_module.normalize_daily_dates(values)
    pd.testing.assert_series_equal(result, pd.Series([], dtype="datetime64[ns]", name="date"))


def test_calendar_sorts_dates_but_keeps_explicit_coverage_and_source():
    content = b"date\n2025-01-06\n2025-01-02\n2025-01-03\n"
    calendar = parse(content, source="  synthetic fixture  ")
    assert calendar.dates == (date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6))
    assert calendar.source == "synthetic fixture"
    assert calendar.coverage_start == date(2025, 1, 1)
    assert calendar.coverage_end == date(2025, 1, 7)
    assert calendar.file_sha256 == sha256(content).hexdigest()
    assert calendar.dates_sha256 == sha256(b'["2025-01-02","2025-01-03","2025-01-06"]').hexdigest()


def test_calendar_accepts_utf8_bom_and_hashes_original_bytes():
    content = b"\xef\xbb\xbfdate\r\n2025-01-02\r\n"
    calendar = parse(content)
    assert calendar.dates == (date(2025, 1, 2),)
    assert calendar.file_sha256 == sha256(content).hexdigest()


def test_calendar_date_fingerprint_ignores_row_order_and_date_spelling():
    first = parse(b"date\n2025-01-03\n2025-01-02\n")
    second = parse(b"date\r\n20250102\r\n2025/01/03\r\n")
    assert first.dates_sha256 == second.dates_sha256
    assert first.file_sha256 != second.file_sha256


@pytest.mark.parametrize("content", [
    b"", b"date\n", b"date\n\n", b"date\n2025-01-02\n\n2025-01-03\n",
    b"date\n2025-01-02\n2025-01-02 00:00:00\n",
    b"date\n2025-01-02\n20250102\n", b"date\n2025-01-02 13:00:00\n",
    b"date\n2025-01-02T00:00:00+08:00\n", b"date\n2025-02-30\n",
    b"date,other\n2025-01-02,x\n", b"date,date\n2025-01-02,2025-01-03\n",
    b"date\n2025-01-02,extra\n", b"other\n2025-01-02\n",
    b'date\n"2025-01-02\n', b"date\n\xff\n", b"date\n2024-12-31\n",
    b"date\n2025-01-08\n", b"date\n1735776000\n",
])
def test_calendar_rejects_invalid_csv_instead_of_dropping_rows(content):
    with pytest.raises(ValueError):
        parse(content)


@pytest.mark.parametrize("options", [
    {"source": ""}, {"source": "  "}, {"source": None},
    {"coverage_start": date(2025, 1, 8)}, {"coverage_start": "2025-01-01"},
    {"coverage_end": None}, {"coverage_end": datetime(2025, 1, 7)},
])
def test_calendar_requires_valid_coverage_and_source(options):
    with pytest.raises(ValueError):
        parse(**options)


def test_observed_union_sorts_deduplicates_and_clips_without_inventing_weekdays():
    observed = (
        date(2025, 1, 8), date(2025, 1, 6), date(2025, 1, 2),
        date(2024, 12, 31), date(2025, 1, 2),
    )
    axis = calendar_module.resolve_calendar(observed, start=date(2025, 1, 2), end=date(2025, 1, 6))
    assert axis.dates == (date(2025, 1, 2), date(2025, 1, 6))
    assert axis.mode == "observed_union"
    assert "并集" in axis.source and "未经" in axis.source
    assert axis.coverage_start == date(2025, 1, 2)
    assert axis.coverage_end == date(2025, 1, 6)
    assert axis.file_sha256 is None
    assert axis.input_dates_sha256 == sha256(b'["2025-01-02","2025-01-06"]').hexdigest()


def test_explicit_calendar_preserves_day_missing_from_all_assets_and_holiday_endpoints():
    calendar = parse()
    axis = calendar_module.resolve_calendar(
        (date(2025, 1, 2), date(2025, 1, 6)),
        start=date(2025, 1, 1), end=date(2025, 1, 7), calendar=calendar,
    )
    assert axis.mode == "supplied"
    assert axis.dates == (date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6))
    assert axis.source == calendar.source
    assert axis.coverage_start == date(2025, 1, 1)
    assert axis.coverage_end == date(2025, 1, 7)
    assert axis.file_sha256 == calendar.file_sha256
    assert axis.input_dates_sha256 == calendar.dates_sha256


def test_supplied_axis_clips_dates_but_keeps_whole_input_fingerprint():
    calendar = parse()
    axis = calendar_module.resolve_calendar(
        (date(2024, 12, 31), date(2025, 1, 3), date(2025, 1, 8)),
        start=date(2025, 1, 3), end=date(2025, 1, 3), calendar=calendar,
    )
    assert axis.dates == (date(2025, 1, 3),)
    assert axis.input_dates_sha256 == calendar.dates_sha256
    assert axis.coverage_start == date(2025, 1, 1)
    assert axis.coverage_end == date(2025, 1, 7)


def test_calendar_conflict_reports_the_dates_instead_of_dropping_observations():
    with pytest.raises(ValueError, match="2025-01-07"):
        calendar_module.resolve_calendar(
            (date(2025, 1, 2), date(2025, 1, 7)),
            start=date(2025, 1, 1), end=date(2025, 1, 7), calendar=parse(),
        )


@pytest.mark.parametrize(("start", "end"), [
    (date(2024, 12, 31), date(2025, 1, 7)),
    (date(2025, 1, 1), date(2025, 1, 8)),
])
def test_research_range_must_fit_declared_calendar_coverage(start, end):
    with pytest.raises(ValueError, match="覆盖"):
        calendar_module.resolve_calendar((date(2025, 1, 2),), start=start, end=end, calendar=parse())


@pytest.mark.parametrize(("start", "end"), [
    (date(2025, 1, 7), date(2025, 1, 1)),
    (None, date(2025, 1, 7)), ("2025-01-01", date(2025, 1, 7)),
    (date(2025, 1, 1), datetime(2025, 1, 7)),
])
def test_research_range_requires_ordered_date_objects(start, end):
    with pytest.raises(ValueError):
        calendar_module.resolve_calendar((date(2025, 1, 2),), start=start, end=end)


def test_empty_union_cannot_create_an_axis():
    with pytest.raises(ValueError, match="日期|观测"):
        calendar_module.resolve_calendar((date(2024, 1, 2),), start=date(2025, 1, 1), end=date(2025, 1, 7))


def test_supplied_calendar_needs_at_least_one_date_inside_research_range():
    with pytest.raises(ValueError, match="日期|交易日"):
        calendar_module.resolve_calendar((), start=date(2025, 1, 4), end=date(2025, 1, 5), calendar=parse())


def test_calendar_axis_does_not_decide_whether_assets_have_observations():
    # 后续对齐模块负责拒绝全部无行情；日历层仍能给出所声明的日期轴。
    axis = calendar_module.resolve_calendar((), start=date(2025, 1, 1), end=date(2025, 1, 7), calendar=parse())
    assert axis.dates == (date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6))


@pytest.mark.parametrize("bad_date", ["2025-01-02", datetime(2025, 1, 2), pd.NaT, None, 20250102])
def test_observed_dates_must_already_be_pure_dates(bad_date):
    with pytest.raises(ValueError):
        calendar_module.resolve_calendar((bad_date,), start=date(2025, 1, 1), end=date(2025, 1, 7))


@pytest.mark.parametrize("changes", [
    {"dates": ()}, {"dates": [date(2025, 1, 2)]},
    {"dates": (date(2025, 1, 3), date(2025, 1, 2))},
    {"dates": (date(2025, 1, 2), date(2025, 1, 2))},
    {"dates": ("2025-01-02",)}, {"dates": (datetime(2025, 1, 2),)},
    {"dates": (date(2025, 1, 8),)}, {"source": " "},
    {"coverage_start": "2025-01-01"}, {"coverage_end": date(2024, 1, 1)},
    {"dates_sha256": "0" * 64}, {"file_sha256": "not-a-hash"},
])
def test_programmatic_calendar_is_revalidated_before_use(changes):
    calendar = replace(parse(), **changes)
    with pytest.raises(ValueError):
        calendar_module.resolve_calendar(
            (date(2025, 1, 2),), start=date(2025, 1, 1), end=date(2025, 1, 7), calendar=calendar,
        )


def test_valid_programmatic_calendar_can_omit_file_hash():
    calendar = calendar_module.CalendarInput(
        dates=(date(2025, 1, 2),), source="programmatic fixture",
        coverage_start=date(2025, 1, 1), coverage_end=date(2025, 1, 7),
        file_sha256=None, dates_sha256=sha256(b'["2025-01-02"]').hexdigest(),
    )
    axis = calendar_module.resolve_calendar(
        (date(2025, 1, 2),), start=date(2025, 1, 1), end=date(2025, 1, 7), calendar=calendar,
    )
    assert axis.dates == (date(2025, 1, 2),)
    assert axis.file_sha256 is None
