"""研究日期轴：区分观测日期并集与用户提供的日历，不猜测交易日。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from io import StringIO
import json
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalendarInput:
    """已规范化的日历及覆盖声明；文件指纹不代表来源经过官方认证。"""

    dates: tuple[date, ...]
    source: str
    coverage_start: date
    coverage_end: date
    file_sha256: str | None
    dates_sha256: str


@dataclass(frozen=True)
class CalendarAxis:
    """本次实际使用的日期，以及生成日期轴所依据的来源信息。"""

    dates: tuple[date, ...]
    mode: str
    source: str
    coverage_start: date
    coverage_end: date
    file_sha256: str | None
    input_dates_sha256: str


_DATE_TEXT = re.compile(
    r"(?:[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{4}/[0-9]{2}/[0-9]{2}|[0-9]{8})"
    r"(?:[ T][0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,9})?)?"
)


def normalize_daily_dates(values: pd.Series) -> pd.Series:
    """返回无时区的午夜日期列，保留索引、名称、顺序和重复项。

    文本接受 YYYY-MM-DD、YYYY/MM/DD、YYYYMMDD，可附完整午夜时间。
    也接受 date、无时区午夜 datetime/Timestamp、numpy.datetime64。
    数值、相对日期、盘中时间和时区值拒绝，不截断或转换后冒充日线。
    """
    if not isinstance(values, pd.Series):
        raise ValueError("日线日期必须使用 pandas Series")
    normalized: list[pd.Timestamp] = []
    for position, value in enumerate(values, start=1):
        label = f"第 {position} 行日期"
        if not isinstance(value, (str, date, np.datetime64)) or pd.isna(value):
            raise ValueError(f"{label}为空或类型无效，不接受数值日期")
        if isinstance(value, str):
            value = value.strip()
            if not _DATE_TEXT.fullmatch(value):
                raise ValueError(f"{label}必须是完整的年-月-日，不接受时区或相对日期")
        try:
            timestamp = pd.Timestamp(value)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"{label}无效") from error
        if pd.isna(timestamp):
            raise ValueError(f"{label}不能为空")
        if timestamp.tzinfo is not None:
            raise ValueError(f"{label}不能带时区")
        if timestamp != timestamp.normalize():
            raise ValueError(f"{label}含非午夜时间，不能作为日线日期")
        normalized.append(timestamp)
    try:
        return pd.Series(normalized, index=values.index, name=values.name, dtype="datetime64[ns]")
    except (ValueError, OverflowError) as error:
        raise ValueError("日期超出 pandas 纳秒日期列可表示的范围") from error


def _date(value: object, label: str) -> None:
    # datetime 是 date 的子类，但边界和规范日期必须是纯 date，不隐式截断。
    if type(value) is not date:
        raise ValueError(f"{label}必须为不含时间的 date 日期")


def _range(start: date, end: date, label: str) -> None:
    _date(start, f"{label}开始日期")
    _date(end, f"{label}结束日期")
    if start > end:
        raise ValueError(f"{label}开始日期不能晚于结束日期")


def _source(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("日历来源不能为空")
    return value.strip()


def _dates_sha256(dates: tuple[date, ...]) -> str:
    """规范指纹：有序 ISO 日期列表、UTF-8 JSON、无额外空格。"""
    payload = json.dumps([day.isoformat() for day in dates], separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _validate_calendar(calendar: CalendarInput) -> None:
    """dataclass 只是容器；程序化构造的日历也必须通过同一组检查。"""
    if not isinstance(calendar, CalendarInput):
        raise ValueError("日历必须是 CalendarInput")
    _source(calendar.source)
    _range(calendar.coverage_start, calendar.coverage_end, "日历覆盖区间")
    if not isinstance(calendar.dates, tuple) or not calendar.dates:
        raise ValueError("日历日期必须为非空元组")
    for day in calendar.dates:
        _date(day, "日历日期")
    if calendar.dates != tuple(sorted(set(calendar.dates))):
        raise ValueError("日历日期必须升序且不重复")
    if calendar.dates[0] < calendar.coverage_start or calendar.dates[-1] > calendar.coverage_end:
        raise ValueError("日历日期超出声明的覆盖范围")
    if calendar.dates_sha256 != _dates_sha256(calendar.dates):
        raise ValueError("日历日期指纹与日期内容不一致")
    if calendar.file_sha256 is not None and (
        not isinstance(calendar.file_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", calendar.file_sha256)
    ):
        raise ValueError("日历文件指纹格式无效")


def parse_calendar_csv(
    content: bytes, *, source: str, coverage_start: date, coverage_end: date,
) -> CalendarInput:
    """解析 UTF-8（含 BOM）单列日历；先校验再排序，不静默删除重复或空行。"""
    source = _source(source)
    _range(coverage_start, coverage_end, "日历覆盖区间")
    if not isinstance(content, bytes) or not content:
        raise ValueError("日历 CSV 内容必须为非空字节")
    try:
        reader = csv.reader(StringIO(content.decode("utf-8-sig"), newline=""), strict=True)
        if next(reader, None) != ["date"]:
            raise ValueError("日历 CSV 必须且只能包含一列 date")
        rows: list[str] = []
        for row_number, row in enumerate(reader, start=2):
            if len(row) != 1 or not row[0].strip():
                raise ValueError(f"日历 CSV 第 {row_number} 行必须有且只有一个非空日期")
            rows.append(row[0])
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError("日历文件必须是格式正确的 UTF-8 CSV") from error
    normalized = normalize_daily_dates(pd.Series(rows, dtype=object))
    if normalized.empty:
        raise ValueError("日历日期不能为空")
    if normalized.duplicated().any():
        raise ValueError("日历包含规范化后重复的日期，请检查原文件")
    dates = tuple(sorted(normalized.dt.date))
    calendar = CalendarInput(
        dates=dates, source=source, coverage_start=coverage_start, coverage_end=coverage_end,
        file_sha256=sha256(content).hexdigest(), dates_sha256=_dates_sha256(dates),
    )
    _validate_calendar(calendar)
    return calendar


def resolve_calendar(
    observed_dates: tuple[date, ...], *, start: date, end: date,
    calendar: CalendarInput | None = None,
) -> CalendarAxis:
    """生成含端点的研究日期轴；只用观测并集或显式日历，不推测工作日。"""
    _range(start, end, "研究区间")
    if not isinstance(observed_dates, tuple):
        raise ValueError("观测日期必须为 date 元组")
    for day in observed_dates:
        _date(day, "观测日期")
    observed = tuple(sorted({day for day in observed_dates if start <= day <= end}))
    if calendar is None:
        if not observed:
            raise ValueError("研究区间内没有观测日期，无法生成日期并集")
        return CalendarAxis(
            dates=observed, mode="observed_union",
            source="基于所选行情的日期并集，未经完整交易日历核验",
            coverage_start=start, coverage_end=end, file_sha256=None,
            input_dates_sha256=_dates_sha256(observed),
        )

    _validate_calendar(calendar)
    if start < calendar.coverage_start or end > calendar.coverage_end:
        raise ValueError("所提供日历的声明覆盖范围不足以包含研究区间")
    dates = tuple(day for day in calendar.dates if start <= day <= end)
    if not dates:
        raise ValueError("研究区间内没有日历日期")
    conflicts = sorted(set(observed) - set(dates))
    if conflicts:
        detail = ", ".join(day.isoformat() for day in conflicts)
        raise ValueError(f"行情日期不在所提供日历内：{detail}")
    return CalendarAxis(
        dates=dates, mode="supplied", source=_source(calendar.source),
        coverage_start=calendar.coverage_start, coverage_end=calendar.coverage_end,
        file_sha256=calendar.file_sha256, input_dates_sha256=calendar.dates_sha256,
    )
