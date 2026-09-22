"""纯内存多 ETF 对齐：缺失不填充，观测不等于可交易，不修改输入表。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from hashlib import sha256
from itertools import product
import json
from numbers import Real
import re

import numpy as np
import pandas as pd

from quantlab.data.calendar import CalendarAxis, CalendarInput, normalize_daily_dates, resolve_calendar
from quantlab.data.validator import REQUIRED_MARKET_COLUMNS, validate_market_columns, validate_market_data


ALIGNMENT_RULE_VERSION = "multi-etf-alignment-v1"
NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume")
ALIGNED_COLUMNS = (*REQUIRED_MARKET_COLUMNS, "has_observation")
COVERAGE_COLUMNS = (
    "symbol", "first_observed", "last_observed", "calendar_count", "observation_count",
    "missing_count", "missing_ratio", "longest_missing_run", "zero_volume_count",
)


@dataclass(frozen=True)
class AlignmentResult:
    data: pd.DataFrame
    coverage: pd.DataFrame
    calendar: CalendarAxis
    common_dates: tuple[date, ...]
    warnings: tuple[str, ...]
    content_sha256: str


def _check_symbol(symbol: str) -> None:
    if not isinstance(symbol, str) or not re.fullmatch(r"[0-9]{6}", symbol):
        raise ValueError("ETF 代码必须为六位数字字符串")


def normalize_market_frame(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """校验完整历史，复制并转为固定七列；数值统一 float64，不混用复权列。"""
    _check_symbol(symbol)
    try:
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("行情表不能为空")
        if not data.columns.is_unique:
            raise ValueError("行情表存在重复列名")
        validate_market_columns(data)
        frame = data.loc[:, list(REQUIRED_MARKET_COLUMNS)].copy(deep=True)
        if not frame.symbol.map(lambda value: isinstance(value, str) and value == symbol).all():
            raise ValueError("行情代码与所选代码不一致")
        frame["date"] = normalize_daily_dates(frame.date)
        for column in NUMERIC_COLUMNS:
            # 防止 True 被当成 1，或时间戳/复数被隐式转为行情数值。
            valid_types = frame[column].map(
                lambda value: isinstance(value, (Real, Decimal, str))
                and not isinstance(value, (bool, np.bool_))
            )
            if not valid_types.all():
                raise ValueError(f"{column} 必须为实数或数字文本")
            numeric = pd.to_numeric(frame[column], errors="raise").astype("float64")
            if not np.isfinite(numeric.to_numpy()).all():
                raise ValueError("行情数值存在空值或无穷值")
            for original, converted in zip(frame[column], numeric):
                exact = Decimal(str(original).strip())
                if exact == exact.to_integral_value() and Decimal.from_float(float(converted)) != exact:
                    raise ValueError(f"{column} 的整数数值无法由 float64 精确表示，拒绝损失精度")
            frame[column] = numeric
        validate_market_data(frame)
        return frame.sort_values("date").reset_index(drop=True)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"ETF {symbol} 行情校验失败：{error}") from error


def _validate_axis(axis: CalendarAxis) -> None:
    """重验可被程序直接构造的日期轴；裁剪后的 supplied 轴不冒称可重验原文件。"""
    if not isinstance(axis, CalendarAxis):
        raise ValueError("日期轴必须为 CalendarAxis")
    expected = resolve_calendar(axis.dates, start=axis.coverage_start, end=axis.coverage_end)
    if expected.dates != axis.dates:
        raise ValueError("日期轴必须升序、不重复且在声明覆盖区间内")
    if not isinstance(axis.source, str) or not axis.source.strip():
        raise ValueError("日期轴来源不能为空")
    if axis.mode == "observed_union":
        if axis != expected:
            raise ValueError("行情日期并集的声明或指纹不一致")
    elif axis.mode == "supplied":
        hashes = (axis.input_dates_sha256,)
        if axis.file_sha256 is not None:
            hashes += (axis.file_sha256,)
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
            raise ValueError("日期轴指纹格式无效")
    else:
        raise ValueError("日期轴模式无效")


def _table_dimensions(data: pd.DataFrame) -> tuple[tuple[date, ...], tuple[str, ...]]:
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("对齐表不能为空")
    if tuple(data.columns) != ALIGNED_COLUMNS:
        raise ValueError("对齐表必须按规范顺序包含八个固定字段")
    if not pd.api.types.is_datetime64_any_dtype(data.date.dtype):
        raise ValueError("对齐表日期必须为 pandas 日期类型")
    normalized = normalize_daily_dates(data.date)
    for symbol in data.symbol:
        _check_symbol(symbol)
    return tuple(sorted(set(normalized.dt.date))), tuple(sorted(set(data.symbol)))


def _validate_table(data: pd.DataFrame, dates: tuple[date, ...], symbols: tuple[str, ...]) -> None:
    """同时用于结果验证及摘要重算；不需要原始文件或目录模块。"""
    _table_dimensions(data)
    if not isinstance(symbols, tuple) or len(symbols) < 2:
        raise ValueError("对齐表至少需要两只 ETF")
    for symbol in symbols:
        _check_symbol(symbol)
    if tuple(sorted(set(symbols))) != symbols:
        raise ValueError("ETF 代码必须升序且唯一")
    actual_keys = list(zip(data.date.dt.date, data.symbol))
    if actual_keys != list(product(dates, symbols)):
        raise ValueError("对齐表必须是有序、无重复的完整日期与代码组合")
    if not pd.api.types.is_bool_dtype(data.has_observation.dtype) or data.has_observation.isna().any():
        raise ValueError("观测标记必须是无空值的布尔列")
    for column in NUMERIC_COLUMNS:
        if data[column].dtype != np.dtype("float64"):
            raise ValueError(f"对齐表 {column} 必须保持规范 float64 类型")
    observed = data.loc[data.has_observation]
    if observed.empty:
        raise ValueError("研究区间内所有 ETF 均无行情")
    missing = data.loc[~data.has_observation, list(NUMERIC_COLUMNS)]
    if not missing.isna().all().all():
        raise ValueError("缺失行的行情字段必须全部为空，不能填充或补零")
    values = observed[list(NUMERIC_COLUMNS)]
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype="float64")).all():
        raise ValueError("真实观测行含空值或无穷值")
    validate_market_data(observed)


def validate_aligned_data(data: pd.DataFrame, axis: CalendarAxis, symbols: tuple[str, ...]) -> None:
    """验证规范长表与日期轴的语义，用于后续保存前及重读后的复核。"""
    _validate_axis(axis)
    _validate_table(data, axis.dates, symbols)
    if axis.mode == "observed_union" and not data.groupby("date").has_observation.any().all():
        raise ValueError("行情日期并集不能包含所有 ETF 均无观测的日期")


def summarize_alignment(data: pd.DataFrame) -> tuple[pd.DataFrame, tuple[date, ...]]:
    """按日期轴位置统计缺失，不把自然日间隔、零成交量或首末记录推断为停牌。"""
    dates, symbols = _table_dimensions(data)
    _validate_table(data, dates, symbols)
    rows = []
    for symbol in symbols:
        group = data.loc[data.symbol.eq(symbol)]
        observed = group.loc[group.has_observation]
        run = longest = 0
        for present in group.has_observation:
            run = 0 if present else run + 1
            longest = max(longest, run)
        missing_count = len(group) - len(observed)
        rows.append({
            "symbol": symbol,
            "first_observed": observed.date.iloc[0].date() if len(observed) else None,
            "last_observed": observed.date.iloc[-1].date() if len(observed) else None,
            "calendar_count": len(group), "observation_count": len(observed),
            "missing_count": missing_count, "missing_ratio": missing_count / len(group),
            "longest_missing_run": longest, "zero_volume_count": int(observed.volume.eq(0).sum()),
        })
    common = data.groupby("date", sort=True).has_observation.all()
    common_dates = tuple(timestamp.date() for timestamp in common.index[common])
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS), common_dates


def alignment_content_sha256(data: pd.DataFrame, axis: CalendarAxis) -> str:
    """计算规范内容指纹；保留浮点精度，缺失写 null，排除原文件排版差异。"""
    _, symbols = _table_dimensions(data)
    validate_aligned_data(data, axis, symbols)
    calendar = {
        "dates": [day.isoformat() for day in axis.dates], "mode": axis.mode, "source": axis.source,
        "coverage_start": axis.coverage_start.isoformat(), "coverage_end": axis.coverage_end.isoformat(),
        "input_dates_sha256": axis.input_dates_sha256,
    }
    rows = [
        [day.date().isoformat(), symbol,
         *([float(value) for value in values[:5]] if bool(values[5]) else [None] * 5), bool(values[5])]
        for day, symbol, *values in data.itertuples(index=False, name=None)
    ]
    payload = json.dumps(
        {"rule_version": ALIGNMENT_RULE_VERSION, "calendar": calendar, "rows": rows},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def alignment_warnings(
    coverage: pd.DataFrame, axis: CalendarAxis, common_dates: tuple[date, ...],
) -> tuple[str, ...]:
    """从已校验的摘要生成限制提示，供对齐与预览复核共用，避免两套规则。"""
    warnings = [
        "基于行情日期并集，未经完整交易日历核验，无法发现所有 ETF 同时缺数据的日期；缺失比例仅相对于当前日期轴。"
        if axis.mode == "observed_union" else "日历由用户提供，覆盖范围为用户声明，不代表已经通过官方认证。"
    ]
    for row in coverage.itertuples(index=False):
        if row.observation_count == 0:
            warnings.append(f"ETF {row.symbol} 在研究区间内无行情，保留全缺失记录，不推断上市或停牌状态。")
        elif row.missing_count:
            warnings.append(f"ETF {row.symbol} 相对于当前日期轴缺失 {row.missing_count} 条记录，未填充行情。")
        if row.zero_volume_count:
            warnings.append(f"ETF {row.symbol} 有 {row.zero_volume_count} 条零成交量观测；有记录不代表可交易。")
    if not common_dates:
        warnings.append("没有所有 ETF 同时有行情的共同日期，不能进行全体 ETF 的同日比较。")
    return tuple(warnings)


def align_market_data(
    datasets: Mapping[str, pd.DataFrame], *, start: date, end: date,
    calendar: CalendarInput | None = None,
) -> AlignmentResult:
    """先校验完整输入，再裁剪、建立完整网格；不计算收益或替代交易状态。"""
    if not isinstance(datasets, Mapping) or len(datasets) < 2:
        raise ValueError("至少需要两只不同 ETF 的行情")
    normalized = {symbol: normalize_market_frame(frame, symbol) for symbol, frame in datasets.items()}
    symbols = tuple(sorted(normalized))
    observed_dates = tuple(day for symbol in symbols for day in normalized[symbol].date.dt.date)
    axis = resolve_calendar(observed_dates, start=start, end=end, calendar=calendar)
    clipped = [normalized[s].loc[normalized[s].date.dt.date.between(start, end)] for s in symbols]
    if all(frame.empty for frame in clipped):
        raise ValueError("研究区间内所有 ETF 均无行情")
    grid = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(axis.dates, dtype="datetime64[ns]"), symbols], names=["date", "symbol"],
    )
    observed = pd.concat(clipped, ignore_index=True).set_index(["date", "symbol"])
    observed["has_observation"] = True
    aligned = observed.reindex(grid)
    aligned["has_observation"] = aligned.has_observation.eq(True)
    aligned = aligned.reset_index().loc[:, list(ALIGNED_COLUMNS)]
    validate_aligned_data(aligned, axis, symbols)
    coverage, common_dates = summarize_alignment(aligned)
    return AlignmentResult(
        data=aligned, coverage=coverage, calendar=axis, common_dates=common_dates,
        warnings=alignment_warnings(coverage, axis, common_dates),
        content_sha256=alignment_content_sha256(aligned, axis),
    )
