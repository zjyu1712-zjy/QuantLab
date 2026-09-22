"""固定版本的多 ETF 研究预览：只读源文件，不保存、不计算因子或收益。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
from hashlib import sha256
import json
from pathlib import Path
import re

import pandas as pd

from quantlab.data.alignment import (
    ALIGNMENT_RULE_VERSION, AlignmentResult, align_market_data, alignment_content_sha256,
    alignment_warnings, normalize_market_frame, summarize_alignment, validate_aligned_data,
)
from quantlab.data.calendar import CalendarInput
from quantlab.data.catalog import MarketVersionRecord, read_market_version


SOURCE_FIELDS = (
    "symbol", "version_id", "provider", "imported_at", "processed_path", "file_sha256",
    "metadata_sha256", "input_content_sha256", "adjustment", "adjustment_basis",
)
REPORT_SOURCE_FIELDS = ("symbol", "version_id", "provider", "adjustment", "adjustment_basis")
ADJUSTMENTS = ("none", "qfq", "hfq")


@dataclass(frozen=True)
class DatasetSelection:
    record: MarketVersionRecord
    declared_adjustment: str | None = None


@dataclass(frozen=True)
class ResearchSetPreview:
    alignment: AlignmentResult
    sources: tuple[dict, ...]
    start: date
    end: date
    adjustment: str
    coverage: pd.DataFrame
    warnings: tuple[str, ...]
    preview_sha256: str


def _range(start: date, end: date) -> None:
    if type(start) is not date or type(end) is not date or start > end:
        raise ValueError("研究区间必须为有效日期，且开始日期不晚于结束日期")


def _adjustment(value: object) -> None:
    if not isinstance(value, str) or value not in ADJUSTMENTS:
        raise ValueError("请明确确认复权口径为 none、qfq 或 hfq")


def _digest(payload: object) -> str:
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(content.encode("utf-8")).hexdigest()


def _input_content_sha256(frame: pd.DataFrame) -> str:
    """完整、已规范化的七列历史指纹，不仅是研究区间；不含机器路径。"""
    rows = [[day.date().isoformat(), symbol, *map(float, values)]
            for day, symbol, *values in frame.itertuples(index=False, name=None)]
    return _digest({"rule_version": ALIGNMENT_RULE_VERSION, "columns": list(frame.columns), "rows": rows})


def _checked_selections(selections: tuple[DatasetSelection, ...]) -> tuple[tuple[DatasetSelection, str], ...]:
    """先核对全部选择和声明，再允许打开文件，避免读取一半才发现口径冲突。"""
    if not isinstance(selections, tuple) or len(selections) < 2:
        raise ValueError("请至少选择两只不同 ETF 的固定版本")
    checked = []
    symbols = set()
    for selection in selections:
        if not isinstance(selection, DatasetSelection) or not isinstance(selection.record, MarketVersionRecord):
            raise ValueError("选择项必须包含有效的行情版本记录")
        record = selection.record
        if not isinstance(record.symbol, str) or not re.fullmatch(r"[0-9]{6}", record.symbol):
            raise ValueError("ETF 代码必须为六位数字字符串")
        if record.symbol in symbols:
            raise ValueError(f"ETF {record.symbol} 重复选择；每只 ETF 只能选择一个版本")
        symbols.add(record.symbol)
        declared = selection.declared_adjustment
        if declared is not None:
            _adjustment(declared)
        if record.adjustment is not None:
            _adjustment(record.adjustment)
            if declared is not None and declared != record.adjustment:
                raise ValueError(f"ETF {record.symbol} 不能覆盖数据源记录的复权口径")
        adjustment = record.adjustment if record.adjustment is not None else declared
        _adjustment(adjustment)
        checked.append((selection, adjustment))
    if len({adjustment for _, adjustment in checked}) != 1:
        raise ValueError("所有 ETF 的复权口径必须一致，不能混合不同口径")
    return tuple(sorted(checked, key=lambda item: item[0].record.symbol))


def _coverage(core: pd.DataFrame, sources: tuple[dict, ...]) -> pd.DataFrame:
    provenance = pd.DataFrame(sources).loc[:, list(REPORT_SOURCE_FIELDS)]
    return core.merge(provenance, on="symbol", how="left", validate="one_to_one", sort=False)


def _warnings(aligned_warnings: tuple[str, ...], sources: tuple[dict, ...]) -> tuple[str, ...]:
    warnings = list(aligned_warnings)
    for source in sources:
        if source["adjustment_basis"] == "user_declared":
            warnings.append(
                f"ETF {source['symbol']} 的复权口径 {source['adjustment']} 为用户声明，未由数据源验证；本步骤不计算复权。"
            )
    if len({source["provider"] for source in sources}) > 1:
        warnings.append("集合包含不同数据来源：相同复权标签不保证数据商的计算方法完全相同；未转换成交量单位。")
    return tuple(warnings)


def build_research_set(
    selections: tuple[DatasetSelection, ...], *, start: date, end: date,
    calendar: CalendarInput | None = None,
) -> ResearchSetPreview:
    """只载入指定版本；保留同一文件字节指纹和完整规范行情指纹，不查找最新版本。"""
    _range(start, end)
    checked = _checked_selections(selections)
    datasets, sources = {}, []
    for selection, adjustment in checked:
        record = selection.record
        frame, file_hash = read_market_version(record)
        normalized = normalize_market_frame(frame, record.symbol)
        datasets[record.symbol] = normalized
        sources.append({
            "symbol": record.symbol, "version_id": record.version_id, "provider": record.provider,
            "imported_at": record.imported_at, "processed_path": str(record.processed_path),
            "file_sha256": file_hash, "metadata_sha256": record.metadata_sha256,
            "input_content_sha256": _input_content_sha256(normalized), "adjustment": adjustment,
            "adjustment_basis": "provider" if record.adjustment is not None else "user_declared",
        })
    aligned = align_market_data(datasets, start=start, end=end, calendar=calendar)
    source_tuple = tuple(sources)
    result = ResearchSetPreview(
        alignment=aligned, sources=source_tuple, start=start, end=end, adjustment=checked[0][1],
        coverage=_coverage(aligned.coverage, source_tuple), warnings=_warnings(aligned.warnings, source_tuple),
        preview_sha256="",
    )
    return replace(result, preview_sha256=research_preview_sha256(result))


def _check_sources(sources: tuple[dict, ...], adjustment: str) -> tuple[str, ...]:
    if not isinstance(sources, tuple) or len(sources) < 2:
        raise ValueError("来源清单至少应包含两只 ETF")
    symbols = []
    for source in sources:
        if not isinstance(source, dict) or set(source) != set(SOURCE_FIELDS):
            raise ValueError("来源清单字段不符合白名单")
        if any(not isinstance(value, str) or not value.strip() for value in source.values()):
            raise ValueError("来源字段必须为非空字符串")
        if not re.fullmatch(r"[0-9]{6}", source["symbol"]):
            raise ValueError("来源 ETF 代码必须为六位数字字符串")
        if source["adjustment"] != adjustment or source["adjustment_basis"] not in ("provider", "user_declared"):
            raise ValueError("来源复权口径或声明依据不一致")
        for field in ("file_sha256", "metadata_sha256", "input_content_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", source[field]):
                raise ValueError(f"来源 {field} 指纹格式无效")
        imported_at = datetime.fromisoformat(source["imported_at"])
        if imported_at.tzinfo is None or imported_at.utcoffset() is None:
            raise ValueError("来源导入时间必须包含时区")
        # 仅验证路径声明的形状，不 resolve、不读取，避免重验时依赖或跟随来源文件。
        if not Path(source["processed_path"]).is_absolute():
            raise ValueError("来源文件路径必须为绝对路径")
        symbols.append(source["symbol"])
    if symbols != sorted(set(symbols)):
        raise ValueError("来源代码必须升序且每只 ETF 只选一个版本")
    return tuple(symbols)


def _same_table(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    if not isinstance(actual, pd.DataFrame):
        raise ValueError(f"{label} 必须为数据表")
    try:
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    except AssertionError as error:
        raise ValueError(f"{label} 与对齐行情及来源重算结果不一致") from error


def research_preview_sha256(preview: ResearchSetPreview) -> str:
    """重验内容并计算预览指纹，不改动对象；源文件字节/完整历史指纹仅作保留声明。

    本函数不会重新打开原始文件，无法独立证明其历史真实性。路径、保存 ID、
    保存时间不参与指纹；文件路径变化不改变研究数据内容身份。日历原文件
    指纹在此绑定，保持对齐层的语义指纹不受 CSV 排版差异影响。
    """
    if not isinstance(preview, ResearchSetPreview) or not isinstance(preview.alignment, AlignmentResult):
        raise ValueError("研究预览对象无效")
    _range(preview.start, preview.end)
    _adjustment(preview.adjustment)
    symbols = _check_sources(preview.sources, preview.adjustment)
    aligned = preview.alignment
    validate_aligned_data(aligned.data, aligned.calendar, symbols)
    axis = aligned.calendar
    if (axis.coverage_start > preview.start or axis.coverage_end < preview.end
            or any(day < preview.start or day > preview.end for day in axis.dates)):
        raise ValueError("日期轴与研究区间或日历覆盖声明不一致")
    if axis.mode == "observed_union" and (axis.coverage_start, axis.coverage_end) != (preview.start, preview.end):
        raise ValueError("行情并集的覆盖声明必须等于研究区间")
    actual_hash = alignment_content_sha256(aligned.data, axis)
    if actual_hash != aligned.content_sha256:
        raise ValueError("对齐内容指纹不一致，预览行情可能已被修改")
    core, common = summarize_alignment(aligned.data)
    _same_table(aligned.coverage, core, "核心覆盖统计")
    _same_table(preview.coverage, _coverage(core, preview.sources), "研究覆盖报告")
    if (not isinstance(aligned.common_dates, tuple)
            or any(type(day) is not date for day in aligned.common_dates) or aligned.common_dates != common):
        raise ValueError("共同日期与行情重算结果不一致")
    expected_warnings = alignment_warnings(core, axis, common)
    if aligned.warnings != expected_warnings or preview.warnings != _warnings(expected_warnings, preview.sources):
        raise ValueError("限制提示与当前数据及声明不一致")
    return _digest({
        "alignment_content_sha256": actual_hash,
        "calendar_file_sha256": axis.file_sha256,
        "start": preview.start.isoformat(), "end": preview.end.isoformat(), "adjustment": preview.adjustment,
        "sources": [{key: source[key] for key in SOURCE_FIELDS if key != "processed_path"}
                    for source in preview.sources],
    })


def validate_research_preview(preview: ResearchSetPreview) -> None:
    """保存前及未来重读时复核可变表/字典，与既有指纹比较，绝不重新赋值修复。"""
    if research_preview_sha256(preview) != preview.preview_sha256:
        raise ValueError("研究预览指纹不一致，请重新生成预览")
