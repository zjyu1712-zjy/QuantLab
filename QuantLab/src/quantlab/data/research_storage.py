"""只追加的本地研究快照：最后发布完成标记，重读不依赖原始行情文件。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import platform
import re
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pyarrow

from quantlab.data.alignment import ALIGNMENT_RULE_VERSION, AlignmentResult, summarize_alignment
from quantlab.data.calendar import CalendarAxis
from quantlab.data.research_sets import ResearchSetPreview, validate_research_preview


@dataclass(frozen=True)
class SavedResearchSet:
    set_id: str
    name: str
    created_at: str
    directory: Path


@dataclass(frozen=True)
class ResearchSetScan:
    records: tuple[SavedResearchSet, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True)
class LoadedResearchSet:
    record: SavedResearchSet
    preview: ResearchSetPreview


_MANIFEST_FIELDS = {
    "rule_version", "set_id", "name", "created_at", "start", "end", "adjustment", "sources",
    "calendar", "common_dates", "warnings", "alignment_warnings", "alignment_content_sha256",
    "preview_sha256", "files", "environment",
}
_CALENDAR_FIELDS = {
    "dates", "mode", "source", "coverage_start", "coverage_end", "file_sha256", "input_dates_sha256",
}
_OUTPUT_FILES = ("aligned.parquet", "coverage.csv")
_TEXT_COLUMNS = ("symbol", "version_id", "provider", "adjustment", "adjustment_basis")
_DATE_COLUMNS = ("first_observed", "last_observed")
_COUNT_COLUMNS = ("calendar_count", "observation_count", "missing_count", "longest_missing_run", "zero_volume_count")


def _set_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("研究集合 ID 必须为规范 UUID")
    try:
        canonical = str(UUID(value))
    except (ValueError, AttributeError) as error:
        raise ValueError("研究集合 ID 必须为规范 UUID") from error
    if value != canonical:
        raise ValueError("研究集合 ID 必须为规范 UUID")
    return value


def _name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("请输入非空的研究集合名称")
    return value.strip()


def _resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except (OSError, RuntimeError) as error:
        raise ValueError("研究集合路径无法解析，可能存在循环链接") from error


def _file_bytes(directory: Path, filename: str) -> bytes:
    """只读取固定文件名，同一字节快照用于指纹与解码；拒绝符号链接。"""
    path = directory / filename
    if path.is_symlink():
        raise ValueError(f"研究集合 {filename} 不能是符号链接")
    if not path.is_file():
        raise ValueError(f"研究集合未完成或缺少文件：{filename}")
    return path.read_bytes()


def _json_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"说明文件含重复字段：{key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"说明文件不接受非有限 JSON 数值：{value}")


def _fields(value: object, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label}字段不符合固定格式")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}必须为非空文本")
    return value


def _hash(value: object) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("说明文件中的指纹格式无效")


def _date(value: object) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ValueError("快照日期必须是 YYYY-MM-DD，不得包含时间")
    return date.fromisoformat(value)


def _date_list(values: object) -> tuple[date, ...]:
    if not isinstance(values, list):
        raise ValueError("日期清单必须是列表")
    return tuple(_date(value) for value in values)


def _warnings(values: object) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise ValueError("限制提示必须是列表")
    return tuple(_text(value, "限制提示") for value in values)


def _created_at(value: object) -> datetime:
    instant = datetime.fromisoformat(_text(value, "创建时间"))
    if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
        raise ValueError("创建时间必须包含 UTC 时区")
    return instant


def _manifest(preview: ResearchSetPreview, record: SavedResearchSet, files: dict[str, str]) -> dict:
    """显式字段白名单；不复制进程环境、交易数据或任意元数据字典。"""
    axis = preview.alignment.calendar
    calendar = asdict(axis)
    calendar.update(
        dates=[day.isoformat() for day in axis.dates],
        coverage_start=axis.coverage_start.isoformat(), coverage_end=axis.coverage_end.isoformat(),
    )
    return {
        "rule_version": ALIGNMENT_RULE_VERSION, "set_id": record.set_id, "name": record.name,
        "created_at": record.created_at, "start": preview.start.isoformat(), "end": preview.end.isoformat(),
        "adjustment": preview.adjustment, "sources": list(preview.sources), "calendar": calendar,
        "common_dates": [day.isoformat() for day in preview.alignment.common_dates],
        "warnings": list(preview.warnings), "alignment_warnings": list(preview.alignment.warnings),
        "alignment_content_sha256": preview.alignment.content_sha256, "preview_sha256": preview.preview_sha256,
        "files": files,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__, "pyarrow": pyarrow.__version__},
    }


def save_research_set(preview: ResearchSetPreview, output_dir: Path, *, name: str) -> SavedResearchSet:
    """保存隔离副本，UUID 碰撞直接失败；半成品保留但不发布、不自动清理。"""
    name = _name(name)
    snapshot = deepcopy(preview)
    validate_research_preview(snapshot)
    root = _resolve(Path(output_dir))
    set_id = _set_id(str(uuid4()))
    destination = root / set_id
    record = SavedResearchSet(set_id, name, datetime.now(timezone.utc).isoformat(), destination)
    # 名称只是显示文本，不参与目录定位；独占创建防止覆盖任何既有集合。
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "aligned.parquet").open("xb") as stream:
        snapshot.alignment.data.to_parquet(stream, index=False)
    with (destination / "coverage.csv").open("x", encoding="utf-8", newline="") as stream:
        snapshot.coverage.to_csv(stream, index=False, na_rep="", lineterminator="\n")
    files = {filename: sha256(_file_bytes(destination, filename)).hexdigest() for filename in _OUTPUT_FILES}
    serialized = json.dumps(_manifest(snapshot, record, files), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    temporary = destination / ".manifest.tmp"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    # 同一目录中最后一次原子改名，读者不会看到半份 manifest。
    temporary.replace(destination / "manifest.json")
    return record


def _read_manifest(directory: Path, set_id: str) -> tuple[dict, SavedResearchSet]:
    content = json.loads(_file_bytes(directory, "manifest.json"), object_pairs_hook=_json_pairs,
                         parse_constant=_reject_json_constant)
    _fields(content, _MANIFEST_FIELDS, "说明文件")
    if content["rule_version"] != ALIGNMENT_RULE_VERSION:
        raise ValueError("不支持该研究集合的规则版本")
    if content["set_id"] != set_id:
        raise ValueError("说明文件中的集合 ID 与目录不一致")
    name = _name(content["name"])
    if name != content["name"]:
        raise ValueError("保存的集合名称不能含首尾空白")
    _created_at(content["created_at"])
    _fields(content["files"], set(_OUTPUT_FILES), "输出文件")
    for value in content["files"].values():
        _hash(value)
    _fields(content["environment"], {"python", "pandas", "numpy", "pyarrow"}, "运行版本")
    for value in content["environment"].values():
        _text(value, "运行版本")
    for key in ("alignment_content_sha256", "preview_sha256"):
        _hash(content[key])
    return content, SavedResearchSet(set_id, name, content["created_at"], directory)


def _coverage_from_csv(payload: bytes) -> pd.DataFrame:
    """代码/版本/来源保持文本；空观测日期恢复 None；比例使用 round_trip 精确解析。"""
    dtypes = {column: str for column in (*_TEXT_COLUMNS, *_DATE_COLUMNS)}
    dtypes.update({column: "int64" for column in _COUNT_COLUMNS})
    dtypes["missing_ratio"] = "float64"
    coverage = pd.read_csv(BytesIO(payload), dtype=dtypes, keep_default_na=False,
                           skip_blank_lines=False, float_precision="round_trip")
    if not set(_DATE_COLUMNS).issubset(coverage.columns):
        raise ValueError("覆盖报告缺少观测日期字段")
    for column in _DATE_COLUMNS:
        coverage[column] = pd.Series(
            [None if value == "" else _date(value) for value in coverage[column]], dtype=object,
        )
    return coverage


def _preview(content: dict, payloads: dict[str, bytes]) -> ResearchSetPreview:
    calendar = content["calendar"]
    _fields(calendar, _CALENDAR_FIELDS, "日历")
    axis = CalendarAxis(
        dates=_date_list(calendar["dates"]), mode=_text(calendar["mode"], "日历模式"),
        source=_text(calendar["source"], "日历来源"), coverage_start=_date(calendar["coverage_start"]),
        coverage_end=_date(calendar["coverage_end"]), file_sha256=calendar["file_sha256"],
        input_dates_sha256=calendar["input_dates_sha256"],
    )
    if not isinstance(content["sources"], list):
        raise ValueError("来源清单必须是列表")
    aligned = pd.read_parquet(BytesIO(payloads["aligned.parquet"]))
    core, _ = summarize_alignment(aligned)
    result = ResearchSetPreview(
        alignment=AlignmentResult(
            data=aligned, coverage=core, calendar=axis, common_dates=_date_list(content["common_dates"]),
            warnings=_warnings(content["alignment_warnings"]), content_sha256=content["alignment_content_sha256"],
        ),
        sources=tuple(content["sources"]), start=_date(content["start"]), end=_date(content["end"]),
        adjustment=content["adjustment"], coverage=_coverage_from_csv(payloads["coverage.csv"]),
        warnings=_warnings(content["warnings"]), preview_sha256=content["preview_sha256"],
    )
    validate_research_preview(result)
    return result


def load_research_set(output_dir: Path, set_id: str) -> LoadedResearchSet:
    """只读固定目录内三个固定文件；字节指纹和内容语义都通过才返回。

    内容指纹用于发现不一致，不是数字签名或行情真实性认证；不访问来源路径，
    不恢复旧运行环境，也不自行修复已保存数据。
    """
    set_id = _set_id(set_id)
    root = _resolve(Path(output_dir))
    directory = root / set_id
    if directory.is_symlink() or _resolve(directory) != directory:
        raise ValueError("研究集合目录包含符号链接或越界路径")
    try:
        content, record = _read_manifest(directory, set_id)
        payloads = {filename: _file_bytes(directory, filename) for filename in _OUTPUT_FILES}
        for filename, payload in payloads.items():
            if sha256(payload).hexdigest() != content["files"][filename]:
                raise ValueError(f"{filename} 文件指纹不一致，快照可能已经损坏")
        return LoadedResearchSet(record, _preview(content, payloads))
    except (OSError, ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"研究集合 {set_id} 无法载入：{error}") from error


def list_research_sets(output_dir: Path) -> ResearchSetScan:
    """忽略未发布目录，隔离损坏集合；只返回经重读校验的记录，按 UTC 时间倒序。"""
    root = _resolve(Path(output_dir))
    if not root.exists():
        return ResearchSetScan((), ())
    if not root.is_dir():
        raise ValueError("研究集合根路径必须是目录")
    records, issues = [], []
    for directory in sorted(root.iterdir()):
        if directory.is_symlink():
            issues.append(f"研究集合 {directory.name} 不可用：目录不能是符号链接")
            continue
        if not directory.is_dir():
            continue
        marker = directory / "manifest.json"
        if not marker.exists() and not marker.is_symlink():
            continue
        try:
            records.append(load_research_set(root, directory.name).record)
        except (OSError, ValueError) as error:
            issues.append(f"研究集合 {directory.name} 不可用：{error}")
    records.sort(key=lambda record: record.set_id)
    records.sort(key=lambda record: _created_at(record.created_at), reverse=True)
    return ResearchSetScan(tuple(records), tuple(issues))
