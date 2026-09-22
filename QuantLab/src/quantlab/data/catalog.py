"""只读行情版本目录：选定具体版本，不替用户猜测最新版本或复权口径。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class MarketVersionRecord:
    """一份已完成导入、身份明确的行情版本。None 表示复权口径未知。"""

    symbol: str
    version_id: str
    row_count: int
    imported_at: str
    processed_path: Path
    raw_version_dir: Path
    provider: str
    adjustment: str | None
    metadata_sha256: str


@dataclass(frozen=True)
class CatalogScan:
    """健康版本与被隔离版本的原因分别返回，单份损坏不阻断整个目录。"""

    records: tuple[MarketVersionRecord, ...]
    issues: tuple[str, ...]


def _resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except RuntimeError as error:
        # Python 3.12 对循环符号链接抛 RuntimeError，统一为可展示的校验错误。
        raise ValueError("路径含循环符号链接，无法确定数据位置") from error


def _inside(path: Path, root: Path) -> Path:
    resolved = _resolve(path)
    if not resolved.is_relative_to(root):
        raise ValueError("路径越出允许的数据目录")
    return resolved


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须为非空字符串")
    return value


def _row_count(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("row_count 必须为正整数")
    return value


def _import_time(value: object) -> datetime:
    result = datetime.fromisoformat(_text(value, "导入时间"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("导入时间必须包含时区")
    return result


def _matching_path(value: object, expected: Path) -> None:
    # 兼容旧导入器以项目工作目录为基准保存的相对路径。
    if _resolve(Path(_text(value, "processed_path"))) != expected:
        raise ValueError("元数据路径与所选行情文件不一致")


def _metadata(directory: Path) -> tuple[dict[str, dict], str]:
    """只读取固定的元数据文件；不跟随 JSON 中的原始文件或其他任意路径。"""
    documents: dict[str, dict] = {}
    fingerprint = sha256()
    for name in sorted((
        "import_result.json", "metadata.json", "provider_metadata.json", "source_format.json",
    )):
        path = directory / name
        if not path.exists() and not path.is_symlink():
            continue
        safe_path = _inside(path, directory)
        payload = safe_path.read_bytes()
        document = json.loads(payload)
        if not isinstance(document, dict):
            raise ValueError(f"{name} 必须是 JSON 对象")
        documents[name] = document
        fingerprint.update(name.encode("utf-8") + b"\0" + sha256(payload).digest())
    return documents, fingerprint.hexdigest()


def _record(parquet_path: Path, directory: Path) -> MarketVersionRecord:
    """扫描和读取共用身份校验，避免页面显示的口径与实际读取时不同。"""
    documents, fingerprint = _metadata(directory)
    if "metadata.json" in documents and "provider_metadata.json" in documents:
        raise ValueError("同时存在 CSV 与 Provider 元数据，来源身份不明确")
    if "import_result.json" not in documents:
        raise ValueError("缺少导入完成标记 import_result.json")
    commit = documents["import_result.json"]
    version = commit.get("version")
    version_id = parquet_path.stem
    if not isinstance(version, dict) or version.get("version_id") != version_id:
        raise ValueError("提交标记版本号与行情文件不一致")
    if directory.name != version_id:
        raise ValueError("原始数据目录与行情版本不一致")
    symbol = _text(commit.get("symbol"), "symbol")
    if len(symbol) != 6 or not symbol.isascii() or not symbol.isdigit():
        raise ValueError("symbol 必须为六位数字字符串")
    if symbol != parquet_path.parent.name:
        raise ValueError("ETF 代码与行情目录不一致")
    count = _row_count(commit.get("row_count"))
    _matching_path(commit.get("processed_path"), parquet_path)

    if "provider_metadata.json" in documents:
        metadata = documents["provider_metadata.json"]
        if metadata.get("data_version") != version_id or metadata.get("symbol") != symbol:
            raise ValueError("Provider 元数据与版本身份不一致")
        if _row_count(metadata.get("row_count")) != count:
            raise ValueError("Provider 元数据行数不一致")
        _matching_path(metadata.get("processed_path"), parquet_path)
        if metadata.get("period") != "daily":
            raise ValueError("行情版本必须为日线")
        adjustment = metadata.get("adjustment")
        if adjustment not in ("none", "qfq", "hfq"):
            raise ValueError("Provider 复权口径无效")
        provider = _text(metadata.get("provider"), "provider")
        imported_at = metadata.get("fetched_at")
    else:
        if "metadata.json" not in documents:
            raise ValueError("缺少 CSV 版本元数据 metadata.json")
        metadata = documents["metadata.json"]
        if metadata.get("version_id") != version_id:
            raise ValueError("CSV 元数据版本号不一致")
        provider = "csv"
        if "source_format.json" in documents:
            provider = _text(documents["source_format.json"].get("detected_source"), "数据来源")
        adjustment = None
        imported_at = metadata.get("imported_at")

    _import_time(imported_at)
    return MarketVersionRecord(
        symbol=symbol, version_id=version_id, row_count=count,
        imported_at=imported_at, processed_path=parquet_path,
        raw_version_dir=directory, provider=provider, adjustment=adjustment,
        metadata_sha256=fingerprint,
    )


def scan_market_catalog(processed_dir: Path, raw_archive_dir: Path) -> CatalogScan:
    """列出所有健康版本，按实际导入时刻倒序；不读取整份行情内容。"""
    processed_root = _resolve(Path(processed_dir))
    raw_root = _resolve(Path(raw_archive_dir))
    markers: dict[str, list[Path]] = {}
    for path in raw_root.rglob("import_result.json"):
        markers.setdefault(path.parent.name, []).append(path)

    records: list[MarketVersionRecord] = []
    issues: list[str] = []
    for path in sorted(processed_root.glob("*/*.parquet")):
        try:
            parquet_path = _inside(path, processed_root)
            candidates = markers.get(path.stem, [])
            if len(candidates) != 1:
                raise ValueError("缺少或存在多份导入完成标记，无法唯一确定版本")
            directory = _inside(candidates[0].parent, raw_root)
            records.append(_record(parquet_path, directory))
        except (OSError, ValueError, TypeError) as error:
            issues.append(f"版本 {path.stem}（{path}）不可用：{error}")

    # 两次稳定排序：先定义同一时刻的代码/版本顺序，再按真实时刻倒序。
    records.sort(key=lambda record: (record.symbol, record.version_id))
    records.sort(key=lambda record: _import_time(record.imported_at), reverse=True)
    return CatalogScan(tuple(records), tuple(issues))


def read_market_version(record: MarketVersionRecord) -> tuple[pd.DataFrame, str]:
    """重新核对身份，使用同一份字节解码行情并计算 SHA-256 内容指纹。"""
    if (_resolve(record.processed_path) != record.processed_path
            or _resolve(record.raw_version_dir) != record.raw_version_dir):
        raise ValueError("所选版本的路径已经改变，请重新扫描目录")
    current = _record(record.processed_path, record.raw_version_dir)
    if current != record:
        raise ValueError("所选版本的元数据已经改变，请重新扫描目录")

    payload = record.processed_path.read_bytes()
    try:
        data = pd.read_parquet(BytesIO(payload))
    except (OSError, ValueError) as error:
        raise ValueError("所选版本的 Parquet 文件无法读取") from error
    if len(data) != record.row_count:
        raise ValueError("行情文件行数与版本记录不一致")
    if "symbol" not in data.columns or not data["symbol"].map(
        lambda value: isinstance(value, str) and value == record.symbol
    ).all():
        raise ValueError("行情文件内的 ETF 代码与版本记录不一致")
    return data.copy(deep=True), sha256(payload).hexdigest()
