from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from shutil import copy2


MARKET_SCHEMA_VERSION = "market-v1"


@dataclass(frozen=True)
class MarketDataVersion:
    """一次行情 CSV 导入所对应的版本信息。"""

    version_id: str
    source_filename: str
    archived_path: str
    imported_at: str
    sha256: str
    schema_version: str
    file_size_bytes: int


def _calculate_sha256(file_path: Path) -> str:
    """分块计算文件哈希，避免一次把大文件全部读入内存。"""
    digest = sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def archive_market_csv(
    source_path: str | Path,
    archive_dir: str | Path,
    *,
    imported_at: datetime | None = None,
) -> MarketDataVersion:
    """原样归档行情 CSV，并将数据版本信息保存为 JSON。"""
    source = Path(source_path)

    if not source.is_file():
        raise FileNotFoundError(f"找不到行情 CSV：{source}")

    if source.suffix.lower() != ".csv":
        raise ValueError("行情原始文件必须是 CSV")

    import_time = imported_at or datetime.now(timezone.utc)
    if import_time.tzinfo is None:
        raise ValueError("导入时间必须包含时区")

    import_time = import_time.astimezone(timezone.utc)
    file_hash = _calculate_sha256(source)
    timestamp = import_time.strftime("%Y%m%dT%H%M%S%fZ")
    version_id = f"{timestamp}_{file_hash[:12]}"

    version_dir = Path(archive_dir) / version_id
    version_dir.mkdir(parents=True, exist_ok=False)

    archived_file = version_dir / source.name
    copy2(source, archived_file)

    version = MarketDataVersion(
        version_id=version_id,
        source_filename=source.name,
        archived_path=str(archived_file),
        imported_at=import_time.isoformat(),
        sha256=file_hash,
        schema_version=MARKET_SCHEMA_VERSION,
        file_size_bytes=source.stat().st_size,
    )

    metadata_path = version_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            asdict(version),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return version
