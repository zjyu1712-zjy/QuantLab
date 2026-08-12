from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from quantlab.data.versioning import (
    MARKET_SCHEMA_VERSION,
    archive_market_csv,
)


def test_archive_market_csv_preserves_original_and_writes_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.csv"
    original_content = (
        "date,symbol,open,high,low,close,volume\n"
        "2025-01-02,510300,3.95,4.02,3.93,4.00,1200000\n"
    )
    source.write_text(original_content, encoding="utf-8")
    imported_at = datetime(
        2025,
        1,
        10,
        8,
        30,
        tzinfo=timezone.utc,
    )

    version = archive_market_csv(
        source,
        tmp_path / "archive",
        imported_at=imported_at,
    )

    archived_path = Path(version.archived_path)
    metadata_path = archived_path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert source.read_text(encoding="utf-8") == original_content
    assert archived_path.read_text(encoding="utf-8") == original_content
    assert version.source_filename == "sample.csv"
    assert version.schema_version == MARKET_SCHEMA_VERSION
    assert version.file_size_bytes == source.stat().st_size
    assert metadata["version_id"] == version.version_id
    assert metadata["sha256"] == version.sha256
    assert metadata["imported_at"] == "2025-01-10T08:30:00+00:00"


def test_archive_market_csv_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="找不到行情 CSV"):
        archive_market_csv(
            tmp_path / "missing.csv",
            tmp_path / "archive",
        )


def test_archive_market_csv_rejects_non_csv(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("not csv", encoding="utf-8")

    with pytest.raises(ValueError, match="必须是 CSV"):
        archive_market_csv(
            source,
            tmp_path / "archive",
        )


def test_archive_market_csv_requires_timezone(tmp_path: Path) -> None:
    source = tmp_path / "sample.csv"
    source.write_text("date,symbol\n", encoding="utf-8")

    with pytest.raises(ValueError, match="必须包含时区"):
        archive_market_csv(
            source,
            tmp_path / "archive",
            imported_at=datetime(2025, 1, 10, 8, 30),
        )
