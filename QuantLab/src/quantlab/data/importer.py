from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path

from quantlab.data.market_loader import load_market_csv
from quantlab.data.storage import save_market_parquet
from quantlab.data.versioning import (
    MarketDataVersion,
    archive_market_csv,
)


@dataclass(frozen=True)
class MarketImportResult:
    """一次成功的 ETF 行情导入结果。"""

    version: MarketDataVersion
    symbol: str
    row_count: int
    processed_path: str


def import_market_csv(
    source_path: str | Path,
    raw_archive_dir: str | Path,
    processed_dir: str | Path,
    *,
    imported_at: datetime | None = None,
) -> MarketImportResult:
    """完成原始文件归档、校验、标准化和 Parquet 保存。"""
    version = archive_market_csv(
        source_path,
        raw_archive_dir,
        imported_at=imported_at,
    )

    market_data = load_market_csv(version.archived_path)
    symbols = market_data["symbol"].unique()

    if len(symbols) != 1:
        raise ValueError("第一阶段每个行情 CSV 只能包含一只 ETF")

    symbol = str(symbols[0])
    processed_path = (
        Path(processed_dir)
        / symbol
        / f"{version.version_id}.parquet"
    )
    save_market_parquet(market_data, processed_path)

    result = MarketImportResult(
        version=version,
        symbol=symbol,
        row_count=len(market_data),
        processed_path=str(processed_path),
    )

    result_path = Path(version.archived_path).parent / "import_result.json"
    result_path.write_text(
        json.dumps(
            asdict(result),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return result
