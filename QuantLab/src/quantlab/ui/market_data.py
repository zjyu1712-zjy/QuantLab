from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from quantlab.data.importer import (
    MarketImportResult,
    import_market_csv,
)
from quantlab.data.market_formats import (
    MarketFileInspection,
    MarketFormatError,
    create_market_template_bytes,
    inspect_market_file,
)
from quantlab.data.market_service import (
    MarketDataServiceError,
    MarketDataValidationError,
)
from quantlab.data.providers import (
    MarketDataProviderError,
    MarketDataProviderResponseError,
)


PROVIDER_LABELS = {"akshare": "AkShare", "ifind": "iFinD"}


def market_source_label(provider: str, cache_status: str) -> str:
    """把内部行情来源和缓存状态转换为用户可读文案。"""
    provider_key = provider.strip().lower() if isinstance(provider, str) else ""
    label = PROVIDER_LABELS.get(provider_key, "其他行情服务")
    if cache_status == "hit":
        return f"本地数据（{label}）"
    return label


@dataclass(frozen=True)
class MarketDatasetRecord:
    """数据管理页面展示的一份标准化行情记录。"""

    symbol: str
    version_id: str
    row_count: int | None
    imported_at: str
    processed_path: str


@dataclass(frozen=True)
class FriendlyMarketImportResult:
    """页面需要的一次友好导入结果及来源说明。"""

    import_result: MarketImportResult
    inspection: MarketFileInspection
    original_path: str

    @property
    def symbol(self) -> str:
        return self.import_result.symbol

    @property
    def row_count(self) -> int:
        return self.import_result.row_count


def import_uploaded_market_csv(
    filename: str,
    content: bytes,
    raw_archive_dir: str | Path,
    processed_dir: str | Path,
) -> MarketImportResult:
    """把网页上传的 CSV 交给既有行情导入流程处理。"""
    safe_filename = Path(filename).name

    if not safe_filename or safe_filename != filename:
        raise ValueError("上传文件名无效")

    if Path(safe_filename).suffix.lower() != ".csv":
        raise ValueError("行情原始文件必须是 CSV")

    if not content:
        raise ValueError("上传的 CSV 不能为空")

    with TemporaryDirectory(prefix="quantlab-upload-") as temporary_dir:
        temporary_path = Path(temporary_dir) / safe_filename
        temporary_path.write_bytes(content)

        return import_market_csv(
            temporary_path,
            raw_archive_dir,
            processed_dir,
        )


def inspect_uploaded_market_history(
    filename: str,
    content: bytes,
    *,
    symbol: str | None = None,
    source: str = "auto",
) -> MarketFileInspection:
    """在写入任何目录前识别并检查用户上传的历史行情。"""
    safe_filename = Path(filename).name
    if not safe_filename or safe_filename != filename:
        raise MarketFormatError("上传文件名无效，请重新选择文件。")
    if Path(safe_filename).suffix.lower() != ".csv":
        raise MarketFormatError("目前支持上传 CSV 行情文件。")
    return inspect_market_file(
        safe_filename,
        content,
        symbol=symbol,
        source=source,
    )


def import_uploaded_market_history(
    filename: str,
    content: bytes,
    raw_archive_dir: str | Path,
    processed_dir: str | Path,
    *,
    symbol: str | None = None,
    source: str = "auto",
) -> FriendlyMarketImportResult:
    """识别外部行情，保留原文件，再交给既有导入管线。"""
    inspection = inspect_uploaded_market_history(
        filename,
        content,
        symbol=symbol,
        source=source,
    )

    with TemporaryDirectory(prefix="quantlab-friendly-upload-") as temp_dir:
        normalized_path = Path(temp_dir) / Path(filename).name
        inspection.to_pipeline_data().to_csv(
            normalized_path,
            index=False,
            date_format="%Y-%m-%d",
        )
        imported = import_market_csv(
            normalized_path,
            raw_archive_dir,
            processed_dir,
        )

    version_dir = Path(imported.version.archived_path).parent
    original_dir = version_dir / "source_original"
    original_dir.mkdir(parents=True, exist_ok=True)
    original_path = original_dir / Path(filename).name
    original_path.write_bytes(content)
    source_metadata = {
        "detected_source": inspection.source_key,
        "source_label": inspection.source_label,
        "encoding": inspection.encoding,
        "field_mapping": inspection.field_mapping,
        "original_path": str(original_path),
    }
    (version_dir / "source_format.json").write_text(
        json.dumps(source_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return FriendlyMarketImportResult(
        import_result=imported,
        inspection=inspection,
        original_path=str(original_path),
    )


def friendly_market_error(error: Exception) -> tuple[str, str]:
    """把底层异常转换成主提示和可折叠的高级说明。"""
    if isinstance(error, MarketFormatError):
        return error.user_message, error.technical_detail

    if isinstance(error, MarketDataProviderResponseError) and "未找到" in str(error):
        return "未找到该 ETF 的行情数据，请检查代码或日期范围。", str(error)
    if isinstance(error, MarketDataProviderError):
        return (
            "行情服务暂时无法连接，请稍后重试，或者使用文件上传方式导入数据。",
            str(error),
        )
    if isinstance(error, MarketDataValidationError):
        return (
            "获取到的行情存在明显异常，系统已停止保存。请稍后重试或使用文件上传。",
            str(error),
        )
    if isinstance(error, MarketDataServiceError):
        return str(error), str(error)

    message = str(error)
    if "多个" in message or "只能包含一只" in message:
        return "请每次只导入一只 ETF 的历史行情。", message
    if "日期" in message:
        return "有些日期无法识别，请检查日期列后重新上传。", message
    if "价格" in message or "成交量" in message:
        return "部分行情数值不合理，请检查价格和成交量后重新上传。", message
    return (
        "暂未识别该行情格式，请选择数据来源，或下载 QuantLab 模板整理后上传。",
        message,
    )


def list_market_datasets(
    processed_dir: str | Path,
    raw_archive_dir: str | Path,
) -> list[MarketDatasetRecord]:
    """列出已经标准化保存的行情版本，最新版本排在最前。"""
    processed_root = Path(processed_dir)
    raw_root = Path(raw_archive_dir)
    records: list[MarketDatasetRecord] = []

    for parquet_path in processed_root.glob("*/*.parquet"):
        version_id = parquet_path.stem
        result_paths = list(raw_root.glob(f"**/{version_id}/import_result.json"))
        metadata_paths = list(raw_root.glob(f"**/{version_id}/metadata.json"))
        provider_metadata_paths = list(
            raw_root.glob(f"**/{version_id}/provider_metadata.json")
        )
        result_path = result_paths[0] if result_paths else Path()
        metadata_path = metadata_paths[0] if metadata_paths else Path()
        provider_metadata_path = (
            provider_metadata_paths[0] if provider_metadata_paths else Path()
        )
        row_count: int | None = None
        imported_at = datetime.fromtimestamp(
            parquet_path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()

        if result_path.is_file():
            result_data = json.loads(
                result_path.read_text(encoding="utf-8")
            )
            row_count = int(result_data["row_count"])

        if metadata_path.is_file():
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            imported_at = str(metadata["imported_at"])

        if provider_metadata_path.is_file():
            provider_metadata = json.loads(
                provider_metadata_path.read_text(encoding="utf-8")
            )
            row_count = int(provider_metadata["row_count"])
            imported_at = str(provider_metadata["fetched_at"])

        records.append(
            MarketDatasetRecord(
                symbol=parquet_path.parent.name,
                version_id=version_id,
                row_count=row_count,
                imported_at=imported_at,
                processed_path=str(parquet_path),
            )
        )

    return sorted(
        records,
        key=lambda record: record.imported_at,
        reverse=True,
    )


def load_market_dataset(
    processed_path: str | Path,
) -> pd.DataFrame:
    """读取一份已经标准化的 Parquet 行情。"""
    path = Path(processed_path)

    if not path.is_file():
        raise FileNotFoundError(f"找不到标准化行情：{path}")

    return pd.read_parquet(path)
