"""自动行情获取、校验、缓存与版本保存服务。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import logging
from pathlib import Path
import re

import pandas as pd

from quantlab.data.providers import (
    MarketDataProvider,
    MarketDataProviderNoDataError,
)
from quantlab.data.storage import save_market_parquet
from quantlab.data.validator import validate_market_data


SUPPORTED_ADJUSTMENTS = {"none", "qfq", "hfq"}
SUPPORTED_PERIODS = {"daily"}
SYMBOL_PATTERN = re.compile(
    r"^(?:(?P<prefix>sh|sz))?(?P<symbol>\d{6})(?:\.(?P<suffix>sh|sz))?$",
    re.IGNORECASE,
)


class MarketDataServiceError(RuntimeError):
    """服务层可安全转换成用户提示的错误。"""


class MarketDataValidationError(MarketDataServiceError):
    """获取的数据未通过 QuantLab 质量检查。"""


@dataclass(frozen=True)
class MarketDataMetadata:
    symbol: str
    provider: str
    period: str
    adjustment: str
    start_date: str
    end_date: str
    fetched_at: str
    data_version: str
    row_count: int
    processed_path: str
    requested_start_date: str | None = None
    requested_end_date: str | None = None
    requested_ranges: tuple[tuple[str, str], ...] | None = None


@dataclass(frozen=True)
class MarketDataFetchResult:
    data: pd.DataFrame
    metadata: MarketDataMetadata
    cache_status: str
    fetched_ranges: tuple[tuple[str, str], ...]

    @property
    def symbol(self) -> str:
        return self.metadata.symbol

    @property
    def row_count(self) -> int:
        return len(self.data)


def normalize_etf_symbol(value: str) -> str:
    """接受常见沪深代码写法并统一为六位数字。"""
    normalized = value.strip().lower().replace(" ", "")
    if not normalized:
        raise ValueError("请输入 ETF 代码。")
    match = SYMBOL_PATTERN.fullmatch(normalized)
    if match is None:
        raise ValueError(
            "ETF 代码应为六位数字，例如 159516、159516.SZ 或 sh510300。"
        )
    prefix = match.group("prefix")
    suffix = match.group("suffix")
    if prefix and suffix and prefix.lower() != suffix.lower():
        raise ValueError("ETF 代码的交易所前缀和后缀不一致。")
    return match.group("symbol")


def validate_date_range(start_date: date, end_date: date) -> None:
    if not isinstance(start_date, date) or not isinstance(end_date, date):
        raise ValueError("开始日期和结束日期必须是有效日期。")
    if start_date > end_date:
        raise ValueError("开始日期不能晚于结束日期。")
    if end_date > date.today():
        raise ValueError("结束日期不能晚于今天。")


class MarketDataService:
    """页面调用的唯一自动行情服务。"""

    def __init__(
        self,
        provider: MarketDataProvider,
        raw_dir: str | Path,
        processed_dir: str | Path,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.provider = provider
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.logger = logger or logging.getLogger("quantlab.market_data")

    def _metadata_paths(self) -> list[Path]:
        return list(self.raw_dir.glob("**/provider_metadata.json"))

    @staticmethod
    def _merge_ranges(
        ranges: list[tuple[date, date]],
    ) -> tuple[tuple[date, date], ...]:
        merged: list[tuple[date, date]] = []
        for start, end in sorted(ranges):
            if start > end:
                raise ValueError("缓存请求覆盖区间无效")
            previous_end = merged[-1][1] if merged else None
            overlaps_or_adjacent = previous_end is not None and (
                start <= previous_end
                or (
                    previous_end < date.max
                    and start == previous_end + timedelta(days=1)
                )
            )
            if overlaps_or_adjacent:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return tuple(merged)

    @staticmethod
    def _requested_coverage_ranges(
        metadata: MarketDataMetadata,
    ) -> tuple[tuple[date, date], ...] | None:
        legacy_coverage: tuple[tuple[date, date], ...] | None = None
        start_text = metadata.requested_start_date
        end_text = metadata.requested_end_date
        if start_text is None and end_text is None:
            pass
        elif start_text is None or end_text is None:
            raise ValueError("缓存请求覆盖区间不完整")
        else:
            start = date.fromisoformat(start_text)
            end = date.fromisoformat(end_text)
            if start > end:
                raise ValueError("缓存请求覆盖区间无效")
            legacy_coverage = ((start, end),)

        if metadata.requested_ranges is not None:
            if not isinstance(metadata.requested_ranges, (list, tuple)):
                raise ValueError("缓存请求覆盖区间格式无效")
            parsed_ranges: list[tuple[date, date]] = []
            for item in metadata.requested_ranges:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise ValueError("缓存请求覆盖区间格式无效")
                parsed_ranges.append(
                    (date.fromisoformat(item[0]), date.fromisoformat(item[1]))
                )
            if not parsed_ranges:
                raise ValueError("缓存请求覆盖区间不能为空")
            return MarketDataService._merge_ranges(parsed_ranges)
        return legacy_coverage

    def _find_latest_cache(
        self,
        symbol: str,
        period: str,
        adjustment: str,
    ) -> tuple[pd.DataFrame, MarketDataMetadata] | None:
        candidates: list[tuple[datetime, MarketDataMetadata]] = []
        for metadata_path in self._metadata_paths():
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata = MarketDataMetadata(**payload)
                requested_ranges = self._requested_coverage_ranges(metadata)
                if metadata.requested_ranges is not None:
                    metadata = replace(
                        metadata,
                        requested_ranges=tuple(
                            (start.isoformat(), end.isoformat())
                            for start, end in requested_ranges or ()
                        ),
                    )
                if (
                    metadata.symbol == symbol
                    and metadata.provider == self.provider.key
                    and metadata.period == period
                    and metadata.adjustment == adjustment
                    and Path(metadata.processed_path).is_file()
                ):
                    candidates.append(
                        (datetime.fromisoformat(metadata.fetched_at), metadata)
                    )
            except (OSError, ValueError, TypeError, KeyError) as error:
                self.logger.warning("忽略损坏的行情缓存元数据 %s: %s", metadata_path, error)

        if not candidates:
            return None
        metadata = max(candidates, key=lambda item: item[0])[1]
        try:
            data = pd.read_parquet(metadata.processed_path)
            data = self._standardize_and_validate(data, symbol)
        except Exception as error:
            self.logger.warning("忽略损坏的行情缓存 %s: %s", metadata.processed_path, error)
            return None
        return data, metadata

    @staticmethod
    def _standardize_and_validate(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        required = ["date", "open", "high", "low", "close", "volume"]
        missing = set(required) - set(data.columns)
        if missing:
            raise MarketDataValidationError(
                "行情服务返回字段发生变化，请稍后重试或改用文件上传。"
            )

        standardized = data.copy()
        standardized["date"] = pd.to_datetime(standardized["date"], errors="coerce")
        for column in ["open", "high", "low", "close", "volume"]:
            standardized[column] = pd.to_numeric(standardized[column], errors="coerce")
        standardized["symbol"] = symbol
        standardized = standardized[
            ["date", "symbol", "open", "high", "low", "close", "volume"]
        ]
        standardized = standardized.drop_duplicates().sort_values("date").reset_index(drop=True)

        try:
            validate_market_data(standardized)
        except ValueError as error:
            raise MarketDataValidationError(str(error)) from error
        return standardized

    @staticmethod
    def _missing_ranges(
        start_date: date,
        end_date: date,
        cached_data: pd.DataFrame | None,
        cached_metadata: MarketDataMetadata | None = None,
    ) -> list[tuple[date, date]]:
        if cached_data is None or cached_data.empty:
            return [(start_date, end_date)]
        requested_coverage = (
            MarketDataService._requested_coverage_ranges(cached_metadata)
            if cached_metadata is not None
            else None
        )
        if requested_coverage is None:
            requested_coverage = (
                (
                    pd.Timestamp(cached_data["date"].min()).date(),
                    pd.Timestamp(cached_data["date"].max()).date(),
                ),
            )

        ranges: list[tuple[date, date]] = []
        cursor = start_date
        for covered_start, covered_end in requested_coverage:
            if covered_end < cursor:
                continue
            if covered_start > end_date:
                break
            if covered_start > cursor:
                ranges.append(
                    (cursor, min(end_date, covered_start - timedelta(days=1)))
                )
            if covered_end >= end_date:
                return ranges
            cursor = max(cursor, covered_end + timedelta(days=1))
        if cursor <= end_date:
            ranges.append((cursor, end_date))
        return ranges

    def _fetch_range(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        period: str,
        adjustment: str,
    ) -> pd.DataFrame:
        self.logger.info(
            "获取 ETF 行情 provider=%s symbol=%s range=%s..%s adjustment=%s",
            self.provider.key,
            symbol,
            start_date,
            end_date,
            adjustment,
        )
        data = self.provider.fetch_etf_history(
            symbol,
            start_date,
            end_date,
            period=period,
            adjustment=adjustment,
        )
        return self._standardize_and_validate(data, symbol)

    def _save_version(
        self,
        data: pd.DataFrame,
        raw_parts: list[pd.DataFrame],
        symbol: str,
        period: str,
        adjustment: str,
        requested_ranges: tuple[tuple[date, date], ...],
    ) -> MarketDataMetadata:
        fetched_at = datetime.now(timezone.utc)
        stable_bytes = data.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8")
        digest = sha256(stable_bytes).hexdigest()
        timestamp = fetched_at.strftime("%Y%m%dT%H%M%S%fZ")
        version_id = f"{timestamp}_{digest[:12]}"

        raw_version_dir = self.raw_dir / self.provider.key / version_id
        raw_version_dir.mkdir(parents=True, exist_ok=False)
        raw_snapshot = pd.concat(raw_parts, ignore_index=True) if raw_parts else data
        requested_start = requested_ranges[0][0]
        requested_end = requested_ranges[-1][1]
        raw_snapshot.to_csv(
            raw_version_dir / f"{symbol}_{requested_start}_{requested_end}.csv",
            index=False,
            date_format="%Y-%m-%d",
        )

        processed_path = self.processed_dir / symbol / f"{version_id}.parquet"
        save_market_parquet(data, processed_path)
        metadata = MarketDataMetadata(
            symbol=symbol,
            provider=self.provider.key,
            period=period,
            adjustment=adjustment,
            start_date=pd.Timestamp(data["date"].min()).date().isoformat(),
            end_date=pd.Timestamp(data["date"].max()).date().isoformat(),
            fetched_at=fetched_at.isoformat(),
            data_version=version_id,
            row_count=len(data),
            processed_path=str(processed_path),
            requested_start_date=requested_start.isoformat(),
            requested_end_date=requested_end.isoformat(),
            requested_ranges=tuple(
                (start.isoformat(), end.isoformat())
                for start, end in requested_ranges
            ),
        )
        (raw_version_dir / "provider_metadata.json").write_text(
            json.dumps(asdict(metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (raw_version_dir / "import_result.json").write_text(
            json.dumps(
                {
                    "symbol": symbol,
                    "row_count": len(data),
                    "processed_path": str(processed_path),
                    "version": {"version_id": version_id},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return metadata

    def fetch_etf_history(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        period: str = "daily",
        adjustment: str = "none",
    ) -> MarketDataFetchResult:
        normalized_symbol = normalize_etf_symbol(symbol)
        validate_date_range(start_date, end_date)
        if period not in SUPPORTED_PERIODS:
            raise ValueError("当前阶段仅支持日线行情。")
        if adjustment not in SUPPORTED_ADJUSTMENTS:
            raise ValueError("复权方式必须是不复权、前复权或后复权。")

        cache = self._find_latest_cache(normalized_symbol, period, adjustment)
        cached_data = cache[0] if cache else None
        cached_metadata = cache[1] if cache else None
        cached_coverage = (
            self._requested_coverage_ranges(cached_metadata)
            if cached_metadata is not None
            else None
        )
        if cached_coverage is None and cached_data is not None and not cached_data.empty:
            cached_coverage = (
                (
                    pd.Timestamp(cached_data["date"].min()).date(),
                    pd.Timestamp(cached_data["date"].max()).date(),
                ),
            )
        missing_ranges = self._missing_ranges(
            start_date,
            end_date,
            cached_data,
            cached_metadata,
        )

        if not missing_ranges and cached_data is not None and cached_metadata is not None:
            requested = cached_data.loc[
                cached_data["date"].between(
                    pd.Timestamp(start_date), pd.Timestamp(end_date)
                )
            ].reset_index(drop=True)
            if requested.empty:
                raise MarketDataServiceError(
                    "未找到该 ETF 的行情数据，请检查代码或日期范围。"
                )
            return MarketDataFetchResult(
                data=requested,
                metadata=cached_metadata,
                cache_status="hit",
                fetched_ranges=(),
            )

        fetched_parts: list[pd.DataFrame] = []
        for range_start, range_end in missing_ranges:
            try:
                fetched_parts.append(
                    self._fetch_range(
                        normalized_symbol,
                        range_start,
                        range_end,
                        period,
                        adjustment,
                    )
                )
            except MarketDataProviderNoDataError:
                if cached_data is None:
                    raise
                self.logger.info(
                    "缺少区间没有返回交易日，继续使用已有缓存 symbol=%s range=%s..%s",
                    normalized_symbol,
                    range_start,
                    range_end,
                )

        combined_parts = ([cached_data] if cached_data is not None else []) + fetched_parts
        combined = self._standardize_and_validate(
            pd.concat(combined_parts, ignore_index=True), normalized_symbol
        )
        confirmed_ranges = self._merge_ranges(
            list(cached_coverage or ()) + missing_ranges
        )
        metadata = self._save_version(
            combined,
            fetched_parts,
            normalized_symbol,
            period,
            adjustment,
            confirmed_ranges,
        )
        requested = combined.loc[
            combined["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        ].reset_index(drop=True)
        if requested.empty:
            raise MarketDataServiceError(
                "未找到该 ETF 的行情数据，请检查代码或日期范围。"
            )
        return MarketDataFetchResult(
            data=requested,
            metadata=metadata,
            cache_status="extended" if cached_data is not None else "miss",
            fetched_ranges=tuple(
                (left.isoformat(), right.isoformat()) for left, right in missing_ranges
            ),
        )


__all__ = [
    "MarketDataFetchResult",
    "MarketDataMetadata",
    "MarketDataService",
    "MarketDataServiceError",
    "MarketDataValidationError",
    "normalize_etf_symbol",
    "validate_date_range",
]
