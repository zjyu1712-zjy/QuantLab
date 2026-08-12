"""面向个人投资者的行情文件识别与标准化适配层。"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import unicodedata

import pandas as pd

from quantlab.data.validator import validate_market_data


STANDARD_MARKET_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

SUPPORTED_MARKET_SOURCES = {
    "auto": "自动识别（推荐）",
    "quantlab": "QuantLab 标准文件",
    "eastmoney": "东方财富历史行情",
    "ths": "同花顺历史行情",
    "tdx": "通达信历史行情",
    "yahoo": "Yahoo Finance",
}

RESERVED_MARKET_SOURCES = (
    "雪球",
    "AkShare",
    "JoinQuant 聚宽",
    "Baostock",
)


class MarketFormatError(ValueError):
    """可直接展示给普通用户的行情文件错误。"""

    def __init__(self, user_message: str, *, technical_detail: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.technical_detail = technical_detail


@dataclass(frozen=True)
class MarketFileInspection:
    """一次文件识别的结果，计算模块只接收标准化后的数据。"""

    source_key: str
    source_label: str
    encoding: str
    symbol: str
    field_mapping: dict[str, str]
    standardized_data: pd.DataFrame

    @property
    def row_count(self) -> int:
        return len(self.standardized_data)

    def to_pipeline_data(self) -> pd.DataFrame:
        """补充资产代码，转换为既有导入管线需要的七列结构。"""
        pipeline_data = self.standardized_data.copy()
        pipeline_data.insert(1, "symbol", self.symbol)
        return pipeline_data.loc[
            :,
            [
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ],
        ]


def _normalize_header(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return re.sub(r"[\s_\-./()（）\[\]【】]+", "", text)


_ALIASES = {
    "date": ("date", "日期", "交易日期", "trade_date", "时间"),
    "open": ("open", "开盘", "开盘价"),
    "high": ("high", "最高", "最高价"),
    "low": ("low", "最低", "最低价"),
    "close": ("close", "收盘", "收盘价"),
    "volume": (
        "volume",
        "vol",
        "成交量",
        "成交量(手)",
        "成交量（手）",
        "总手",
    ),
}

_SYMBOL_ALIASES = (
    "symbol",
    "code",
    "代码",
    "证券代码",
    "股票代码",
    "基金代码",
)

_ALIAS_LOOKUP = {
    _normalize_header(alias): target
    for target, aliases in _ALIASES.items()
    for alias in aliases
}

_SYMBOL_KEYS = {_normalize_header(alias) for alias in _SYMBOL_ALIASES}


def _read_market_file(content: bytes) -> tuple[pd.DataFrame, str]:
    if not content:
        raise MarketFormatError("上传的行情文件是空的，请重新选择文件。")

    errors: list[str] = []
    for encoding in ("utf-8-sig", "gb18030", "utf-8"):
        try:
            frame = pd.read_csv(
                BytesIO(content),
                encoding=encoding,
                sep=None,
                engine="python",
            )
            frame = frame.dropna(how="all").reset_index(drop=True)
            if len(frame.columns) <= 1:
                raise ValueError("未检测到分隔字段")
            return frame, encoding
        except Exception as error:  # pragma: no cover - 聚合解析原因
            errors.append(f"{encoding}: {error}")

    raise MarketFormatError(
        "暂未识别该行情文件，请选择数据来源，或下载 QuantLab 模板整理后上传。",
        technical_detail="；".join(errors),
    )


def _detect_source(
    filename: str,
    normalized_headers: set[str],
    requested_source: str,
) -> str:
    if requested_source not in SUPPORTED_MARKET_SOURCES:
        raise MarketFormatError("请选择页面列出的行情来源。")
    if requested_source != "auto":
        return requested_source

    lowered_name = Path(filename).name.lower()
    if "yahoo" in lowered_name or "adjclose" in normalized_headers:
        return "yahoo"
    if "eastmoney" in lowered_name or "东方财富" in lowered_name:
        return "eastmoney"
    if "同花顺" in lowered_name or "ths" in lowered_name:
        return "ths"
    if "通达信" in lowered_name or "tdx" in lowered_name:
        return "tdx"

    if {"date", "open", "high", "low", "close", "volume"}.issubset(
        normalized_headers
    ):
        return "quantlab"
    if {"振幅", "涨跌幅", "涨跌额", "换手率"} & normalized_headers:
        return "eastmoney"
    if {"涨幅", "涨跌", "总手"} & normalized_headers:
        return "ths"
    if {"时间", "成交额"} & normalized_headers:
        return "tdx"
    return "auto"


def _parse_dates(series: pd.Series) -> pd.Series:
    raw = series.astype("string").str.strip().str.replace(
        r"\.0$", "", regex=True
    )
    parsed = pd.to_datetime(raw, errors="coerce")
    if parsed.isna().any():
        raise MarketFormatError(
            "有些日期无法识别，请确认日期列使用常见年月日格式。",
            technical_detail=f"无法识别 {int(parsed.isna().sum())} 个日期",
        )
    return parsed


def _parse_numbers(series: pd.Series, field_label: str) -> pd.Series:
    raw = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("--", "", regex=False)
    )
    parsed = pd.to_numeric(raw, errors="coerce")
    if parsed.isna().any():
        raise MarketFormatError(
            f"{field_label}中有无法识别的数值，请检查空白或特殊符号。",
            technical_detail=f"无法识别 {int(parsed.isna().sum())} 个数值",
        )
    return parsed


def inspect_market_file(
    filename: str,
    content: bytes,
    *,
    symbol: str | None = None,
    source: str = "auto",
) -> MarketFileInspection:
    """识别常见行情字段，并转换为 QuantLab 的统一六列格式。"""
    frame, encoding = _read_market_file(content)
    original_headers = list(frame.columns)
    normalized_headers = {
        _normalize_header(column) for column in original_headers
    }
    source_key = _detect_source(
        filename,
        normalized_headers,
        source,
    )

    field_mapping: dict[str, str] = {}
    symbol_column: str | None = None
    for column in original_headers:
        key = _normalize_header(column)
        target = _ALIAS_LOOKUP.get(key)
        if target is not None and target not in field_mapping.values():
            field_mapping[str(column)] = target
        if key in _SYMBOL_KEYS and symbol_column is None:
            symbol_column = str(column)

    missing = [
        column
        for column in STANDARD_MARKET_COLUMNS
        if column not in field_mapping.values()
    ]
    if missing:
        missing_text = "、".join(missing)
        raise MarketFormatError(
            "暂未识别该行情格式，请选择数据来源，或下载 QuantLab 模板整理后上传。",
            technical_detail=f"未识别字段：{missing_text}",
        )

    resolved_symbol = (symbol or "").strip()
    if not resolved_symbol and symbol_column is not None:
        symbol_values = (
            frame[symbol_column]
            .astype("string")
            .str.strip()
            .dropna()
            .unique()
        )
        symbol_values = [value for value in symbol_values if value]
        if len(symbol_values) == 1:
            resolved_symbol = str(symbol_values[0])
        elif len(symbol_values) > 1:
            raise MarketFormatError(
                "一个文件中识别到多个资产代码，请每次只导入一只 ETF。"
            )

    if not resolved_symbol:
        raise MarketFormatError(
            "请输入 ETF 代码，系统会把它与这份历史行情关联。"
        )

    renamed = frame.rename(columns=field_mapping)
    standardized = renamed.loc[:, list(STANDARD_MARKET_COLUMNS)].copy()
    standardized["date"] = _parse_dates(standardized["date"])
    for field in ("open", "high", "low", "close"):
        standardized[field] = _parse_numbers(
            standardized[field],
            {"open": "开盘价", "high": "最高价", "low": "最低价", "close": "收盘价"}[field],
        ).astype(float)
    standardized["volume"] = _parse_numbers(
        standardized["volume"], "成交量"
    )
    standardized = standardized.sort_values("date").reset_index(drop=True)

    pipeline_data = standardized.copy()
    pipeline_data.insert(1, "symbol", resolved_symbol)
    validate_market_data(pipeline_data)

    source_label = SUPPORTED_MARKET_SOURCES.get(
        source_key,
        "自动识别行情文件",
    )
    if source_key == "auto":
        source_label = "自动识别行情文件"

    return MarketFileInspection(
        source_key=source_key,
        source_label=source_label,
        encoding=encoding,
        symbol=resolved_symbol,
        field_mapping=field_mapping,
        standardized_data=standardized,
    )


def create_market_template_bytes() -> bytes:
    """生成不含示例隐私数据的 QuantLab 行情模板。"""
    return (",".join(STANDARD_MARKET_COLUMNS) + "\n").encode("utf-8")
