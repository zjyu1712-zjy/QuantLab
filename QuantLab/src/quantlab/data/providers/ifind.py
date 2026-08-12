"""iFinD QuantAPI ETF history provider."""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol

import pandas as pd

from .base import (
    STANDARD_PROVIDER_COLUMNS,
    MarketDataProvider,
    MarketDataProviderNoDataError,
    MarketDataProviderResponseError,
)


HISTORY_PATH = "/api/v1/cmd_history_quotation"
IFIND_INDICATORS = "open,high,low,close,volume"
ADJUSTMENT_TO_CPS = {"none": "1", "qfq": "2", "hfq": "3"}
_NO_DATA_MESSAGE = "未找到该 ETF 的行情数据，请检查代码或日期范围。"
_INVALID_RESPONSE_MESSAGE = "行情服务返回内容无法识别。"


class IFindHistoryClient(Protocol):
    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]: ...


def to_ifind_etf_code(symbol: str) -> str:
    """Attach the exchange suffix required by iFinD's ETF history endpoint."""
    if (
        not isinstance(symbol, str)
        or len(symbol) != 6
        or not symbol.isascii()
        or not symbol.isdigit()
    ):
        raise ValueError("ETF 代码必须是 6 位数字")
    if symbol.startswith(("50", "51", "52", "56", "58")):
        return f"{symbol}.SH"
    if symbol.startswith(("15", "16")):
        return f"{symbol}.SZ"
    raise ValueError("第一阶段仅支持沪深 ETF 代码")


def adjustment_to_ifind_cps(adjustment: str) -> str:
    """Translate QuantLab adjustment names to the QuantAPI CPS option."""
    try:
        return ADJUSTMENT_TO_CPS[adjustment]
    except KeyError as error:
        raise ValueError("不支持的复权方式") from error


class IFindETFProvider(MarketDataProvider):
    """Fetch daily ETF OHLCV data from the HTTP iFinD QuantAPI."""

    key = "ifind"

    def __init__(self, client: IFindHistoryClient) -> None:
        self.client = client

    def fetch_etf_history(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        period: str = "daily",
        adjustment: str = "none",
    ) -> pd.DataFrame:
        if period != "daily":
            raise ValueError("第一阶段仅支持日频 ETF 行情")

        code = to_ifind_etf_code(symbol)
        cps = adjustment_to_ifind_cps(adjustment)
        response = self.client.post_json(
            HISTORY_PATH,
            {
                "codes": code,
                "indicators": IFIND_INDICATORS,
                "startdate": start_date.isoformat(),
                "enddate": end_date.isoformat(),
                "functionpara": {"Interval": "D", "Fill": "Omit", "CPS": cps},
            },
        )
        return self._parse_history_response(response, code)

    @staticmethod
    def _parse_history_response(
        response: dict[str, Any], expected_code: str
    ) -> pd.DataFrame:
        if response.get("errorcode") != 0:
            raise MarketDataProviderResponseError(_INVALID_RESPONSE_MESSAGE)

        tables = response.get("tables")
        if not isinstance(tables, list):
            raise MarketDataProviderResponseError(_INVALID_RESPONSE_MESSAGE)
        if not tables:
            raise MarketDataProviderNoDataError(_NO_DATA_MESSAGE)
        if len(tables) != 1 or not isinstance(tables[0], dict):
            raise MarketDataProviderResponseError(_INVALID_RESPONSE_MESSAGE)

        result_table = tables[0]
        if result_table.get("thscode") != expected_code:
            raise MarketDataProviderResponseError(_INVALID_RESPONSE_MESSAGE)
        dates = result_table.get("time")
        if not isinstance(dates, list):
            raise MarketDataProviderResponseError(_INVALID_RESPONSE_MESSAGE)
        if not dates:
            raise MarketDataProviderNoDataError(_NO_DATA_MESSAGE)

        values = result_table.get("table")
        if not isinstance(values, dict):
            raise MarketDataProviderResponseError(_INVALID_RESPONSE_MESSAGE)

        parsed: dict[str, list[Any]] = {"date": dates}
        for field in STANDARD_PROVIDER_COLUMNS[1:]:
            field_values = values.get(field)
            if not isinstance(field_values, list) or len(field_values) != len(dates):
                raise MarketDataProviderResponseError(_INVALID_RESPONSE_MESSAGE)
            parsed[field] = field_values

        return pd.DataFrame(parsed, columns=STANDARD_PROVIDER_COLUMNS)
