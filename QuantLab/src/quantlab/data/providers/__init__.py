"""行情 Provider 的稳定导出入口。"""

from __future__ import annotations

from collections.abc import Mapping
import os

from quantlab.data.providers.akshare import (
    ADJUSTMENT_VALUES,
    AKSHARE_FIELD_MAPPING,
    AkShareETFProvider,
)
from quantlab.data.providers.base import (
    STANDARD_PROVIDER_COLUMNS,
    MarketDataProvider,
    MarketDataProviderAuthenticationError,
    MarketDataProviderError,
    MarketDataProviderRateLimitError,
    MarketDataProviderResponseError,
    MarketDataProviderTimeout,
    MarketDataProviderUnavailable,
)
from quantlab.data.providers.ifind import (
    ADJUSTMENT_TO_CPS,
    HISTORY_PATH,
    IFIND_INDICATORS,
    IFindETFProvider,
    adjustment_to_ifind_cps,
    to_ifind_etf_code,
)
from quantlab.data.providers.ifind_client import (
    HTTPTransport,
    IFindQuantAPIClient,
)


VALID_PROVIDER_MODES = {"auto", "akshare", "ifind"}


def get_market_data_provider(
    env: Mapping[str, str] | None = None,
    *,
    ifind_transport: HTTPTransport | None = None,
) -> MarketDataProvider:
    """按服务端配置选择一个行情来源。"""
    settings = os.environ if env is None else env
    mode = settings.get("MARKET_DATA_PROVIDER", "auto").strip().lower()
    if mode not in VALID_PROVIDER_MODES:
        raise ValueError("MARKET_DATA_PROVIDER 必须是 auto、akshare 或 ifind")

    refresh_token = settings.get("IFIND_REFRESH_TOKEN", "").strip()
    if mode == "akshare" or (mode == "auto" and not refresh_token):
        return AkShareETFProvider()
    if not refresh_token:
        raise MarketDataProviderUnavailable("iFinD 行情服务尚未配置")

    return IFindETFProvider(
        IFindQuantAPIClient(refresh_token, transport=ifind_transport)
    )


__all__ = [
    "ADJUSTMENT_VALUES",
    "ADJUSTMENT_TO_CPS",
    "AKSHARE_FIELD_MAPPING",
    "HISTORY_PATH",
    "HTTPTransport",
    "IFIND_INDICATORS",
    "STANDARD_PROVIDER_COLUMNS",
    "VALID_PROVIDER_MODES",
    "AkShareETFProvider",
    "IFindETFProvider",
    "IFindQuantAPIClient",
    "MarketDataProvider",
    "MarketDataProviderAuthenticationError",
    "MarketDataProviderError",
    "MarketDataProviderRateLimitError",
    "MarketDataProviderResponseError",
    "MarketDataProviderTimeout",
    "MarketDataProviderUnavailable",
    "adjustment_to_ifind_cps",
    "get_market_data_provider",
    "to_ifind_etf_code",
]
