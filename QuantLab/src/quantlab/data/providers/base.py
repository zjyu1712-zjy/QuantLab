"""可替换的历史行情数据源接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


STANDARD_PROVIDER_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


class MarketDataProviderError(RuntimeError):
    """行情提供方调用失败。"""


class MarketDataProviderUnavailable(MarketDataProviderError):
    """当前运行环境无法使用指定行情提供方。"""


class MarketDataProviderTimeout(MarketDataProviderError):
    """行情提供方在限定时间内没有返回结果。"""


class MarketDataProviderResponseError(MarketDataProviderError):
    """行情提供方返回无法识别的响应。"""


class MarketDataProviderNoDataError(MarketDataProviderResponseError):
    """行情提供方明确表示请求区间没有数据。"""


class MarketDataProviderAuthenticationError(MarketDataProviderUnavailable):
    """Raised when provider credentials are absent, expired, or rejected."""


class MarketDataProviderRateLimitError(MarketDataProviderUnavailable):
    """Raised when the provider refuses a request because of quota limits."""


class MarketDataProvider(ABC):
    """所有行情来源必须实现的稳定接口。"""

    name: str
    key: str

    @property
    def is_available(self) -> bool:
        """当前运行环境是否已具备该 Provider 的依赖。"""
        return True

    @abstractmethod
    def fetch_etf_history(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        period: str = "daily",
        adjustment: str = "none",
    ) -> pd.DataFrame:
        """获取 ETF 行情并返回 QuantLab 六列标准结构。"""

    def fetch(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """兼容早期占位接口。"""
        return self.fetch_etf_history(symbol, start_date, end_date)
