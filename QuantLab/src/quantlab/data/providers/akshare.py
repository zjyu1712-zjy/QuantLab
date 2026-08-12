"""AkShare ETF 日线行情 Provider。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import date
import importlib.util
import time
from typing import Callable

import pandas as pd

from quantlab.data.providers.base import (
    STANDARD_PROVIDER_COLUMNS,
    MarketDataProvider,
    MarketDataProviderNoDataError,
    MarketDataProviderResponseError,
    MarketDataProviderTimeout,
    MarketDataProviderUnavailable,
)


AKSHARE_FIELD_MAPPING = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
}

ADJUSTMENT_VALUES = {
    "none": "",
    "qfq": "qfq",
    "hfq": "hfq",
}


class AkShareETFProvider(MarketDataProvider):
    """通过 AkShare 的东方财富接口获取沪深 ETF 历史行情。"""

    name = "AkShare"
    key = "akshare"

    def __init__(
        self,
        *,
        timeout_seconds: float = 20.0,
        max_attempts: int = 2,
        retry_delay_seconds: float = 0.5,
        fetcher: Callable[..., pd.DataFrame] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("请求超时时间必须大于 0")
        if max_attempts < 1:
            raise ValueError("请求次数至少为 1")
        if retry_delay_seconds < 0:
            raise ValueError("重试等待时间不能为负数")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self._fetcher = fetcher

    @property
    def is_available(self) -> bool:
        return self._fetcher is not None or importlib.util.find_spec("akshare") is not None

    def _resolve_fetcher(self) -> Callable[..., pd.DataFrame]:
        if self._fetcher is not None:
            return self._fetcher
        if not self.is_available:
            raise MarketDataProviderUnavailable(
                "当前环境尚未安装 AkShare，请先安装项目依赖。"
            )
        import akshare as ak

        return ak.fund_etf_hist_em

    def _call_once(self, **kwargs: object) -> pd.DataFrame:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._resolve_fetcher(), **kwargs)
        try:
            return future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError as error:
            future.cancel()
            raise MarketDataProviderTimeout(
                f"AkShare 请求超过 {self.timeout_seconds:g} 秒"
            ) from error
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def fetch_etf_history(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        period: str = "daily",
        adjustment: str = "none",
    ) -> pd.DataFrame:
        if period not in {"daily", "weekly", "monthly"}:
            raise ValueError("行情周期必须是 daily、weekly 或 monthly")
        if adjustment not in ADJUSTMENT_VALUES:
            raise ValueError("复权方式必须是 none、qfq 或 hfq")

        kwargs = {
            "symbol": symbol,
            "period": period,
            "start_date": start_date.strftime("%Y%m%d"),
            "end_date": end_date.strftime("%Y%m%d"),
            "adjust": ADJUSTMENT_VALUES[adjustment],
        }
        last_error: Exception | None = None
        raw_data: pd.DataFrame | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                raw_data = self._call_once(**kwargs)
                break
            except MarketDataProviderTimeout as error:
                last_error = error
            except Exception as error:  # 第三方接口异常统一交给服务层解释
                last_error = error
            if attempt < self.max_attempts:
                time.sleep(self.retry_delay_seconds)

        if raw_data is None:
            raise MarketDataProviderUnavailable(
                "AkShare 行情接口暂时不可用"
            ) from last_error
        if raw_data.empty:
            raise MarketDataProviderNoDataError(
                "未找到该 ETF 的行情数据，请检查代码或日期范围。"
            )

        missing = set(AKSHARE_FIELD_MAPPING) - set(raw_data.columns)
        if missing:
            missing_text = "、".join(sorted(missing))
            raise MarketDataProviderResponseError(
                f"AkShare 返回字段发生变化，缺少：{missing_text}"
            )

        standardized = raw_data.rename(columns=AKSHARE_FIELD_MAPPING).loc[
            :, list(STANDARD_PROVIDER_COLUMNS)
        ].copy()
        return standardized
