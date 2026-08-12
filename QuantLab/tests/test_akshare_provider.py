from datetime import date

import pandas as pd
import pytest

from quantlab.data.providers import (
    AkShareETFProvider,
    MarketDataProviderNoDataError,
    MarketDataProviderResponseError,
    MarketDataProviderUnavailable,
)


def make_akshare_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "日期": ["2025-01-02", "2025-01-03"],
            "开盘": [3.95, 4.01],
            "收盘": [4.00, 4.06],
            "最高": [4.02, 4.08],
            "最低": [3.93, 3.99],
            "成交量": [1_200_000, 1_350_000],
            "成交额": [4_800_000, 5_400_000],
        }
    )


def test_akshare_provider_maps_fields_and_arguments() -> None:
    calls: list[dict[str, object]] = []

    def fake_fetcher(**kwargs: object) -> pd.DataFrame:
        calls.append(kwargs)
        return make_akshare_frame()

    provider = AkShareETFProvider(fetcher=fake_fetcher, max_attempts=1)
    result = provider.fetch_etf_history(
        "159516",
        date(2025, 1, 1),
        date(2025, 1, 31),
        adjustment="qfq",
    )

    assert calls == [
        {
            "symbol": "159516",
            "period": "daily",
            "start_date": "20250101",
            "end_date": "20250131",
            "adjust": "qfq",
        }
    ]
    assert list(result.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert result.loc[0, "close"] == pytest.approx(4.00)


def test_akshare_provider_rejects_empty_data() -> None:
    provider = AkShareETFProvider(
        fetcher=lambda **kwargs: pd.DataFrame(),
        max_attempts=1,
    )

    with pytest.raises(MarketDataProviderNoDataError, match="未找到"):
        provider.fetch_etf_history(
            "159516", date(2025, 1, 1), date(2025, 1, 31)
        )


def test_akshare_provider_retries_then_reports_failure() -> None:
    calls = 0

    def broken_fetcher(**kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        raise ConnectionError("temporary failure")

    provider = AkShareETFProvider(
        fetcher=broken_fetcher,
        max_attempts=2,
        retry_delay_seconds=0,
    )

    with pytest.raises(MarketDataProviderUnavailable, match="暂时不可用"):
        provider.fetch_etf_history(
            "159516", date(2025, 1, 1), date(2025, 1, 31)
        )
    assert calls == 2


def test_akshare_provider_rejects_changed_fields() -> None:
    provider = AkShareETFProvider(
        fetcher=lambda **kwargs: make_akshare_frame().drop(columns="成交量"),
        max_attempts=1,
    )

    with pytest.raises(MarketDataProviderResponseError, match="缺少：成交量"):
        provider.fetch_etf_history(
            "159516", date(2025, 1, 1), date(2025, 1, 31)
        )
