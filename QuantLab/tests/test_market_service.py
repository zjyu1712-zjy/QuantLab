from datetime import date
import json
from pathlib import Path

import pandas as pd
import pytest

from quantlab.data.market_service import (
    MarketDataService,
    MarketDataValidationError,
    normalize_etf_symbol,
    validate_date_range,
)
from quantlab.data.providers import (
    MarketDataProvider,
    MarketDataProviderNoDataError,
    MarketDataProviderResponseError,
)


def make_standard_data(
    dates: list[str] | None = None,
) -> pd.DataFrame:
    dates = dates or ["2025-01-02", "2025-01-03", "2025-01-06"]
    size = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "open": [4.00 + index * 0.01 for index in range(size)],
            "high": [4.05 + index * 0.01 for index in range(size)],
            "low": [3.95 + index * 0.01 for index in range(size)],
            "close": [4.02 + index * 0.01 for index in range(size)],
            "volume": [1_000_000 + index for index in range(size)],
        }
    )


class FakeProvider(MarketDataProvider):
    name = "Fake"
    key = "fake"

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.calls: list[tuple[str, date, date, str, str]] = []

    def fetch_etf_history(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        period: str = "daily",
        adjustment: str = "none",
    ) -> pd.DataFrame:
        self.calls.append((symbol, start_date, end_date, period, adjustment))
        data = self.data.copy()
        parsed = pd.to_datetime(data["date"], errors="coerce")
        return data.loc[
            parsed.between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        ].reset_index(drop=True)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("159516", "159516"),
        ("159516.SZ", "159516"),
        ("sz159516", "159516"),
        ("510300", "510300"),
        ("510300.SH", "510300"),
        ("sh510300", "510300"),
    ],
)
def test_normalize_etf_symbol(value: str, expected: str) -> None:
    assert normalize_etf_symbol(value) == expected


@pytest.mark.parametrize("value", ["", "12345", "ABCDEF", "510300-X", "sh510300.SZ"])
def test_normalize_etf_symbol_rejects_invalid_input(value: str) -> None:
    with pytest.raises(ValueError):
        normalize_etf_symbol(value)


def test_validate_date_range_rejects_invalid_range() -> None:
    with pytest.raises(ValueError, match="开始日期"):
        validate_date_range(date(2025, 2, 1), date(2025, 1, 1))


def test_market_service_saves_standardized_data_and_metadata(tmp_path: Path) -> None:
    provider = FakeProvider(make_standard_data())
    service = MarketDataService(provider, tmp_path / "raw", tmp_path / "processed")

    result = service.fetch_etf_history(
        "sz159516", date(2025, 1, 1), date(2025, 1, 31)
    )

    assert result.symbol == "159516"
    assert result.cache_status == "miss"
    assert list(result.data.columns) == [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert Path(result.metadata.processed_path).is_file()
    assert list((tmp_path / "raw" / "fake").glob("*/provider_metadata.json"))


def test_market_service_cache_hit_does_not_call_provider_again(tmp_path: Path) -> None:
    provider = FakeProvider(make_standard_data())
    service = MarketDataService(provider, tmp_path / "raw", tmp_path / "processed")
    service.fetch_etf_history(
        "159516", date(2025, 1, 2), date(2025, 1, 6)
    )
    second = service.fetch_etf_history(
        "159516", date(2025, 1, 2), date(2025, 1, 3)
    )

    assert len(provider.calls) == 1
    assert second.cache_status == "hit"
    assert len(second.data) == 2


def test_market_service_fetches_only_missing_tail(tmp_path: Path) -> None:
    provider = FakeProvider(make_standard_data())
    service = MarketDataService(provider, tmp_path / "raw", tmp_path / "processed")
    service.fetch_etf_history(
        "159516", date(2025, 1, 2), date(2025, 1, 3)
    )
    result = service.fetch_etf_history(
        "159516", date(2025, 1, 2), date(2025, 1, 6)
    )

    assert provider.calls[1][1:3] == (date(2025, 1, 4), date(2025, 1, 6))
    assert result.cache_status == "extended"
    assert len(result.data) == 3


def test_market_service_refetches_when_cache_is_corrupted(tmp_path: Path) -> None:
    provider = FakeProvider(make_standard_data())
    service = MarketDataService(provider, tmp_path / "raw", tmp_path / "processed")
    first = service.fetch_etf_history(
        "159516", date(2025, 1, 2), date(2025, 1, 6)
    )
    Path(first.metadata.processed_path).write_bytes(b"not a parquet file")

    second = service.fetch_etf_history(
        "159516", date(2025, 1, 2), date(2025, 1, 6)
    )

    assert len(provider.calls) == 2
    assert second.cache_status == "miss"
    assert len(second.data) == 3


def test_market_service_removes_fully_duplicate_records(tmp_path: Path) -> None:
    data = pd.concat([make_standard_data(), make_standard_data().iloc[[0]]])
    service = MarketDataService(
        FakeProvider(data), tmp_path / "raw", tmp_path / "processed"
    )

    result = service.fetch_etf_history(
        "159516", date(2025, 1, 1), date(2025, 1, 31)
    )

    assert len(result.data) == 3


def test_market_service_rejects_conflicting_duplicate_date(tmp_path: Path) -> None:
    data = make_standard_data()
    conflicting = data.iloc[[0]].copy()
    conflicting["close"] = 4.03
    data = pd.concat([data, conflicting], ignore_index=True)
    service = MarketDataService(
        FakeProvider(data), tmp_path / "raw", tmp_path / "processed"
    )

    with pytest.raises(MarketDataValidationError, match="重复"):
        service.fetch_etf_history(
            "159516", date(2025, 1, 1), date(2025, 1, 31)
        )


def test_market_service_rejects_invalid_price(tmp_path: Path) -> None:
    data = make_standard_data()
    data.loc[0, "high"] = 3.90
    service = MarketDataService(
        FakeProvider(data), tmp_path / "raw", tmp_path / "processed"
    )

    with pytest.raises(MarketDataValidationError, match="最高价"):
        service.fetch_etf_history(
            "159516", date(2025, 1, 1), date(2025, 1, 31)
        )


def test_market_service_accepts_replacement_provider(tmp_path: Path) -> None:
    provider = FakeProvider(make_standard_data())
    provider.name = "Commercial Provider"
    provider.key = "commercial"
    service = MarketDataService(provider, tmp_path / "raw", tmp_path / "processed")

    result = service.fetch_etf_history(
        "510300", date(2025, 1, 1), date(2025, 1, 31), adjustment="hfq"
    )

    assert result.metadata.provider == "commercial"
    assert result.metadata.adjustment == "hfq"


def test_market_service_extension_preserves_existing_requested_coverage(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        make_standard_data(
            ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"]
        )
    )
    service = MarketDataService(provider, tmp_path / "raw", tmp_path / "processed")

    service.fetch_etf_history("159516", date(2025, 1, 2), date(2025, 1, 6))
    extended = service.fetch_etf_history(
        "159516", date(2025, 1, 1), date(2025, 1, 3)
    )
    cached_tail = service.fetch_etf_history(
        "159516", date(2025, 1, 4), date(2025, 1, 6)
    )

    assert extended.metadata.requested_start_date == "2025-01-01"
    assert extended.metadata.requested_end_date == "2025-01-06"
    assert cached_tail.cache_status == "hit"
    assert len(provider.calls) == 2


class ScriptedProvider(MarketDataProvider):
    name = "Scripted"
    key = "scripted"

    def __init__(self) -> None:
        self.calls: list[tuple[date, date]] = []
        self.fail_right = False
        self.empty_left = False

    def fetch_etf_history(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        period: str = "daily",
        adjustment: str = "none",
    ) -> pd.DataFrame:
        self.calls.append((start_date, end_date))
        if self.empty_left and (start_date, end_date) == (
            date(2025, 1, 1),
            date(2025, 1, 1),
        ):
            raise MarketDataProviderNoDataError("没有交易日")
        if self.fail_right and (start_date, end_date) == (
            date(2025, 1, 4),
            date(2025, 1, 4),
        ):
            raise MarketDataProviderResponseError("响应结构损坏")
        dates = pd.date_range(start_date, end_date, freq="D")
        return make_standard_data([item.date().isoformat() for item in dates])


def test_market_service_records_confirmed_no_data_gap_as_covered(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider()
    service = MarketDataService(provider, tmp_path / "raw", tmp_path / "processed")
    service.fetch_etf_history("159516", date(2025, 1, 2), date(2025, 1, 3))
    provider.empty_left = True

    extended = service.fetch_etf_history(
        "159516", date(2025, 1, 1), date(2025, 1, 3)
    )
    repeated = service.fetch_etf_history(
        "159516", date(2025, 1, 1), date(2025, 1, 3)
    )

    assert extended.cache_status == "extended"
    assert extended.fetched_ranges == (("2025-01-01", "2025-01-01"),)
    assert extended.metadata.requested_start_date == "2025-01-01"
    assert extended.metadata.requested_end_date == "2025-01-03"
    assert repeated.cache_status == "hit"
    assert len(provider.calls) == 2


def test_market_service_does_not_commit_partial_extension_on_response_error(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider()
    service = MarketDataService(provider, tmp_path / "raw", tmp_path / "processed")
    original = service.fetch_etf_history(
        "159516", date(2025, 1, 2), date(2025, 1, 3)
    )
    provider.fail_right = True
    metadata_paths_before = set((tmp_path / "raw").glob("**/provider_metadata.json"))

    with pytest.raises(MarketDataProviderResponseError, match="结构损坏"):
        service.fetch_etf_history("159516", date(2025, 1, 1), date(2025, 1, 4))

    metadata_paths_after = set((tmp_path / "raw").glob("**/provider_metadata.json"))
    assert metadata_paths_after == metadata_paths_before
    payload = json.loads(next(iter(metadata_paths_after)).read_text(encoding="utf-8"))
    assert payload["data_version"] == original.metadata.data_version
    assert payload["requested_start_date"] == "2025-01-02"
    assert payload["requested_end_date"] == "2025-01-03"

    with pytest.raises(MarketDataProviderResponseError, match="结构损坏"):
        service.fetch_etf_history("159516", date(2025, 1, 1), date(2025, 1, 4))

    assert provider.calls[-1] == (date(2025, 1, 4), date(2025, 1, 4))


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("requested_start_date", "not-a-date"),
        ("requested_end_date", "2025-00-99"),
    ],
)
def test_market_service_ignores_corrupted_requested_coverage(
    tmp_path: Path,
    field: str,
    bad_value: str,
) -> None:
    first_provider = FakeProvider(make_standard_data())
    service = MarketDataService(
        first_provider, tmp_path / "raw", tmp_path / "processed"
    )
    service.fetch_etf_history("159516", date(2025, 1, 2), date(2025, 1, 6))
    metadata_path = next((tmp_path / "raw").glob("**/provider_metadata.json"))
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload[field] = bad_value
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    replacement_provider = FakeProvider(make_standard_data())
    replacement_service = MarketDataService(
        replacement_provider, tmp_path / "raw", tmp_path / "processed"
    )
    result = replacement_service.fetch_etf_history(
        "159516", date(2025, 1, 2), date(2025, 1, 6)
    )

    assert result.cache_status == "miss"
    assert len(replacement_provider.calls) == 1
