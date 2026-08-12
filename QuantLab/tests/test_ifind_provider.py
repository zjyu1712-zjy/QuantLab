from copy import deepcopy
from datetime import date

import pandas as pd
import pytest

from quantlab.data.providers.base import MarketDataProviderResponseError
from quantlab.data.providers.ifind import (
    IFindETFProvider,
    adjustment_to_ifind_cps,
    to_ifind_etf_code,
)


IFIND_HISTORY_RESPONSE = {
    "errorcode": 0,
    "errmsg": "Success",
    "tables": [
        {
            "thscode": "510300.SH",
            "time": ["2025-01-02", "2025-01-03"],
            "table": {
                "open": [3.95, 4.01],
                "high": [4.02, 4.08],
                "low": [3.93, 3.99],
                "close": [4.00, 4.06],
                "volume": [1_200_000, 1_350_000],
            },
        }
    ],
}

NO_DATA_ERROR = "未找到该 ETF 的行情数据，请检查代码或日期范围。"


class FakeIFindClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    def post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append((path, payload))
        return self.response


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [("510300", "510300.SH"), ("588000", "588000.SH"), ("159516", "159516.SZ")],
)
def test_to_ifind_etf_code(symbol: str, expected: str) -> None:
    assert to_ifind_etf_code(symbol) == expected


@pytest.mark.parametrize(
    ("adjustment", "expected"),
    [("none", "1"), ("qfq", "2"), ("hfq", "3")],
)
def test_adjustment_to_ifind_cps(adjustment: str, expected: str) -> None:
    assert adjustment_to_ifind_cps(adjustment) == expected


def test_to_ifind_etf_code_rejects_unsupported_prefix() -> None:
    with pytest.raises(ValueError, match="沪深 ETF"):
        to_ifind_etf_code("000001")


@pytest.mark.parametrize("symbol", ["510300.SH", "510300abc"])
def test_to_ifind_etf_code_rejects_non_six_digit_input(symbol: str) -> None:
    with pytest.raises(ValueError):
        to_ifind_etf_code(symbol)


def test_provider_fetches_daily_history_in_standard_format() -> None:
    client = FakeIFindClient(IFIND_HISTORY_RESPONSE)
    provider = IFindETFProvider(client)

    result = provider.fetch_etf_history(
        "510300",
        date(2025, 1, 2),
        date(2025, 1, 3),
        period="daily",
        adjustment="none",
    )

    assert result.columns.tolist() == ["date", "open", "high", "low", "close", "volume"]
    assert result["date"].tolist() == ["2025-01-02", "2025-01-03"]
    assert client.calls == [
        (
            "/api/v1/cmd_history_quotation",
            {
                "codes": "510300.SH",
                "indicators": "open,high,low,close,volume",
                "startdate": "2025-01-02",
                "enddate": "2025-01-03",
                "functionpara": {"Interval": "D", "Fill": "Omit", "CPS": "1"},
            },
        )
    ]


def test_provider_rejects_non_daily_period() -> None:
    provider = IFindETFProvider(FakeIFindClient(IFIND_HISTORY_RESPONSE))

    with pytest.raises(ValueError):
        provider.fetch_etf_history("510300", date(2025, 1, 2), date(2025, 1, 3), period="weekly")


def test_provider_rejects_unsupported_adjustment() -> None:
    provider = IFindETFProvider(FakeIFindClient(IFIND_HISTORY_RESPONSE))

    with pytest.raises(ValueError):
        provider.fetch_etf_history("510300", date(2025, 1, 2), date(2025, 1, 3), adjustment="bad")


def test_provider_hides_raw_error_response() -> None:
    response = {"errorcode": 1001, "errmsg": "secret provider failure", "private": "token"}
    provider = IFindETFProvider(FakeIFindClient(response))

    with pytest.raises(MarketDataProviderResponseError) as error:
        provider.fetch_etf_history("510300", date(2025, 1, 2), date(2025, 1, 3))

    assert "secret provider failure" not in str(error.value)
    assert "token" not in str(error.value)


def test_provider_rejects_response_without_requested_code() -> None:
    response = deepcopy(IFIND_HISTORY_RESPONSE)
    del response["tables"][0]["thscode"]
    provider = IFindETFProvider(FakeIFindClient(response))

    with pytest.raises(MarketDataProviderResponseError) as error:
        provider.fetch_etf_history("510300", date(2025, 1, 2), date(2025, 1, 3))

    assert str(error.value) == "行情服务返回内容无法识别。"


def test_provider_rejects_response_for_different_code() -> None:
    response = deepcopy(IFIND_HISTORY_RESPONSE)
    response["tables"][0]["thscode"] = "159516.SZ"
    provider = IFindETFProvider(FakeIFindClient(response))

    with pytest.raises(MarketDataProviderResponseError) as error:
        provider.fetch_etf_history("510300", date(2025, 1, 2), date(2025, 1, 3))

    assert str(error.value) == "行情服务返回内容无法识别。"


@pytest.mark.parametrize(
    "response",
    [
        {"errorcode": 0},
        {"errorcode": 0, "tables": [{}, {}]},
        {"errorcode": 0, "tables": [{"table": {}}]},
        {
            "errorcode": 0,
            "tables": [
                {"time": ["2025-01-02"], "table": {"open": [1], "high": [1], "low": [1], "close": [1]}}
            ],
        },
        {
            "errorcode": 0,
            "tables": [
                {"time": ["2025-01-02"], "table": {"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1, 2]}}
            ],
        },
    ],
)
def test_provider_rejects_malformed_response(response: dict[str, object]) -> None:
    provider = IFindETFProvider(FakeIFindClient(response))

    with pytest.raises(MarketDataProviderResponseError):
        provider.fetch_etf_history("510300", date(2025, 1, 2), date(2025, 1, 3))


@pytest.mark.parametrize(
    "response",
    [
        {"errorcode": 0, "tables": []},
        {"errorcode": 0, "tables": [{"thscode": "510300.SH", "time": [], "table": {}}]},
    ],
)
def test_provider_translates_empty_response_to_no_data_error(response: dict[str, object]) -> None:
    provider = IFindETFProvider(FakeIFindClient(response))

    with pytest.raises(MarketDataProviderResponseError, match=f"^{NO_DATA_ERROR}$"):
        provider.fetch_etf_history("510300", date(2025, 1, 2), date(2025, 1, 3))


def test_provider_leaves_numeric_strings_for_standardization() -> None:
    response = {
        "errorcode": 0,
        "tables": [
            {
                "thscode": "510300.SH",
                "time": ["2025-01-02"],
                "table": {
                    "open": ["3.95"],
                    "high": ["4.02"],
                    "low": ["3.93"],
                    "close": ["4.00"],
                    "volume": ["1200000"],
                },
            }
        ],
    }
    provider = IFindETFProvider(FakeIFindClient(response))

    result = provider.fetch_etf_history("510300", date(2025, 1, 2), date(2025, 1, 2))

    assert result.iloc[0].to_dict() == {
        "date": "2025-01-02",
        "open": "3.95",
        "high": "4.02",
        "low": "3.93",
        "close": "4.00",
        "volume": "1200000",
    }
