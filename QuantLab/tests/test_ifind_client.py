from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
import requests

from quantlab.data.providers.base import (
    MarketDataProviderAuthenticationError,
    MarketDataProviderRateLimitError,
    MarketDataProviderResponseError,
    MarketDataProviderTimeout,
    MarketDataProviderUnavailable,
)
from quantlab.data.providers.ifind_client import IFindQuantAPIClient


@dataclass
class FakeResponse:
    status_code: int
    payload: dict[str, Any]

    def json(self) -> dict[str, Any]:
        return self.payload


class InvalidJSONResponse:
    status_code = 200

    def json(self) -> Any:
        raise ValueError("not json")


class FakeTransport:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> Any:
        self.calls.append({"url": url, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_client_exchanges_and_reuses_access_token() -> None:
    """Catches a client that exchanges a refresh token for every request."""
    now = datetime(2026, 8, 13, tzinfo=UTC)
    transport = FakeTransport(
        [
            FakeResponse(200, {"data": {"access_token": "access-secret"}}),
            FakeResponse(200, {"errorcode": 0, "tables": []}),
            FakeResponse(200, {"errorcode": 0, "tables": []}),
        ]
    )
    client = IFindQuantAPIClient(
        "refresh-secret",
        transport=transport,
        clock=lambda: now,
        sleep=lambda _: None,
    )

    first_result = client.post_json(
        "/api/v1/cmd_history_quotation", {"codes": "510300.SH"}
    )
    second_result = client.post_json(
        "/api/v1/cmd_history_quotation", {"codes": "510300.SH"}
    )

    assert first_result == {"errorcode": 0, "tables": []}
    assert second_result == {"errorcode": 0, "tables": []}
    assert len(transport.calls) == 3
    assert transport.calls[0]["headers"]["refresh_token"] == "refresh-secret"
    assert transport.calls[1]["headers"]["access_token"] == "access-secret"
    assert transport.calls[2]["headers"]["access_token"] == "access-secret"


class TimeoutTransport:
    def __init__(self) -> None:
        self.call_count = 0

    def post(self, *args: Any, **kwargs: Any) -> FakeResponse:
        self.call_count += 1
        raise requests.Timeout("refresh-secret must never appear")


def test_client_retries_timeout_only_to_configured_limit() -> None:
    """Catches unbounded timeout retries or an exposed refresh token."""
    transport = TimeoutTransport()
    client = IFindQuantAPIClient(
        "refresh-secret",
        transport=transport,
        max_attempts=2,
        sleep=lambda _: None,
    )

    with pytest.raises(MarketDataProviderTimeout, match="连接超时") as captured:
        client.post_json("/api/v1/cmd_history_quotation", {})

    assert transport.call_count == 2
    assert "refresh-secret" not in str(captured.value)
    assert captured.value.__cause__ is None


def test_client_retries_connection_failure_only_to_configured_limit() -> None:
    """Catches a connection-failure retry loop that exceeds its configured limit."""
    transport = FakeTransport([requests.ConnectionError("offline")] * 3)
    client = IFindQuantAPIClient(
        "refresh-secret",
        transport=transport,
        max_attempts=2,
        sleep=lambda _: None,
    )

    with pytest.raises(MarketDataProviderUnavailable, match="暂时不可用"):
        client.post_json("/api/v1/cmd_history_quotation", {})

    assert len(transport.calls) == 2


def test_client_translates_rejected_refresh_token_without_leaking_it() -> None:
    """Catches an authentication error that includes a credential or response body."""
    transport = FakeTransport([FakeResponse(401, {"message": "refresh-secret"})])
    client = IFindQuantAPIClient(
        "refresh-secret",
        transport=transport,
        sleep=lambda _: None,
    )

    with pytest.raises(MarketDataProviderAuthenticationError) as captured:
        client.post_json("/api/v1/cmd_history_quotation", {})

    assert "refresh-secret" not in str(captured.value)


def test_client_translates_rate_limit_after_configured_retries() -> None:
    """Catches a 429 response treated as success or retried more than allowed."""
    transport = FakeTransport(
        [
            FakeResponse(200, {"data": {"access_token": "access-secret"}}),
            FakeResponse(429, {}),
            FakeResponse(429, {}),
        ]
    )
    client = IFindQuantAPIClient(
        "refresh-secret",
        transport=transport,
        max_attempts=2,
        sleep=lambda _: None,
    )

    with pytest.raises(MarketDataProviderRateLimitError, match="请求过于频繁"):
        client.post_json("/api/v1/cmd_history_quotation", {})

    assert len(transport.calls) == 3


def test_client_retries_server_errors_only_to_configured_limit() -> None:
    """Catches a server-error retry loop that exceeds its configured limit."""
    transport = FakeTransport(
        [
            FakeResponse(200, {"data": {"access_token": "access-secret"}}),
            FakeResponse(500, {}),
            FakeResponse(503, {}),
        ]
    )
    client = IFindQuantAPIClient(
        "refresh-secret",
        transport=transport,
        max_attempts=2,
        sleep=lambda _: None,
    )

    with pytest.raises(MarketDataProviderUnavailable, match="暂时不可用"):
        client.post_json("/api/v1/cmd_history_quotation", {})

    assert len(transport.calls) == 3


def test_client_translates_invalid_json_to_response_error() -> None:
    """Catches an invalid provider payload leaking a decoder exception."""
    transport = FakeTransport([InvalidJSONResponse()])
    client = IFindQuantAPIClient(
        "refresh-secret",
        transport=transport,
        sleep=lambda _: None,
    )

    with pytest.raises(MarketDataProviderResponseError, match="无法识别"):
        client.post_json("/api/v1/cmd_history_quotation", {})


def test_data_authentication_failure_refreshes_once_before_failing() -> None:
    """Catches a rejected access token that is retried without one fresh exchange."""
    transport = FakeTransport(
        [
            FakeResponse(200, {"data": {"access_token": "first-access-secret"}}),
            FakeResponse(401, {}),
            FakeResponse(200, {"data": {"access_token": "second-access-secret"}}),
            FakeResponse(401, {}),
        ]
    )
    client = IFindQuantAPIClient(
        "refresh-secret",
        transport=transport,
        sleep=lambda _: None,
    )

    with pytest.raises(MarketDataProviderAuthenticationError, match="身份验证失败"):
        client.post_json("/api/v1/cmd_history_quotation", {})

    assert len(transport.calls) == 4
    assert transport.calls[1]["headers"]["access_token"] == "first-access-secret"
    assert transport.calls[3]["headers"]["access_token"] == "second-access-secret"


@pytest.mark.parametrize(
    ("refresh_token", "timeout_seconds", "max_attempts"),
    [
        ("", 20.0, 2),
        ("refresh-secret", 0.0, 2),
        ("refresh-secret", 20.0, 0),
    ],
)
def test_client_rejects_invalid_constructor_values(
    refresh_token: str, timeout_seconds: float, max_attempts: int
) -> None:
    """Catches invalid configuration reaching the HTTP layer."""
    with pytest.raises(ValueError):
        IFindQuantAPIClient(
            refresh_token,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
        )
