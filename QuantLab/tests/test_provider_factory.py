from __future__ import annotations

from datetime import date
from typing import Any

import pytest
import requests

import quantlab.data.providers as providers_module
from quantlab.data.providers import get_market_data_provider
from quantlab.data.providers.akshare import AkShareETFProvider
from quantlab.data.providers.base import (
    MarketDataProviderAuthenticationError,
    MarketDataProviderTimeout,
    MarketDataProviderUnavailable,
)
from quantlab.data.providers.ifind import IFindETFProvider


def test_auto_without_token_selects_akshare() -> None:
    assert isinstance(get_market_data_provider(env={}), AkShareETFProvider)


def test_auto_with_token_selects_ifind() -> None:
    provider = get_market_data_provider(
        env={"IFIND_REFRESH_TOKEN": "test-refresh-token"},
    )

    assert isinstance(provider, IFindETFProvider)


def test_auto_with_access_token_selects_ifind() -> None:
    provider = get_market_data_provider(
        env={"IFIND_ACCESS_TOKEN": "test-access-token"},
    )

    assert isinstance(provider, IFindETFProvider)


def test_explicit_akshare_ignores_ifind_token() -> None:
    provider = get_market_data_provider(
        env={
            "MARKET_DATA_PROVIDER": "akshare",
            "IFIND_ACCESS_TOKEN": "test-access-token",
            "IFIND_REFRESH_TOKEN": "test-refresh-token",
        }
    )

    assert isinstance(provider, AkShareETFProvider)


def test_explicit_ifind_requires_server_token() -> None:
    with pytest.raises(MarketDataProviderUnavailable, match="尚未配置"):
        get_market_data_provider(env={"MARKET_DATA_PROVIDER": "ifind"})


def test_unknown_provider_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="MARKET_DATA_PROVIDER"):
        get_market_data_provider(env={"MARKET_DATA_PROVIDER": "unknown"})


class TimeoutTransport:
    def __init__(self) -> None:
        self.call_count = 0

    def post(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        raise requests.Timeout("simulated timeout")


def test_selected_ifind_timeout_is_not_silently_routed_to_akshare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid_akshare_construction() -> None:
        raise AssertionError("AkShare must not be constructed after iFinD selection")

    monkeypatch.setattr(
        providers_module,
        "AkShareETFProvider",
        forbid_akshare_construction,
    )
    transport = TimeoutTransport()
    provider = get_market_data_provider(
        env={"IFIND_REFRESH_TOKEN": "test-refresh-token"},
        ifind_transport=transport,
    )

    with pytest.raises(MarketDataProviderTimeout, match="连接超时"):
        provider.fetch_etf_history(
            "510300",
            start_date=date(2025, 1, 2),
            end_date=date(2025, 1, 3),
        )

    assert transport.call_count == 2


class RejectedTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> Any:
        self.calls.append({"url": url, **kwargs})
        return type("Response", (), {"status_code": 401})()


def test_access_token_takes_precedence_without_refresh_fallback() -> None:
    transport = RejectedTransport()
    provider = get_market_data_provider(
        env={
            "MARKET_DATA_PROVIDER": "ifind",
            "IFIND_ACCESS_TOKEN": "test-access-token",
            "IFIND_REFRESH_TOKEN": "test-refresh-token",
        },
        ifind_transport=transport,
    )

    with pytest.raises(
        MarketDataProviderAuthenticationError,
        match="access token 已失效",
    ):
        provider.fetch_etf_history(
            "510300",
            start_date=date(2025, 1, 2),
            end_date=date(2025, 1, 3),
        )

    assert len(transport.calls) == 1
    assert transport.calls[0]["headers"]["access_token"] == "test-access-token"
    assert "refresh_token" not in transport.calls[0]["headers"]
