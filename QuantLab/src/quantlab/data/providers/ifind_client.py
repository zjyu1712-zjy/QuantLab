"""Authenticated HTTP client for the iFinD QuantAPI."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
import math
import time
from typing import Any, Protocol

import requests

from .base import (
    MarketDataProviderAuthenticationError,
    MarketDataProviderRateLimitError,
    MarketDataProviderResponseError,
    MarketDataProviderTimeout,
    MarketDataProviderUnavailable,
)


IFIND_BASE_URL = "https://quantapi.51ifind.com"
ACCESS_TOKEN_PATH = "/api/v1/get_access_token"
ACCESS_TOKEN_CACHE_TTL = timedelta(days=6)


class HTTPResponse(Protocol):
    """The subset of a requests response used by the client."""

    status_code: int

    def json(self) -> Any: ...


class HTTPTransport(Protocol):
    """Injectable transport to keep client tests fully offline."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any] | None,
        timeout: float,
    ) -> HTTPResponse: ...


class IFindQuantAPIClient:
    """Make authenticated JSON requests to iFinD's HTTP QuantAPI."""

    def __init__(
        self,
        refresh_token: str,
        *,
        transport: HTTPTransport | None = None,
        timeout_seconds: float = 20.0,
        max_attempts: int = 2,
        retry_delay_seconds: float = 0.5,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not refresh_token.strip():
            raise ValueError("刷新令牌不能为空")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("请求超时时间必须大于 0")
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts < 1
        ):
            raise ValueError("请求次数至少为 1")
        if retry_delay_seconds < 0:
            raise ValueError("重试等待时间不能为负数")

        self._refresh_token = refresh_token
        self._transport = transport if transport is not None else requests.Session()
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._retry_delay_seconds = retry_delay_seconds
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleep = sleep
        self._access_token: str | None = None
        self._access_token_expires_at: datetime | None = None

    def post_json(
        self,
        path: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """POST a JSON request, renewing an access token once if it is rejected."""
        for auth_attempt in range(2):
            response = self._post(
                path,
                headers={
                    "Content-Type": "application/json",
                    "ifindlang": "cn",
                    "access_token": self._get_access_token(),
                },
                payload=payload,
            )
            if response.status_code not in {401, 403}:
                return self._response_json(response)
            self._invalidate_access_token()
            if auth_attempt == 1:
                raise MarketDataProviderAuthenticationError(
                    "行情服务身份验证失败，请检查访问凭据。"
                )

        raise AssertionError("unreachable")

    def _get_access_token(self) -> str:
        if self._has_valid_access_token():
            return self._access_token  # type: ignore[return-value]

        response = self._post(
            ACCESS_TOKEN_PATH,
            headers={
                "Content-Type": "application/json",
                "refresh_token": self._refresh_token,
            },
            payload=None,
        )
        if response.status_code in {401, 403}:
            raise MarketDataProviderAuthenticationError(
                "行情服务身份验证失败，请检查访问凭据。"
            )
        if response.status_code >= 400:
            raise MarketDataProviderUnavailable("行情服务暂时不可用，请稍后重试。")

        invalid_response = False
        try:
            access_token = response.json()["data"]["access_token"]
        except (KeyError, TypeError, ValueError):
            invalid_response = True
            access_token = None
        if invalid_response:
            raise MarketDataProviderResponseError(
                "行情服务返回内容无法识别。"
            )
        if not isinstance(access_token, str) or not access_token:
            raise MarketDataProviderResponseError("行情服务返回内容无法识别。")

        self._access_token = access_token
        self._access_token_expires_at = self._clock() + ACCESS_TOKEN_CACHE_TTL
        return access_token

    def _has_valid_access_token(self) -> bool:
        return (
            self._access_token is not None
            and self._access_token_expires_at is not None
            and self._clock() < self._access_token_expires_at
        )

    def _invalidate_access_token(self) -> None:
        self._access_token = None
        self._access_token_expires_at = None

    def _post(
        self,
        path: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any] | None,
    ) -> HTTPResponse:
        url = f"{IFIND_BASE_URL}{path}"
        for attempt in range(1, self._max_attempts + 1):
            failure: MarketDataProviderUnavailable | MarketDataProviderTimeout | None = None
            try:
                response = self._transport.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout_seconds,
                )
            except requests.Timeout:
                if attempt == self._max_attempts:
                    failure = MarketDataProviderTimeout(
                        "行情服务连接超时，请稍后重试。"
                    )
            except requests.ConnectionError:
                if attempt == self._max_attempts:
                    failure = MarketDataProviderUnavailable(
                        "行情服务暂时不可用，请稍后重试。"
                    )
            except requests.RequestException:
                failure = MarketDataProviderUnavailable(
                    "行情服务暂时不可用，请稍后重试。"
                )
            else:
                if response.status_code == 429:
                    if attempt == self._max_attempts:
                        raise MarketDataProviderRateLimitError(
                            "行情服务请求过于频繁，请稍后再试。"
                        )
                elif 500 <= response.status_code < 600:
                    if attempt == self._max_attempts:
                        raise MarketDataProviderUnavailable(
                            "行情服务暂时不可用，请稍后重试。"
                        )
                else:
                    return response

            if failure is not None:
                raise failure
            if attempt < self._max_attempts:
                self._sleep(self._retry_delay_seconds)

        raise AssertionError("unreachable")

    @staticmethod
    def _response_json(response: HTTPResponse) -> dict[str, Any]:
        if response.status_code >= 400:
            raise MarketDataProviderResponseError("行情服务请求未被接受。")
        invalid_response = False
        try:
            payload = response.json()
        except (TypeError, ValueError):
            invalid_response = True
            payload = None
        if invalid_response:
            raise MarketDataProviderResponseError(
                "行情服务返回内容无法识别。"
            )
        if not isinstance(payload, dict):
            raise MarketDataProviderResponseError("行情服务返回内容无法识别。")
        return payload
