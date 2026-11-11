# iFinD QuantAPI Market Data Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a secure, replaceable iFinD HTTP QuantAPI provider so QuantLab can fetch ETF daily history through the existing `MarketDataService`, while retaining AkShare and keeping every default unit test offline.

**Architecture:** A small HTTP client owns token exchange, bounded retries, timeouts, and safe error translation. `IFindETFProvider` implements the existing `MarketDataProvider` contract and converts iFinD history responses into QuantLab's standard columns. The provider factory selects `ifind`, `akshare`, or `auto` from server-side environment settings; once iFinD is selected, request failures are reported and are never silently routed to AkShare.

**Tech Stack:** Python 3.12, pandas, requests, pytest, Streamlit, existing QuantLab provider/service/storage layers.

**Global Constraints:**

- Use iFinD's HTTP QuantAPI only; no desktop SDK process or Super Command runtime is required by QuantLab.
- Read `IFIND_REFRESH_TOKEN` only from the server environment. Never put a real token in source, test fixtures, logs, exceptions, screenshots, or Git.
- Keep all existing analytics, strategies, backtests, and storage contracts unchanged.
- Keep Streamlit pages dependent on `MarketDataService`, never on `IFindQuantAPIClient` or `IFindETFProvider` directly.
- Default tests must not access the internet. The real integration test is opt-in.
- Support daily ETF history and `none`, `qfq`, and `hfq` adjustments in this increment.
- Do not silently fall back from a selected iFinD request to AkShare. `auto` chooses a provider before the request based on configuration availability only.
- Show ordinary users natural-language failures; keep technical details in server logs.

---

## Task 1: Build the authenticated HTTP client

**Files:**

- Create: `src/quantlab/data/providers/ifind_client.py`
- Modify: `src/quantlab/data/providers/base.py`
- Modify: `requirements.txt`
- Create: `tests/test_ifind_client.py`

- [ ] **Step 1: Add failing tests for token exchange and token reuse**

Create a deterministic fake transport and test that the first API request exchanges the refresh token, while the second API request reuses the cached access token.

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from quantlab.data.providers.ifind_client import IFindQuantAPIClient


@dataclass
class FakeResponse:
    status_code: int
    payload: dict[str, Any]

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeTransport:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def test_client_exchanges_and_reuses_access_token() -> None:
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

    client.post_json("/api/v1/cmd_history_quotation", {"codes": "510300.SH"})
    client.post_json("/api/v1/cmd_history_quotation", {"codes": "510300.SH"})

    assert len(transport.calls) == 3
    assert transport.calls[0]["headers"]["refresh_token"] == "refresh-secret"
    assert transport.calls[1]["headers"]["access_token"] == "access-secret"
    assert transport.calls[2]["headers"]["access_token"] == "access-secret"
```

- [ ] **Step 2: Add failing tests for timeout, bounded retry, authentication failure, and secret redaction**

Cover these cases in the same file:

```python
import requests
import pytest

from quantlab.data.providers.base import (
    MarketDataProviderAuthenticationError,
    MarketDataProviderTimeout,
)


class TimeoutTransport:
    def __init__(self) -> None:
        self.call_count = 0

    def post(self, *args: Any, **kwargs: Any) -> FakeResponse:
        self.call_count += 1
        raise requests.Timeout("refresh-secret must never appear")


def test_client_retries_timeout_only_to_configured_limit() -> None:
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


def test_client_translates_rejected_refresh_token_without_leaking_it() -> None:
    transport = FakeTransport([FakeResponse(401, {"message": "refresh-secret"})])
    client = IFindQuantAPIClient(
        "refresh-secret",
        transport=transport,
        sleep=lambda _: None,
    )

    with pytest.raises(MarketDataProviderAuthenticationError) as captured:
        client.post_json("/api/v1/cmd_history_quotation", {})

    assert "refresh-secret" not in str(captured.value)
```

Also test HTTP 429 maps to `MarketDataProviderRateLimitError`, HTTP 5xx retries at most `max_attempts`, invalid JSON maps to `MarketDataProviderResponseError`, and a 401 from a data endpoint forces exactly one access-token refresh before failing.

- [ ] **Step 3: Run the new client tests and confirm the expected import failure**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_ifind_client.py -v
```

Expected: FAIL because `ifind_client.py` and the new exception classes do not exist.

- [ ] **Step 4: Add provider-specific error subclasses**

Append these subclasses to `src/quantlab/data/providers/base.py` without changing existing exception inheritance:

```python
class MarketDataProviderAuthenticationError(MarketDataProviderUnavailable):
    """Raised when provider credentials are absent, expired, or rejected."""


class MarketDataProviderRateLimitError(MarketDataProviderUnavailable):
    """Raised when the provider refuses a request because of quota limits."""
```

- [ ] **Step 5: Implement the HTTP client with injectable transport**

Create `src/quantlab/data/providers/ifind_client.py` with these public contracts:

```python
from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
import time

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
    status_code: int

    def json(self) -> Any: ...


class HTTPTransport(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any] | None,
        timeout: float,
    ) -> HTTPResponse: ...


class IFindQuantAPIClient:
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
    ) -> None: ...

    def post_json(
        self,
        path: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]: ...
```

Implementation rules:

1. Reject an empty refresh token and invalid timeout/attempt values in `__init__`.
2. Use `requests.Session()` when no transport is injected.
3. Exchange the refresh token by POSTing to `ACCESS_TOKEN_PATH` with `Content-Type: application/json` and `refresh_token` headers.
4. Read the access token only from `response.json()["data"]["access_token"]`.
5. Cache the token in memory for six days, a conservative margin inside iFinD's documented seven-day validity.
6. Add `Content-Type: application/json`, `ifindlang: cn`, and `access_token` to data requests.
7. Retry only timeouts, connection failures, 429, and 5xx responses; never exceed `max_attempts`.
8. On a data-request 401/403, invalidate the cached access token and attempt one fresh token exchange.
9. Raise only friendly provider exceptions. Do not interpolate response bodies, headers, tokens, or raw request payloads into exception messages.

- [ ] **Step 6: Declare requests as a direct dependency**

Add this line to `requirements.txt`:

```text
requests>=2.32,<3
```

- [ ] **Step 7: Run the client tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_ifind_client.py -v
```

Expected: all client tests PASS.

- [ ] **Step 8: Commit Task 1**

```bash
git add QuantLab/src/quantlab/data/providers/base.py \
  QuantLab/src/quantlab/data/providers/ifind_client.py \
  QuantLab/tests/test_ifind_client.py \
  QuantLab/requirements.txt
git commit -m "feat: add authenticated ifind quantapi client"
```

---

## Task 2: Implement ETF history mapping and response normalization

**Files:**

- Create: `src/quantlab/data/providers/ifind.py`
- Create: `tests/test_ifind_provider.py`

- [ ] **Step 1: Add failing tests for ETF code and adjustment mapping**

Create `tests/test_ifind_provider.py` and assert these exact mappings:

```python
import pytest

from quantlab.data.providers.ifind import (
    adjustment_to_ifind_cps,
    to_ifind_etf_code,
)


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
```

The suffix mapping only identifies the exchange required by the API. It must not claim that the code exists; an empty provider response remains the existence check.

- [ ] **Step 2: Add failing tests for request construction and standard DataFrame output**

Use a fake client with this representative QuantAPI response:

```python
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


class FakeIFindClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    def post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append((path, payload))
        return self.response


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
```

- [ ] **Step 3: Add failing tests for provider error boundaries**

Test all of the following:

- `period != "daily"` raises `ValueError`.
- An unsupported adjustment raises `ValueError`.
- `errorcode != 0` raises `MarketDataProviderResponseError` without including the raw payload.
- Missing `tables`, more than one instrument table, missing `time`, unequal column lengths, or a missing core field raises `MarketDataProviderResponseError`.
- An empty `tables` list or an empty `time` list raises `MarketDataProviderResponseError("未找到该 ETF 的行情数据，请检查代码或日期范围。")`.
- Numeric strings are accepted for OHLCV and left for the existing standardization layer to convert.

- [ ] **Step 4: Run the provider tests and confirm failure**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_ifind_provider.py -v
```

Expected: FAIL because `ifind.py` does not exist.

- [ ] **Step 5: Implement the provider**

Create `src/quantlab/data/providers/ifind.py` with this public surface:

```python
from __future__ import annotations

from datetime import date
from typing import Any, Protocol

import pandas as pd

from .base import (
    STANDARD_PROVIDER_COLUMNS,
    MarketDataProvider,
    MarketDataProviderResponseError,
)

HISTORY_PATH = "/api/v1/cmd_history_quotation"
IFIND_INDICATORS = "open,high,low,close,volume"
ADJUSTMENT_TO_CPS = {"none": "1", "qfq": "2", "hfq": "3"}


class IFindHistoryClient(Protocol):
    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]: ...


def to_ifind_etf_code(symbol: str) -> str: ...


def adjustment_to_ifind_cps(adjustment: str) -> str: ...


class IFindETFProvider(MarketDataProvider):
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
    ) -> pd.DataFrame: ...
```

Use these exchange rules for the first沪深 ETF increment:

```python
if symbol.startswith(("50", "51", "52", "56", "58")):
    return f"{symbol}.SH"
if symbol.startswith(("15", "16")):
    return f"{symbol}.SZ"
raise ValueError("第一阶段仅支持沪深 ETF 代码")
```

The private response parser must:

1. Check top-level `errorcode` before accessing data.
2. Require exactly one non-empty table for the requested code; translate an empty result into the standard no-data provider error.
3. Read dates from `time` and values from the nested `table` mapping.
4. Verify every core field is a list with the same length as `time`.
5. Return only the six standard fields in standard order; optional iFinD fields are intentionally ignored in this increment.
6. Never include raw response data in a raised user-facing exception.

- [ ] **Step 6: Run provider and existing market service tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_ifind_provider.py \
  tests/test_market_service.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add QuantLab/src/quantlab/data/providers/ifind.py \
  QuantLab/tests/test_ifind_provider.py
git commit -m "feat: add ifind etf history provider"
```

---

## Task 3: Add server-side provider selection

**Files:**

- Modify: `src/quantlab/data/providers/__init__.py`
- Modify: `src/quantlab/data/__init__.py`
- Create: `tests/test_provider_factory.py`
- Modify: `tests/test_market_formats.py`

- [ ] **Step 1: Add failing factory tests for all three modes**

Create `tests/test_provider_factory.py`:

```python
import pytest

from quantlab.data.providers import get_market_data_provider
from quantlab.data.providers.akshare import AkShareETFProvider
from quantlab.data.providers.base import MarketDataProviderUnavailable
from quantlab.data.providers.ifind import IFindETFProvider


def test_auto_without_token_selects_akshare() -> None:
    assert isinstance(get_market_data_provider(env={}), AkShareETFProvider)


def test_auto_with_token_selects_ifind() -> None:
    provider = get_market_data_provider(
        env={"IFIND_REFRESH_TOKEN": "test-refresh-token"},
    )
    assert isinstance(provider, IFindETFProvider)


def test_explicit_akshare_ignores_ifind_token() -> None:
    provider = get_market_data_provider(
        env={
            "MARKET_DATA_PROVIDER": "akshare",
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
```

Replace the ambient-environment assertion in `tests/test_market_formats.py` with an explicit `env={}` call so a developer's real token cannot change unit-test behavior.

- [ ] **Step 2: Run the factory tests and confirm signature failures**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_provider_factory.py \
  tests/test_market_formats.py -v
```

Expected: FAIL because the factory does not accept `env` and does not construct iFinD.

- [ ] **Step 3: Implement deterministic provider selection**

Change `src/quantlab/data/providers/__init__.py` to expose:

```python
from __future__ import annotations

from collections.abc import Mapping
import os

from .akshare import AkShareETFProvider
from .base import MarketDataProvider, MarketDataProviderUnavailable
from .ifind import IFindETFProvider
from .ifind_client import HTTPTransport, IFindQuantAPIClient

VALID_PROVIDER_MODES = {"auto", "akshare", "ifind"}


def get_market_data_provider(
    env: Mapping[str, str] | None = None,
    *,
    ifind_transport: HTTPTransport | None = None,
) -> MarketDataProvider:
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
```

Export the new provider, client class, and safe exception types through `providers.__all__`. Add the provider and exception classes to the existing `quantlab.data` exports without removing any current public name. Do not expose credential-bearing client instances at package level.

- [ ] **Step 4: Prove selection happens once and has no request-level fallback**

Add one test whose fake iFinD client raises `MarketDataProviderTimeout`, then assert the exception reaches the caller and no AkShare provider is constructed or called. This protects the explicit no-silent-fallback decision.

- [ ] **Step 5: Run factory and package-import tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_provider_factory.py \
  tests/test_market_formats.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add QuantLab/src/quantlab/data/providers/__init__.py \
  QuantLab/src/quantlab/data/__init__.py \
  QuantLab/tests/test_provider_factory.py \
  QuantLab/tests/test_market_formats.py
git commit -m "feat: configure market data provider selection"
```

---

## Task 4: Verify service integration and make the selected source visible

**Files:**

- Create: `tests/test_ifind_market_service.py`
- Modify: `src/quantlab/ui/market_data.py`
- Modify: `pages/1_数据管理.py`
- Modify: `tests/test_ui_market_data.py`

- [ ] **Step 1: Add a failing end-to-end service test with an in-memory provider**

Test the existing service boundary, not HTTP again:

```python
from datetime import date

import pandas as pd

from quantlab.data.market_service import MarketDataService
from quantlab.data.providers.ifind import IFindETFProvider


def test_ifind_provider_flows_through_service_and_cache(tmp_path) -> None:
    client = FakeIFindClient(IFIND_HISTORY_RESPONSE)
    service = MarketDataService(
        IFindETFProvider(client),
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
    )

    first = service.fetch_etf_history("510300", date(2025, 1, 2), date(2025, 1, 3))
    second = service.fetch_etf_history("510300", date(2025, 1, 2), date(2025, 1, 3))

    assert first.metadata.provider == "ifind"
    assert first.cache_status == "miss"
    assert second.cache_status == "hit"
    assert len(client.calls) == 1
    assert second.data.equals(first.data)
```

Also assert metadata records `symbol`, `period`, `adjustment`, requested dates, actual dates, fetch time, and data version using the existing schema.

- [ ] **Step 2: Add failing UI label tests**

Add to `tests/test_ui_market_data.py`:

```python
from quantlab.ui.market_data import market_source_label


def test_market_source_label_for_ifind_fetch() -> None:
    assert market_source_label("ifind", "miss") == "iFinD"


def test_market_source_label_for_cached_ifind_data() -> None:
    assert market_source_label("ifind", "hit") == "本地数据（iFinD）"


def test_market_source_label_for_akshare_fetch() -> None:
    assert market_source_label("akshare", "miss") == "AkShare"
```

- [ ] **Step 3: Run the integration and UI tests and confirm the missing helper failure**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_ifind_market_service.py \
  tests/test_ui_market_data.py -v
```

Expected: FAIL because `market_source_label` does not exist.

- [ ] **Step 4: Implement the source label helper**

Add to `src/quantlab/ui/market_data.py`:

```python
PROVIDER_LABELS = {"akshare": "AkShare", "ifind": "iFinD"}


def market_source_label(provider: str, cache_status: str) -> str:
    label = PROVIDER_LABELS.get(provider, provider)
    if cache_status == "hit":
        return f"本地数据（{label}）"
    return label
```

- [ ] **Step 5: Remove the hard-coded AkShare success label from the page**

In `pages/1_数据管理.py`, import `market_source_label` and replace the current hard-coded source expression with:

```python
source_text = market_source_label(
    result.metadata.provider,
    result.cache_status,
)
```

Keep the existing progress messages and existing `MarketDataProviderError` handling. Do not import the iFinD client or provider into the Streamlit page.

- [ ] **Step 6: Run the focused tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_ifind_market_service.py \
  tests/test_ui_market_data.py \
  tests/test_market_service.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```bash
git add QuantLab/src/quantlab/ui/market_data.py \
  QuantLab/pages/1_数据管理.py \
  QuantLab/tests/test_ifind_market_service.py \
  QuantLab/tests/test_ui_market_data.py
git commit -m "feat: connect ifind provider to market data workflow"
```

---

## Task 5: Add secure configuration guidance and opt-in integration coverage

**Files:**

- Create: `.env.example`
- Create: `tests/test_ifind_integration.py`
- Modify: `README.md`

- [ ] **Step 1: Add an opt-in real-interface integration test**

Create `tests/test_ifind_integration.py`:

```python
from __future__ import annotations

from datetime import date, timedelta
import os

import pytest

from quantlab.data.providers import get_market_data_provider

RUN_REAL = os.getenv("RUN_IFIND_INTEGRATION") == "1"
HAS_TOKEN = bool(os.getenv("IFIND_REFRESH_TOKEN", "").strip())

pytestmark = pytest.mark.skipif(
    not (RUN_REAL and HAS_TOKEN),
    reason="set RUN_IFIND_INTEGRATION=1 and IFIND_REFRESH_TOKEN to run",
)


def test_ifind_real_etf_daily_history() -> None:
    provider = get_market_data_provider(
        env={
            "MARKET_DATA_PROVIDER": "ifind",
            "IFIND_REFRESH_TOKEN": os.environ["IFIND_REFRESH_TOKEN"],
        }
    )
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=14)

    result = provider.fetch_etf_history("510300", start, end)

    assert not result.empty
    assert result.columns.tolist() == ["date", "open", "high", "low", "close", "volume"]
```

This test is expected to be skipped in normal CI and local unit-test runs.

- [ ] **Step 2: Add a safe configuration example**

Create `.env.example` containing names and non-secret examples only:

```text
# auto: use iFinD when IFIND_REFRESH_TOKEN is configured; otherwise use AkShare
MARKET_DATA_PROVIDER=auto

# Keep the real value in the server environment. Never commit it.
IFIND_REFRESH_TOKEN=

# Only needed when deliberately running the live integration test.
RUN_IFIND_INTEGRATION=0
```

Confirm `.gitignore` ignores `.env`, `.env.local`, and `.streamlit/secrets.toml`, while allowing `.env.example` to be committed.

- [ ] **Step 3: Document local and server operation**

Add an `iFinD QuantAPI` section to `README.md` containing:

1. `MARKET_DATA_PROVIDER=auto|akshare|ifind` behavior.
2. A statement that the refresh token belongs in the server environment, not in the browser or repository.
3. The normal local start command:

   ```bash
   cd <project-root>/QuantLab
   source .venv/bin/activate
   PYTHONPATH=src streamlit run app.py
   ```

4. The opt-in integration-test command, written so the token value is supplied by the shell environment and never echoed in documentation:

   ```bash
   RUN_IFIND_INTEGRATION=1 PYTHONPATH=src \
     .venv/bin/python -m pytest tests/test_ifind_integration.py -v
   ```

5. Current limitations:沪深 ETF daily history only; response-shape verification requires a licensed account; no silent provider fallback; shared website use requires appropriate iFinD authorization.
6. Future commercial providers belong under `src/quantlab/data/providers/` and must implement `MarketDataProvider`; UI and analytics remain unchanged.

- [ ] **Step 4: Run the offline full test suite**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -v
```

Expected: all tests PASS and `test_ifind_real_etf_daily_history` is SKIPPED.

- [ ] **Step 5: Check that no likely credential is tracked**

Run:

```bash
git grep -n -E 'IFIND_REFRESH_TOKEN=.+|refresh-secret|access-secret' -- . \
  ':(exclude)QuantLab/tests/test_ifind_client.py' \
  ':(exclude)QuantLab/tests/test_provider_factory.py'
```

Expected: no output. The two excluded test files use explicit fake secrets only.

- [ ] **Step 6: Run the live integration test only after credentials are supplied**

Do not run this step before the user has configured a licensed refresh token in the local/server environment. When available, run the command from Step 3 and verify the provider's actual response shape against the parser. If the official response shape differs from the mocked documented shape, add a sanitized fixture and update only the private parser plus tests; do not change the provider or service contracts.

- [ ] **Step 7: Commit Task 5**

```bash
git add QuantLab/.env.example \
  QuantLab/.gitignore \
  QuantLab/README.md \
  QuantLab/tests/test_ifind_integration.py
git commit -m "docs: add ifind configuration and integration check"
```

---

## Final Verification

- [ ] Run the complete offline suite twice to detect environment leakage:

```bash
MARKET_DATA_PROVIDER=akshare PYTHONPATH=src .venv/bin/python -m pytest -q
env -u MARKET_DATA_PROVIDER -u IFIND_REFRESH_TOKEN \
  PYTHONPATH=src .venv/bin/python -m pytest -q
```

- [ ] Start Streamlit without an iFinD token and verify the automatic-fetch page still uses AkShare normally.

- [ ] Start Streamlit with `MARKET_DATA_PROVIDER=ifind` but no token and verify the page shows a short natural-language configuration error without a traceback or credential field.

- [ ] After a licensed token is available, start with `MARKET_DATA_PROVIDER=ifind`, fetch one short 510300 interval, verify source label `iFinD`, local cache reuse, standard preview columns, and compatibility with investment analysis.

- [ ] Inspect `git diff --check` and `git status --short`; confirm only intended files changed and no generated data/cache files are staged.

## Official References

- iFinD QuantAPI HTTP deployment: <https://quantapi.51ifind.com/gwstatic/static/ds_web/quantapi-web/help-center/deploy.html>
- iFinD QuantAPI Python examples: <https://quantapi.51ifind.com/gwstatic/static/ds_web/quantapi-web/example.html>
- iFinD QuantAPI interface manual: <https://quantapi.51ifind.com/gwstatic/static/ds_web/quantapi-web/help-center/manual.html>
