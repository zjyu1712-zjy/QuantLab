"""Licensed iFinD QuantAPI integration coverage.

This module is deliberately opt-in so the normal test suite remains offline.
"""

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
    """A licensed, explicitly enabled run returns QuantLab's standard columns."""
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
    assert result.columns.tolist() == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
