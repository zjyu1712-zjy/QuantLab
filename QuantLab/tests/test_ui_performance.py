from datetime import date

import pandas as pd
import pytest

from quantlab.ui.performance import build_market_performance_view


def make_market_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5),
            "symbol": ["510300"] * 5,
            "close": [100.0, 110.0, 99.0, 108.9, 119.79],
        }
    )


def test_build_market_performance_view() -> None:
    view = build_market_performance_view(
        make_market_data(),
        periods_per_year=4,
    )

    assert view.symbol == "510300"
    assert view.observation_count == 5
    assert view.start_date == date(2025, 1, 1)
    assert view.end_date == date(2025, 1, 5)
    assert view.analysis.cumulative_return == pytest.approx(0.1979)
    assert list(view.detail.columns) == [
        "date",
        "symbol",
        "close",
        "daily_return",
        "net_value",
        "drawdown",
    ]
    assert pd.isna(view.detail.loc[0, "daily_return"])


def test_build_market_performance_view_applies_date_range() -> None:
    view = build_market_performance_view(
        make_market_data(),
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 4),
        periods_per_year=2,
    )

    assert view.observation_count == 3
    assert view.start_date == date(2025, 1, 2)
    assert view.end_date == date(2025, 1, 4)


def test_build_market_performance_view_rejects_short_range() -> None:
    with pytest.raises(ValueError, match="至少需要 3 个"):
        build_market_performance_view(
            make_market_data(),
            start_date=date(2025, 1, 4),
        )
