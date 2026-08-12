import pandas as pd
import pytest

from quantlab.analytics.returns import (
    calculate_annualized_return,
    calculate_daily_returns,
    calculate_net_value,
    calculate_total_return,
)


def make_prices() -> pd.Series:
    """创建两个周期均上涨10%的测试价格。"""
    return pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.date_range("2025-01-01", periods=3),
        name="close",
    )


def test_calculate_daily_returns() -> None:
    result = calculate_daily_returns(make_prices())
    expected = pd.Series(
        [0.10, 0.10],
        index=pd.date_range("2025-01-02", periods=2),
        name="daily_return",
    )

    pd.testing.assert_series_equal(result, expected)


def test_calculate_total_return() -> None:
    result = calculate_total_return(make_prices())

    assert result == pytest.approx(0.21)


def test_calculate_net_value() -> None:
    result = calculate_net_value(make_prices())
    expected = pd.Series(
        [1.0, 1.1, 1.21],
        index=pd.date_range("2025-01-01", periods=3),
        name="net_value",
    )

    pd.testing.assert_series_equal(result, expected)


def test_calculate_annualized_return() -> None:
    result = calculate_annualized_return(
        make_prices(),
        periods_per_year=2,
    )

    assert result == pytest.approx(0.21)


def test_returns_reject_missing_price() -> None:
    prices = make_prices()
    prices.iloc[1] = float("nan")

    with pytest.raises(ValueError, match="空值"):
        calculate_daily_returns(prices)


def test_returns_reject_non_positive_price() -> None:
    prices = make_prices()
    prices.iloc[1] = 0

    with pytest.raises(ValueError, match="必须大于0"):
        calculate_total_return(prices)


def test_returns_require_two_prices() -> None:
    prices = pd.Series([100.0])

    with pytest.raises(ValueError, match="至少需要 2 个"):
        calculate_annualized_return(prices)


def test_annualized_return_rejects_invalid_periods_per_year() -> None:
    with pytest.raises(ValueError, match="每年期数必须大于0"):
        calculate_annualized_return(
            make_prices(),
            periods_per_year=0,
        )
