from math import sqrt

import pandas as pd
import pytest

from quantlab.analytics.performance import analyze_performance


def make_performance_prices() -> pd.Series:
    """三个周期收益率依次为10%、-10%和10%。"""
    return pd.Series(
        [100.0, 110.0, 99.0, 108.9],
        index=pd.date_range("2025-01-01", periods=4),
        name="close",
    )


def test_analyze_performance_end_to_end() -> None:
    prices = make_performance_prices()
    result = analyze_performance(
        prices,
        periods_per_year=3,
    )

    expected_daily_returns = pd.Series(
        [0.10, -0.10, 0.10],
        index=pd.date_range("2025-01-02", periods=3),
        name="daily_return",
    )
    expected_net_value = pd.Series(
        [1.0, 1.1, 0.99, 1.089],
        index=pd.date_range("2025-01-01", periods=4),
        name="net_value",
    )
    expected_drawdown = pd.Series(
        [0.0, 0.0, -0.10, -0.01],
        index=pd.date_range("2025-01-01", periods=4),
        name="drawdown",
    )
    expected_volatility = expected_daily_returns.std(ddof=1) * sqrt(3)
    expected_sharpe = (
        expected_daily_returns.mean()
        / expected_daily_returns.std(ddof=1)
        * sqrt(3)
    )

    pd.testing.assert_series_equal(
        result.daily_returns,
        expected_daily_returns,
    )
    pd.testing.assert_series_equal(
        result.net_value,
        expected_net_value,
    )
    pd.testing.assert_series_equal(
        result.drawdown,
        expected_drawdown,
    )
    assert result.cumulative_return == pytest.approx(0.089)
    assert result.annualized_return == pytest.approx(0.089)
    assert result.annualized_volatility == pytest.approx(
        expected_volatility
    )
    assert result.max_drawdown == pytest.approx(-0.10)
    assert result.sharpe_ratio == pytest.approx(expected_sharpe)


def test_analyze_performance_propagates_undefined_sharpe() -> None:
    constant_prices = pd.Series([100.0, 100.0, 100.0])

    with pytest.raises(ValueError, match="无法计算夏普比率"):
        analyze_performance(constant_prices)
