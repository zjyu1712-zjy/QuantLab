from math import sqrt

import pandas as pd
import pytest

from quantlab.analytics.risk import (
    calculate_annualized_volatility,
    calculate_drawdown_curve,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
)


def make_daily_returns() -> pd.Series:
    return pd.Series(
        [0.01, -0.01, 0.02],
        index=pd.date_range("2025-01-02", periods=3),
        name="daily_return",
    )


def test_calculate_annualized_volatility() -> None:
    daily_returns = make_daily_returns()
    expected = daily_returns.std(ddof=1) * sqrt(252)

    result = calculate_annualized_volatility(daily_returns)

    assert result == pytest.approx(expected)


def test_volatility_accepts_custom_periods_per_year() -> None:
    daily_returns = make_daily_returns()
    expected = daily_returns.std(ddof=1) * sqrt(4)

    result = calculate_annualized_volatility(
        daily_returns,
        periods_per_year=4,
    )

    assert result == pytest.approx(expected)


def test_volatility_rejects_missing_return() -> None:
    daily_returns = make_daily_returns()
    daily_returns.iloc[1] = float("nan")

    with pytest.raises(ValueError, match="不能包含空值"):
        calculate_annualized_volatility(daily_returns)


def test_volatility_requires_two_returns() -> None:
    daily_returns = pd.Series([0.01])

    with pytest.raises(ValueError, match="至少需要 2 个"):
        calculate_annualized_volatility(daily_returns)


def test_volatility_rejects_invalid_periods_per_year() -> None:
    with pytest.raises(ValueError, match="每年期数必须大于0"):
        calculate_annualized_volatility(
            make_daily_returns(),
            periods_per_year=0,
        )


def test_calculate_sharpe_ratio() -> None:
    daily_returns = make_daily_returns()
    expected = (
        daily_returns.mean()
        / daily_returns.std(ddof=1)
        * sqrt(252)
    )

    result = calculate_sharpe_ratio(daily_returns)

    assert result == pytest.approx(expected)


def test_sharpe_ratio_uses_compound_risk_free_rate() -> None:
    daily_returns = make_daily_returns()
    daily_risk_free_rate = (1.04 ** (1 / 4)) - 1
    expected = (
        (daily_returns - daily_risk_free_rate).mean()
        / daily_returns.std(ddof=1)
        * sqrt(4)
    )

    result = calculate_sharpe_ratio(
        daily_returns,
        annual_risk_free_rate=0.04,
        periods_per_year=4,
    )

    assert result == pytest.approx(expected)


def test_sharpe_ratio_rejects_zero_volatility() -> None:
    daily_returns = pd.Series([0.01, 0.01, 0.01])

    with pytest.raises(ValueError, match="波动率为0"):
        calculate_sharpe_ratio(daily_returns)


def test_sharpe_ratio_rejects_invalid_risk_free_rate() -> None:
    with pytest.raises(ValueError, match="必须大于-100%"):
        calculate_sharpe_ratio(
            make_daily_returns(),
            annual_risk_free_rate=-1.0,
        )


def test_sharpe_ratio_rejects_invalid_periods_per_year() -> None:
    with pytest.raises(ValueError, match="每年期数必须大于0"):
        calculate_sharpe_ratio(
            make_daily_returns(),
            periods_per_year=0,
        )


def make_net_value() -> pd.Series:
    return pd.Series(
        [1.0, 1.2, 0.9, 1.08, 1.3],
        index=pd.date_range("2025-01-01", periods=5),
        name="net_value",
    )


def test_calculate_drawdown_curve() -> None:
    result = calculate_drawdown_curve(make_net_value())
    expected = pd.Series(
        [0.0, 0.0, -0.25, -0.10, 0.0],
        index=pd.date_range("2025-01-01", periods=5),
        name="drawdown",
    )

    pd.testing.assert_series_equal(result, expected)


def test_calculate_max_drawdown() -> None:
    result = calculate_max_drawdown(make_net_value())

    assert result == pytest.approx(-0.25)


def test_rising_net_value_has_zero_drawdown() -> None:
    net_value = pd.Series([1.0, 1.1, 1.2])

    assert calculate_max_drawdown(net_value) == pytest.approx(0.0)


def test_drawdown_rejects_missing_net_value() -> None:
    net_value = make_net_value()
    net_value.iloc[2] = float("nan")

    with pytest.raises(ValueError, match="不能包含空值"):
        calculate_drawdown_curve(net_value)


def test_drawdown_rejects_non_positive_net_value() -> None:
    net_value = make_net_value()
    net_value.iloc[2] = 0.0

    with pytest.raises(ValueError, match="净值必须大于0"):
        calculate_max_drawdown(net_value)
