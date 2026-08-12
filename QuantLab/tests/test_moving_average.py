import pandas as pd
import pytest

from quantlab.strategies import (
    generate_dual_moving_average_signals,
)


def make_market_data(prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-01",
                periods=len(prices),
            ),
            "symbol": pd.Series(
                ["510300"] * len(prices),
                dtype="string",
            ),
            "open": prices,
            "high": [price + 0.1 for price in prices],
            "low": [price - 0.1 for price in prices],
            "close": prices,
            "volume": [1000] * len(prices),
        }
    )


def test_dual_moving_average_generates_cross_signals() -> None:
    market_data = make_market_data(
        [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]
    )

    result = generate_dual_moving_average_signals(
        market_data,
        short_window=2,
        long_window=3,
    )

    assert result["sma_2"].tolist() == pytest.approx(
        [float("nan"), 2.5, 1.5, 1.5, 2.5, 2.5, 1.5],
        nan_ok=True,
    )
    assert result["sma_3"].tolist() == pytest.approx(
        [
            float("nan"),
            float("nan"),
            2.0,
            5 / 3,
            2.0,
            7 / 3,
            2.0,
        ],
        nan_ok=True,
    )
    assert result.loc[4, "signal"] == "BUY"
    assert result.loc[6, "signal"] == "SELL"
    assert result["signal"].dropna().tolist() == ["BUY", "SELL"]


def test_moving_average_does_not_use_future_prices() -> None:
    first_five = make_market_data([3.0, 2.0, 1.0, 2.0, 3.0])
    extended = make_market_data(
        [3.0, 2.0, 1.0, 2.0, 3.0, 100.0]
    )

    short_result = generate_dual_moving_average_signals(
        first_five,
        short_window=2,
        long_window=3,
    )
    extended_result = generate_dual_moving_average_signals(
        extended,
        short_window=2,
        long_window=3,
    ).iloc[:5]

    pd.testing.assert_frame_equal(
        short_result,
        extended_result.reset_index(drop=True),
    )


def test_moving_average_short_history_has_no_signal() -> None:
    market_data = make_market_data([1.0, 2.0])

    result = generate_dual_moving_average_signals(
        market_data,
        short_window=2,
        long_window=3,
    )

    assert result["sma_3"].isna().all()
    assert result["signal"].isna().all()


@pytest.mark.parametrize(
    ("short_window", "long_window", "message"),
    [
        (0, 3, "必须是正整数"),
        (2.5, 3, "必须是正整数"),
        (3, 3, "必须小于"),
        (4, 3, "必须小于"),
    ],
)
def test_moving_average_rejects_invalid_windows(
    short_window: int,
    long_window: int,
    message: str,
) -> None:
    market_data = make_market_data([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match=message):
        generate_dual_moving_average_signals(
            market_data,
            short_window=short_window,
            long_window=long_window,
        )
