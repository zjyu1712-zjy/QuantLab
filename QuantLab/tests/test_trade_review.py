import pandas as pd
import pytest

from quantlab.trade_review import review_trades_against_signals


def make_market_data() -> pd.DataFrame:
    prices = [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0]
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=len(prices)),
            "symbol": pd.Series(["510300"] * len(prices), dtype="string"),
            "open": prices,
            "high": [price + 0.1 for price in prices],
            "low": [price - 0.1 for price in prices],
            "close": prices,
            "volume": [1000] * len(prices),
        }
    )


def make_trade_data(
    buy_date: str = "2025-01-06 10:00:00+08:00",
    sell_date: str = "2025-01-08 10:00:00+08:00",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_id": pd.Series(["T001", "T002"], dtype="string"),
            "trade_datetime": pd.to_datetime(
                [buy_date, sell_date],
                utc=True,
            ).tz_convert("Asia/Shanghai"),
            "symbol": pd.Series(["510300", "510300"], dtype="string"),
            "side": pd.Series(["BUY", "SELL"], dtype="string"),
            "quantity": pd.Series([100, 100], dtype="int64"),
            "price": [2.0, 2.0],
            "fee": [0.5, 0.5],
        }
    )


def test_review_trades_matches_on_expected_execution_dates() -> None:
    review = review_trades_against_signals(
        make_market_data(),
        make_trade_data(),
        2,
        3,
        initial_cash=1000.0,
        tolerance_trading_days=1,
    )

    assert review.trade_alignment["timing_status"].tolist() == [
        "ON_TIME",
        "ON_TIME",
    ]
    assert review.missed_signals.empty
    assert review.timing_counts["ON_TIME"] == 2
    assert review.total_fees == pytest.approx(1.0)
    assert review.fee_to_turnover_rate == pytest.approx(0.0025)


def test_review_trades_classifies_early_and_late() -> None:
    review = review_trades_against_signals(
        make_market_data(),
        make_trade_data(
            buy_date="2025-01-05 10:00:00+08:00",
            sell_date="2025-01-08 10:00:00+08:00",
        ),
        2,
        3,
        initial_cash=1000.0,
        tolerance_trading_days=1,
    )

    assert review.trade_alignment["timing_status"].tolist() == [
        "EARLY",
        "ON_TIME",
    ]
    assert review.trade_alignment[
        "offset_trading_days"
    ].tolist() == [-1, 0]


def test_review_trades_marks_outside_model_and_missed_signal() -> None:
    review = review_trades_against_signals(
        make_market_data(),
        make_trade_data(
            buy_date="2025-01-01 10:00:00+08:00",
            sell_date="2025-01-02 10:00:00+08:00",
        ),
        2,
        3,
        initial_cash=1000.0,
        tolerance_trading_days=0,
    )

    assert review.timing_counts["NO_MATCH"] == 2
    assert review.timing_counts["MISSED_SIGNAL"] == 2


def test_review_trades_rejects_invalid_tolerance() -> None:
    with pytest.raises(ValueError, match="非负整数"):
        review_trades_against_signals(
            make_market_data(),
            make_trade_data(),
            2,
            3,
            initial_cash=1000.0,
            tolerance_trading_days=-1,
        )
