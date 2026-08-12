import pandas as pd
import pytest

from quantlab.ui.trade_review import build_trade_review_view


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


def make_trade_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_id": pd.Series(["T001", "T002"], dtype="string"),
            "trade_datetime": pd.to_datetime(
                [
                    "2025-01-06 10:00:00+08:00",
                    "2025-01-08 10:00:00+08:00",
                ],
                utc=True,
            ).tz_convert("Asia/Shanghai"),
            "symbol": pd.Series(["510300", "510300"], dtype="string"),
            "side": pd.Series(["BUY", "SELL"], dtype="string"),
            "quantity": pd.Series([100, 100], dtype="int64"),
            "price": [2.0, 2.0],
            "fee": [0.5, 0.5],
        }
    )


def test_build_trade_review_view_summarizes_results() -> None:
    view = build_trade_review_view(
        make_market_data(),
        make_trade_data(),
        short_window=2,
        long_window=3,
        initial_cash=1000.0,
        tolerance_trading_days=1,
    )

    assert view.symbol == "510300"
    assert view.final_equity == pytest.approx(999.0)
    assert view.cumulative_return == pytest.approx(-0.001)
    assert view.win_rate == 0.0
    assert view.profit_loss_ratio is None
    assert view.timing_summary["count"].sum() == 2


def test_build_trade_review_view_generates_actionable_notes() -> None:
    view = build_trade_review_view(
        make_market_data(),
        make_trade_data(),
        short_window=2,
        long_window=3,
        initial_cash=1000.0,
        tolerance_trading_days=1,
    )

    assert any("note" in note for note in view.recommendations)
