import pandas as pd
import pytest

from quantlab.ui.comparison import build_three_way_comparison_view


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
                    "2025-01-02 10:00:00+08:00",
                    "2025-01-06 10:00:00+08:00",
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


def test_build_three_way_comparison_view_normalizes_curves() -> None:
    view = build_three_way_comparison_view(
        make_market_data(),
        make_trade_data(),
        short_window=2,
        long_window=3,
        initial_cash=1000.0,
        commission_rate=0.0,
        minimum_commission=0.0,
        slippage_rate=0.0,
        sell_tax_rate=0.0,
        lot_size=10,
        periods_per_year=7,
    )

    assert view.symbol == "510300"
    assert view.normalized_curves["approach"].unique().tolist() == [
        "买入持有",
        "双均线策略",
        "个人真实交易",
    ]
    assert len(view.normalized_curves) == 24
    assert view.display_metrics["approach"].tolist() == [
        "买入持有",
        "双均线策略",
        "个人真实交易",
    ]
    personal = view.display_metrics.iloc[2]
    assert personal["final_equity"] == pytest.approx(999.0)


def test_build_three_way_comparison_view_exposes_review() -> None:
    view = build_three_way_comparison_view(
        make_market_data(),
        make_trade_data(),
        short_window=2,
        long_window=3,
        initial_cash=1000.0,
        commission_rate=0.0,
        minimum_commission=0.0,
        slippage_rate=0.0,
        sell_tax_rate=0.0,
        lot_size=10,
        periods_per_year=7,
    )

    conclusions = view.review.conclusions
    assert conclusions["strategy_beats_buy_and_hold"] is True
    assert conclusions["personal_beats_strategy"] is False
