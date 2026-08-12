import pandas as pd
import pytest

from quantlab.comparison import compare_three_approaches
from quantlab.config.settings import AnalyticsConfig, BacktestConfig
from quantlab.review import review_comparison


def make_market_data() -> pd.DataFrame:
    prices = [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0]
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=len(prices)),
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


def make_trade_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_id": pd.Series(
                ["T001", "T002", "T003", "T004"],
                dtype="string",
            ),
            "trade_datetime": pd.to_datetime(
                [
                    "2025-01-02 10:00:00+08:00",
                    "2025-01-04 10:00:00+08:00",
                    "2025-01-05 10:00:00+08:00",
                    "2025-01-06 10:00:00+08:00",
                ],
                utc=True,
            ).tz_convert("Asia/Shanghai"),
            "symbol": pd.Series(["510300"] * 4, dtype="string"),
            "side": pd.Series(
                ["BUY", "SELL", "BUY", "SELL"],
                dtype="string",
            ),
            "quantity": pd.Series([100] * 4, dtype="int64"),
            "price": [2.0, 2.5, 3.0, 2.0],
            "fee": [0.5] * 4,
        }
    )


def make_comparison():
    return compare_three_approaches(
        make_market_data(),
        make_trade_data(),
        2,
        3,
        backtest_config=BacktestConfig(
            initial_cash=1000.0,
            commission_rate=0.0,
            slippage_rate=0.0,
            lot_size=10,
        ),
        analytics_config=AnalyticsConfig(periods_per_year=7),
    )


def test_review_comparison_answers_core_questions() -> None:
    review = review_comparison(make_comparison())

    assert review.conclusions["strategy_beats_buy_and_hold"] is True
    assert review.conclusions["personal_beats_strategy"] is False
    assert review.conclusions["lowest_cost_approach"] == (
        "buy_and_hold"
    )
    assert review.cost_attribution["comparison"].tolist() == [
        "strategy_vs_buy_and_hold",
        "personal_vs_strategy",
    ]


def test_review_comparison_ranks_winners_and_losers() -> None:
    review = review_comparison(make_comparison(), top_n=1)

    assert review.top_winners["buy_trade_id"].tolist() == ["T001"]
    assert review.top_winners.iloc[0]["realized_pnl"] == pytest.approx(
        49.0
    )
    assert review.top_losers["buy_trade_id"].tolist() == ["T003"]
    assert review.top_losers.iloc[0]["realized_pnl"] == pytest.approx(
        -101.0
    )


def test_review_comparison_rejects_invalid_top_n() -> None:
    with pytest.raises(ValueError, match="必须是正整数"):
        review_comparison(make_comparison(), top_n=0)
