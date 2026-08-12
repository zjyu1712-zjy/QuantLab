import pandas as pd
import pytest

from quantlab.trades.statistics import calculate_trade_statistics


def test_calculate_trade_statistics() -> None:
    matches = pd.DataFrame(
        {"realized_pnl": [100.0, 50.0, -25.0, -75.0, 0.0]}
    )

    statistics = calculate_trade_statistics(matches)

    assert statistics.closed_trade_count == 5
    assert statistics.winning_trade_count == 2
    assert statistics.losing_trade_count == 2
    assert statistics.breakeven_trade_count == 1
    assert statistics.win_rate == pytest.approx(0.4)
    assert statistics.average_profit == pytest.approx(75.0)
    assert statistics.average_loss == pytest.approx(-50.0)
    assert statistics.profit_loss_ratio == pytest.approx(1.5)


def test_trade_statistics_all_winners_have_undefined_ratio() -> None:
    matches = pd.DataFrame({"realized_pnl": [10.0, 20.0]})

    statistics = calculate_trade_statistics(matches)

    assert statistics.win_rate == 1.0
    assert statistics.average_profit == pytest.approx(15.0)
    assert statistics.average_loss is None
    assert statistics.profit_loss_ratio is None


def test_trade_statistics_all_losers_have_undefined_ratio() -> None:
    matches = pd.DataFrame({"realized_pnl": [-10.0, -20.0]})

    statistics = calculate_trade_statistics(matches)

    assert statistics.win_rate == 0.0
    assert statistics.average_profit is None
    assert statistics.average_loss == pytest.approx(-15.0)
    assert statistics.profit_loss_ratio is None


def test_trade_statistics_rejects_empty_matches() -> None:
    matches = pd.DataFrame(columns=["realized_pnl"])

    with pytest.raises(ValueError, match="没有已平仓交易"):
        calculate_trade_statistics(matches)


def test_trade_statistics_rejects_missing_pnl() -> None:
    matches = pd.DataFrame({"realized_pnl": [10.0, None]})

    with pytest.raises(ValueError, match="不能包含缺失值"):
        calculate_trade_statistics(matches)
