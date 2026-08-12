from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quantlab.data import import_market_csv, load_trade_csv
from quantlab.ui import (
    build_market_performance_view,
    build_moving_average_backtest_view,
    build_three_way_comparison_view,
    build_trade_review_view,
)


def test_stage1_complete_research_loop(tmp_path: Path) -> None:
    """第一阶段的行情、分析、回测、比较和复盘能够首尾贯通。"""
    imported = import_market_csv(
        "tests/fixtures/valid_etf.csv",
        tmp_path / "raw",
        tmp_path / "processed",
        imported_at=datetime(2025, 1, 10, tzinfo=timezone.utc),
    )

    processed_path = Path(imported.processed_path)
    market_data = pd.read_parquet(processed_path)
    trade_data = load_trade_csv("tests/fixtures/valid_trades.csv")

    performance = build_market_performance_view(market_data)
    backtest = build_moving_average_backtest_view(
        market_data,
        short_window=1,
        long_window=2,
        initial_cash=100_000.0,
        commission_rate=0.0003,
        minimum_commission=0.0,
        slippage_rate=0.0001,
        sell_tax_rate=0.0,
        lot_size=100,
    )
    comparison = build_three_way_comparison_view(
        market_data,
        trade_data,
        short_window=1,
        long_window=2,
        initial_cash=100_000.0,
        commission_rate=0.0003,
        minimum_commission=0.0,
        slippage_rate=0.0001,
        sell_tax_rate=0.0,
        lot_size=100,
    )
    trade_review = build_trade_review_view(
        market_data,
        trade_data,
        short_window=1,
        long_window=2,
        initial_cash=100_000.0,
    )

    assert imported.symbol == "510300"
    assert imported.row_count == len(market_data) == 5
    assert processed_path.is_file()
    assert Path(imported.version.archived_path).is_file()
    assert performance.symbol == imported.symbol
    assert performance.observation_count == imported.row_count
    assert len(performance.analysis.daily_returns) == 4
    assert len(performance.analysis.net_value) == imported.row_count
    assert len(performance.analysis.drawdown) == imported.row_count
    assert backtest.symbol == imported.symbol
    assert len(backtest.result.backtest.equity_curve) == imported.row_count
    assert comparison.display_metrics["approach"].tolist() == [
        "买入持有",
        "双均线策略",
        "个人真实交易",
    ]
    assert len(comparison.normalized_curves) == imported.row_count * 3
    assert trade_review.symbol == imported.symbol
    assert len(trade_review.review.trade_alignment) == len(trade_data)
    assert trade_review.timing_summary["count"].sum() >= len(trade_data)
