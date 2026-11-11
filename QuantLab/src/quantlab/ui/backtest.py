from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantlab.backtest import (
    MovingAverageBacktestResult,
    run_dual_moving_average_backtest,
)
from quantlab.config.settings import AnalyticsConfig, BacktestConfig


@dataclass(frozen=True)
class MovingAverageBacktestView:
    """双均线回测页面使用的参数、统计和明细。"""

    symbol: str
    short_window: int
    long_window: int
    final_equity: float
    signal_count: int
    filled_trade_count: int
    total_costs: float
    result: MovingAverageBacktestResult
    chart_data: pd.DataFrame


def build_moving_average_backtest_view(
    market_data: pd.DataFrame,
    *,
    short_window: int,
    long_window: int,
    initial_cash: float,
    commission_rate: float,
    minimum_commission: float,
    slippage_rate: float,
    sell_tax_rate: float,
    lot_size: int,
    annual_risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    data_source: str | None = None,
    data_version: str | None = None,
) -> MovingAverageBacktestView:
    """运行双均线回测并整理页面需要的数据。"""
    backtest_config = BacktestConfig(
        initial_cash=initial_cash,
        commission_rate=commission_rate,
        minimum_commission=minimum_commission,
        slippage_rate=slippage_rate,
        sell_tax_rate=sell_tax_rate,
        lot_size=lot_size,
    )
    analytics_config = AnalyticsConfig(
        periods_per_year=periods_per_year,
        annual_risk_free_rate=annual_risk_free_rate,
    )
    result = run_dual_moving_average_backtest(
        market_data,
        short_window,
        long_window,
        backtest_config=backtest_config,
        analytics_config=analytics_config,
        data_source=data_source,
        data_version=data_version,
    )

    equity = result.backtest.equity_curve.loc[
        :,
        ["date", "equity", "net_value", "cumulative_costs"],
    ]
    chart_data = result.signals.merge(
        equity,
        on="date",
        how="left",
        validate="one_to_one",
    )
    symbol = str(result.signals["symbol"].iloc[0])
    final_equity = float(
        result.backtest.equity_curve["equity"].iloc[-1]
    )
    total_costs = float(
        result.backtest.equity_curve["cumulative_costs"].iloc[-1]
    )

    return MovingAverageBacktestView(
        symbol=symbol,
        short_window=short_window,
        long_window=long_window,
        final_equity=final_equity,
        signal_count=int(result.signals["signal"].notna().sum()),
        filled_trade_count=len(result.backtest.trades),
        total_costs=total_costs,
        result=result,
        chart_data=chart_data,
    )
