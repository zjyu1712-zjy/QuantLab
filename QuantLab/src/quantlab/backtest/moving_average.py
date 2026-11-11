"""双均线策略的完整回测流程。"""

from dataclasses import dataclass

import pandas as pd

from quantlab.analytics.performance import PerformanceAnalysis
from quantlab.backtest.analysis import analyze_backtest
from quantlab.backtest.engine import BacktestResult
from quantlab.config.settings import (
    DEFAULT_ANALYTICS_CONFIG,
    DEFAULT_BACKTEST_CONFIG,
    AnalyticsConfig,
    BacktestConfig,
)
from quantlab.strategies.moving_average import (
    generate_dual_moving_average_signals,
)


@dataclass(frozen=True)
class MovingAverageBacktestResult:
    """双均线研究流程产生的信号、回测和绩效结果。"""

    signals: pd.DataFrame
    backtest: BacktestResult
    performance: PerformanceAnalysis


def run_dual_moving_average_backtest(
    market_data: pd.DataFrame,
    short_window: int,
    long_window: int,
    *,
    backtest_config: BacktestConfig = DEFAULT_BACKTEST_CONFIG,
    analytics_config: AnalyticsConfig = DEFAULT_ANALYTICS_CONFIG,
    data_source: str | None = None,
    data_version: str | None = None,
) -> MovingAverageBacktestResult:
    """生成双均线信号，并按统一规则完成回测和绩效分析。"""
    signals = generate_dual_moving_average_signals(
        market_data,
        short_window,
        long_window,
    )
    analysis = analyze_backtest(
        signals,
        backtest_config=backtest_config,
        analytics_config=analytics_config,
        strategy_name="dual_moving_average",
        strategy_parameters={"short_window": short_window, "long_window": long_window},
        data_source=data_source,
        data_version=data_version,
    )

    return MovingAverageBacktestResult(
        signals=signals,
        backtest=analysis.backtest,
        performance=analysis.performance,
    )
