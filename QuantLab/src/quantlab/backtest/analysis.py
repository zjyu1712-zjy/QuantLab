"""连接回测结果与统一绩效指标。"""

from dataclasses import asdict, dataclass, replace

import pandas as pd

from quantlab.analytics.performance import (
    PerformanceAnalysis,
    analyze_performance,
)
from quantlab.backtest.engine import BacktestResult, run_backtest
from quantlab.config.settings import (
    DEFAULT_ANALYTICS_CONFIG,
    DEFAULT_BACKTEST_CONFIG,
    AnalyticsConfig,
    BacktestConfig,
)


@dataclass(frozen=True)
class BacktestAnalysis:
    """一次回测的成交、资金曲线和绩效结果。"""

    backtest: BacktestResult
    performance: PerformanceAnalysis


def analyze_backtest(
    signal_data: pd.DataFrame,
    *,
    backtest_config: BacktestConfig = DEFAULT_BACKTEST_CONFIG,
    analytics_config: AnalyticsConfig = DEFAULT_ANALYTICS_CONFIG,
    strategy_name: str = "external_signals",
    strategy_parameters: dict | None = None,
    data_source: str | None = None,
    data_version: str | None = None,
) -> BacktestAnalysis:
    """运行回测，并用统一口径分析每日总资产。"""
    backtest = run_backtest(
        signal_data, backtest_config,
        strategy_name=strategy_name, strategy_parameters=strategy_parameters,
        data_source=data_source, data_version=data_version,
    )
    return analyze_backtest_result(
        backtest,
        analytics_config=analytics_config,
    )


def analyze_backtest_result(
    backtest: BacktestResult,
    *,
    analytics_config: AnalyticsConfig = DEFAULT_ANALYTICS_CONFIG,
) -> BacktestAnalysis:
    """分析一份已经生成的回测结果。"""
    equity = backtest.equity_curve.set_index("date")["equity"]

    performance = analyze_performance(
        equity,
        annual_risk_free_rate=(
            analytics_config.annual_risk_free_rate
        ),
        periods_per_year=analytics_config.periods_per_year,
    )
    if backtest.metadata is not None:
        backtest = replace(
            backtest,
            metadata=replace(backtest.metadata, analytics_config=asdict(analytics_config)),
        )

    return BacktestAnalysis(
        backtest=backtest,
        performance=performance,
    )
