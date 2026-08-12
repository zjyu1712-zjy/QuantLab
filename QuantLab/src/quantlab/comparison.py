"""买入持有、双均线和个人真实交易的统一比较。"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantlab.analytics.performance import (
    PerformanceAnalysis,
    analyze_performance,
)
from quantlab.backtest import (
    BacktestAnalysis,
    MovingAverageBacktestResult,
    run_buy_and_hold,
    run_dual_moving_average_backtest,
)
from quantlab.config.settings import (
    DEFAULT_ANALYTICS_CONFIG,
    DEFAULT_BACKTEST_CONFIG,
    AnalyticsConfig,
    BacktestConfig,
)
from quantlab.trades import TradeAnalysis, analyze_trades


@dataclass(frozen=True)
class ThreeWayComparison:
    """三种投资路径的明细、曲线、指标和收益差。"""

    initial_cash: float
    buy_and_hold: BacktestAnalysis
    strategy: MovingAverageBacktestResult
    personal: TradeAnalysis
    personal_performance: PerformanceAnalysis
    equity_curves: pd.DataFrame
    metrics: pd.DataFrame
    return_gaps: dict[str, float]


def _metric_row(
    approach: str,
    performance: PerformanceAnalysis,
    final_equity: float,
    trade_count: int,
    total_costs: float,
) -> dict[str, object]:
    return {
        "approach": approach,
        "final_equity": final_equity,
        "cumulative_return": performance.cumulative_return,
        "annualized_return": performance.annualized_return,
        "annualized_volatility": (
            performance.annualized_volatility
        ),
        "max_drawdown": performance.max_drawdown,
        "sharpe_ratio": performance.sharpe_ratio,
        "trade_count": trade_count,
        "total_costs": total_costs,
    }


def compare_three_approaches(
    market_data: pd.DataFrame,
    trade_source: pd.DataFrame | str | Path,
    short_window: int,
    long_window: int,
    *,
    backtest_config: BacktestConfig = DEFAULT_BACKTEST_CONFIG,
    analytics_config: AnalyticsConfig = DEFAULT_ANALYTICS_CONFIG,
) -> ThreeWayComparison:
    """使用相同资金、行情区间和指标定义比较三种投资路径。"""
    buy_and_hold = run_buy_and_hold(
        market_data,
        backtest_config=backtest_config,
        analytics_config=analytics_config,
    )
    strategy = run_dual_moving_average_backtest(
        market_data,
        short_window,
        long_window,
        backtest_config=backtest_config,
        analytics_config=analytics_config,
    )
    personal = analyze_trades(
        market_data,
        trade_source,
        initial_cash=backtest_config.initial_cash,
    )

    personal_equity = personal.equity_curve.set_index("date")["equity"]
    personal_performance = analyze_performance(
        personal_equity,
        annual_risk_free_rate=(
            analytics_config.annual_risk_free_rate
        ),
        periods_per_year=analytics_config.periods_per_year,
    )

    equity_curves = pd.concat(
        [
            buy_and_hold.backtest.equity_curve.set_index("date")[
                "equity"
            ].rename("buy_and_hold"),
            strategy.backtest.equity_curve.set_index("date")[
                "equity"
            ].rename("dual_moving_average"),
            personal_equity.rename("personal_trades"),
        ],
        axis=1,
        join="inner",
    ).reset_index()

    hold_curve = buy_and_hold.backtest.equity_curve
    strategy_curve = strategy.backtest.equity_curve
    personal_curve = personal.equity_curve

    metrics = pd.DataFrame(
        [
            _metric_row(
                "buy_and_hold",
                buy_and_hold.performance,
                float(hold_curve.iloc[-1]["equity"]),
                len(buy_and_hold.backtest.trades),
                float(hold_curve.iloc[-1]["cumulative_costs"]),
            ),
            _metric_row(
                "dual_moving_average",
                strategy.performance,
                float(strategy_curve.iloc[-1]["equity"]),
                len(strategy.backtest.trades),
                float(strategy_curve.iloc[-1]["cumulative_costs"]),
            ),
            _metric_row(
                "personal_trades",
                personal_performance,
                float(personal_curve.iloc[-1]["equity"]),
                len(personal.trades),
                float(personal_curve.iloc[-1]["cumulative_fees"]),
            ),
        ]
    )

    hold_return = buy_and_hold.performance.cumulative_return
    strategy_return = strategy.performance.cumulative_return
    personal_return = personal_performance.cumulative_return
    return_gaps = {
        "strategy_minus_buy_and_hold": (
            strategy_return - hold_return
        ),
        "personal_minus_strategy": (
            personal_return - strategy_return
        ),
        "personal_minus_buy_and_hold": (
            personal_return - hold_return
        ),
    }

    return ThreeWayComparison(
        initial_cash=backtest_config.initial_cash,
        buy_and_hold=buy_and_hold,
        strategy=strategy,
        personal=personal,
        personal_performance=personal_performance,
        equity_curves=equity_curves,
        metrics=metrics,
        return_gaps=return_gaps,
    )
