from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantlab.comparison import (
    ThreeWayComparison,
    compare_three_approaches,
)
from quantlab.config.settings import AnalyticsConfig, BacktestConfig
from quantlab.review import ComparisonReview, review_comparison


APPROACH_LABELS = {
    "buy_and_hold": "买入持有",
    "dual_moving_average": "双均线策略",
    "personal_trades": "个人真实交易",
}


@dataclass(frozen=True)
class ThreeWayComparisonView:
    """三方对比页面使用的曲线、指标和解释。"""

    symbol: str
    comparison: ThreeWayComparison
    review: ComparisonReview
    normalized_curves: pd.DataFrame
    display_metrics: pd.DataFrame


def build_three_way_comparison_view(
    market_data: pd.DataFrame,
    trade_data: pd.DataFrame,
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
) -> ThreeWayComparisonView:
    """按统一口径运行三方比较并整理界面数据。"""
    comparison = compare_three_approaches(
        market_data,
        trade_data,
        short_window,
        long_window,
        backtest_config=BacktestConfig(
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            minimum_commission=minimum_commission,
            slippage_rate=slippage_rate,
            sell_tax_rate=sell_tax_rate,
            lot_size=lot_size,
        ),
        analytics_config=AnalyticsConfig(
            periods_per_year=periods_per_year,
            annual_risk_free_rate=annual_risk_free_rate,
        ),
    )
    review = review_comparison(comparison)

    normalized_curves = comparison.equity_curves.copy()
    value_columns = list(APPROACH_LABELS)
    normalized_curves[value_columns] = (
        normalized_curves[value_columns] / initial_cash
    )
    normalized_curves = normalized_curves.melt(
        id_vars="date",
        value_vars=value_columns,
        var_name="approach",
        value_name="net_value",
    )
    normalized_curves["approach"] = normalized_curves[
        "approach"
    ].map(APPROACH_LABELS)

    display_metrics = comparison.metrics.copy()
    display_metrics["approach"] = display_metrics["approach"].map(
        APPROACH_LABELS
    )
    symbol = str(market_data["symbol"].iloc[0])

    return ThreeWayComparisonView(
        symbol=symbol,
        comparison=comparison,
        review=review,
        normalized_curves=normalized_curves,
        display_metrics=display_metrics,
    )
