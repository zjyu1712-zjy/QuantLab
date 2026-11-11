"""有限参数对比：复用单次回测与指标入口，只保留汇总，不保存实验。"""

from copy import deepcopy

import pandas as pd

from quantlab.config.settings import (
    AnalyticsConfig, BacktestConfig, DEFAULT_ANALYTICS_CONFIG, DEFAULT_BACKTEST_CONFIG,
)
from quantlab.data.research_sets import ResearchSetPreview
from quantlab.factors import create_default_registry
from quantlab.portfolio.analysis import _checked_config, analyze_portfolio_result
from quantlab.portfolio.research import run_factor_portfolio
from quantlab.portfolio.types import PortfolioBacktestError, PortfolioConfig


TOP_N_CANDIDATES = (1, 2, 3)
INTERVAL_CANDIDATES = (5, 10, 20)
METRIC_FIELDS = ("cumulative_return", "annualized_return", "annualized_volatility", "max_drawdown", "sharpe_ratio")
# 仅保留稳定原因码；异常原文、上下文与回溯不进入会话汇总。
_FAILURE_REASONS = frozenset({
    "invalid_config", "invalid_analytics_config", "invalid_research_input", "invalid_factor",
    "invalid_factor_result", "invalid_run_identity", "invalid_portfolio_result", "invalid_market_data",
    "invalid_schedule", "invalid_execution_input", "missing_open", "missing_close", "zero_volume",
    "negative_cash", "negative_sell_proceeds", "non_positive_equity", "numeric_range",
})


def run_parameter_sensitivity(
    preview: ResearchSetPreview, *, factor_name: str, ascending: bool | None = None,
    backtest_config: BacktestConfig = DEFAULT_BACKTEST_CONFIG,
    analytics_config: AnalyticsConfig = DEFAULT_ANALYTICS_CONFIG,
) -> pd.DataFrame:
    """固定同一快照与费用口径，依次运行至多九组 N/K；不选择最优参数。

    每组都经过原有完整校验，失败只占据自己的汇总行，后续组继续。
    成功但零成交或指标未定义不算失败；失败的指标及成交数保持缺失。
    """
    costs, analytics = deepcopy(backtest_config), deepcopy(analytics_config)
    PortfolioConfig(top_n=1, ascending=ascending, backtest=costs)
    _checked_config(analytics)
    if not isinstance(preview, ResearchSetPreview) or not isinstance(preview.sources, (tuple, list)) or not preview.sources:
        raise PortfolioBacktestError("参数对比需要非空研究快照", reason="invalid_research_input")
    try:
        create_default_registry().get(factor_name)
    except (ValueError, KeyError) as error:
        raise PortfolioBacktestError("参数对比需要内置因子", reason="invalid_factor") from error
    snapshot = deepcopy(preview)
    rows = []
    for n in TOP_N_CANDIDATES:
        if n > len(snapshot.sources):
            continue
        for interval in INTERVAL_CANDIDATES:
            row = dict(top_n=n, rebalance_interval=interval, status="failed",
                       **{name: None for name in METRIC_FIELDS}, trade_count=None,
                       failure_reason="", unavailable_reasons={})
            try:
                result = run_factor_portfolio(snapshot, factor_name=factor_name, config=PortfolioConfig(
                    top_n=n, rebalance_interval=interval, ascending=ascending, backtest=costs))
                analysis = analyze_portfolio_result(result, analytics_config=analytics)
                # 只有完整回测和指标均成功，才发布这一组的任何结果。
                row.update({name: getattr(analysis.performance, name) for name in METRIC_FIELDS})
                row.update(status="complete", trade_count=len(analysis.backtest.trades),
                           unavailable_reasons=deepcopy(analysis.performance.unavailable_reasons))
            except Exception as error:
                reason = error.reason if isinstance(error, PortfolioBacktestError) else None
                row["failure_reason"] = reason if isinstance(reason, str) and reason in _FAILURE_REASONS else "unavailable"
            rows.append(row)
    table = pd.DataFrame(rows)
    table["trade_count"] = table.trade_count.astype("Int64")
    table[list(METRIC_FIELDS)] = table[list(METRIC_FIELDS)].astype("float64")
    return table
