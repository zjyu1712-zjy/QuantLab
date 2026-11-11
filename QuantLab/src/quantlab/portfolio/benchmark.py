"""同热身期的全池等权买入持有基准；仅一次建仓，不使用因子排名。"""

from copy import deepcopy
from dataclasses import asdict, replace

import pandas as pd

from quantlab.backtest.provenance import create_metadata, dataframe_sha256
from quantlab.config.settings import (
    AnalyticsConfig, BacktestConfig, DEFAULT_ANALYTICS_CONFIG, DEFAULT_BACKTEST_CONFIG,
)
from quantlab.data.research_sets import ResearchSetPreview
from quantlab.factors import run_research_factor
from quantlab.portfolio.analysis import _checked_config, _performance_from_equity
from quantlab.portfolio.engine import simulate_portfolio
from quantlab.portfolio.research import _config_metadata, _warnings
from quantlab.portfolio.types import (
    DECISION_DTYPES, TARGET_DTYPES, PortfolioAnalysis, PortfolioBacktestError, PortfolioConfig, RebalanceSchedule,
)


def run_equal_weight_benchmark(
    preview: ResearchSetPreview, *, factor_name: str,
    backtest_config: BacktestConfig = DEFAULT_BACKTEST_CONFIG,
    analytics_config: AnalyticsConfig = DEFAULT_ANALYTICS_CONFIG,
) -> PortfolioAnalysis:
    """固定在内置因子 L+1 位置开盘全池建仓，返回独立的完整曲线及来源。

    复用研究输入验证、交易引擎和绩效公式，不借用因子策略身份。
    因子只用于确定热身长度；有效数量、排名和首次实际成交都不改变基准。
    不回读源文件、不保存档案；必要价格缺失时整次基准失败。
    """
    periods, rate = _checked_config(analytics_config)
    # 先验证交易配置，再接触研究输入；沿用组合侧的严格参数边界。
    PortfolioConfig(top_n=1, backtest=backtest_config)
    if not isinstance(preview, ResearchSetPreview):
        raise PortfolioBacktestError("基准输入必须为研究快照", reason="invalid_research_input")
    if not isinstance(factor_name, str) or not factor_name or factor_name != factor_name.strip():
        raise PortfolioBacktestError("因子名称无效", reason="invalid_factor")
    snapshot = deepcopy(preview)
    try:
        context = run_research_factor(snapshot, factor_name)
    except KeyError as error:
        raise PortfolioBacktestError("基准热身因子不存在", reason="invalid_factor") from error
    except (ValueError, TypeError, OverflowError) as error:
        raise PortfolioBacktestError("基准研究输入校验失败", reason="invalid_research_input") from error
    market = snapshot.alignment.data.copy(deep=True)
    market.index = market.index.set_names([None] * market.index.nlevels)
    market = market.reset_index(drop=True)
    symbols = [source["symbol"] for source in context.sources]
    config = PortfolioConfig(top_n=len(symbols), backtest=deepcopy(backtest_config))
    dates, anchor = context.calendar.dates, context.definition.lookback_periods
    execution_date = dates[anchor + 1] if anchor + 1 < len(dates) else None
    decisions, targets = [], []
    if anchor < len(dates):
        signal_date = pd.Timestamp(dates[anchor])
        decisions.append({
            "decision_id": "PD000001", "signal_date": signal_date,
            "execution_date": pd.Timestamp(execution_date) if execution_date else pd.NaT,
            "valid_count": len(symbols), "top_n": len(symbols),
            "status": "PENDING" if execution_date else "EXPIRED", "cutoff_tie": False,
            "reason": "全池等权目标，仅计划一次建仓，不使用因子排名",
        })
        targets = [{"decision_id": "PD000001", "signal_date": signal_date, "symbol": symbol,
                    "factor_value": 0.0, "target_weight": 1 / len(symbols)} for symbol in symbols]
    schedule = RebalanceSchedule(
        decisions=pd.DataFrame(decisions, columns=list(DECISION_DTYPES)).astype(DECISION_DTYPES),
        targets=pd.DataFrame(targets, columns=list(TARGET_DTYPES)).astype(TARGET_DTYPES), ascending=False,
    )
    try:
        identity = create_metadata(market, config.backtest, strategy_name="equal_weight_buy_and_hold")
    except ValueError as error:
        raise PortfolioBacktestError("基准运行身份校验失败", reason="invalid_run_identity") from error
    simulated = simulate_portfolio(market, context.calendar, schedule, config=config)
    # 常量仅满足通用引擎的内部调度契约，不能作为真实因子值发布。
    simulated.targets["factor_value"] = float("nan")
    calendar = asdict(context.calendar)
    calendar["dates"] = [day.isoformat() for day in dates]
    for field in ("coverage_start", "coverage_end"):
        calendar[field] = calendar[field].isoformat()
    warnings = _warnings(context) + [
        "本基准在 L+1 位置一次建仓，此前持有现金；不跟随策略首次实际成交，不再调仓。",
        "全池等权仅为初始目标，之后权重自然漂移；累计收益差不是市场指数超额收益或风险调整 alpha。",
    ]
    if execution_date is None:
        warnings.append("区间未到计划建仓日（L+1），基准保持完整现金曲线。")
    elif simulated.trades.empty:
        warnings.append("计划建仓日未发生成交，基准保持现金且不会顺延重试。")
    metadata = {
        "schema_version": "portfolio-benchmark-v1", "status": "complete",
        "run_id": identity.run_id, "run_at": identity.run_at,
        "code_sha256": identity.code_sha256, "environment": deepcopy(identity.environment),
        "strategy": {
            "name": "equal_weight_buy_and_hold", "version": "1",
            "config": {"backtest": _config_metadata(config)["backtest"]},
            "signal_anchor_position": anchor, "execution_rule": "L+1_open_once",
            "anchor_factor": {"name": context.definition.name, "version": context.definition.version},
            "planned_execution_date": execution_date.isoformat() if execution_date else None,
            "target_universe": "all_snapshot_symbols", "target_weight": 1 / len(symbols),
        },
        "data": {
            "preview_sha256": context.input_preview_sha256,
            "alignment_content_sha256": context.alignment_content_sha256,
            "market_sha256": dataframe_sha256(market), "sources": deepcopy(list(context.sources)),
            "start": context.start.isoformat(), "end": context.end.isoformat(),
            "adjustment": context.adjustment, "calendar": calendar,
            "row_count": len(market), "symbols": symbols,
        },
        "warnings": warnings, "analytics_config": {"periods_per_year": periods, "annual_risk_free_rate": rate},
    }
    result = replace(simulated, metadata=metadata, input_data=market)
    equity = pd.Series(result.equity_curve.equity.to_numpy(copy=True),
                       index=pd.DatetimeIndex(result.equity_curve.date, name="date"), name="equity")
    performance = _performance_from_equity(equity, periods=periods, risk_free_rate=rate)
    return PortfolioAnalysis(backtest=result, performance=performance)
