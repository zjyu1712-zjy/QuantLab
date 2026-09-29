"""组合研究的公开入口：验证快照、运行内置因子、模拟并绑定运行身份。"""

from copy import deepcopy
from dataclasses import asdict, replace

from quantlab.backtest.provenance import create_metadata, dataframe_sha256
from quantlab.data.research_sets import ResearchSetPreview
from quantlab.factors import run_research_factor
from quantlab.portfolio.engine import simulate_portfolio
from quantlab.portfolio.selection import build_rebalance_schedule
from quantlab.portfolio.types import PortfolioBacktestError, PortfolioBacktestResult, PortfolioConfig


def _warnings(factor) -> list[str]:
    """保留上游限制，补充组合成交与持仓解释；不是投资有效性承诺。"""
    messages = list(factor.warnings)
    messages.extend([
        "下一研究日期不必然是下一官方交易日；并集可能漏掉全体缺失日，用户日历也未经官方认证。",
        "每年 252 期是日线等间隔的年化假设；日期缺口可能使相邻观测跨越多个交易日，本步骤尚未计算绩效。",
        "当前静态 ETF 集合不是历史逐日可投资池，仍可能有幸存者偏差。",
        "等权是调仓目标，不保证每日实际权重；费用、整手取整、预算不足、价格变动及跳过调仓均可产生偏离。",
        "实际成交才校验当日成交量大于零；这是回顾性输入适用性检查，不声称开盘时已知道全天成交量。",
        "正成交量不保证按开盘价可成交；本模型没有停牌恢复、涨跌停、盘口或容量限制。",
        "闲置现金不计利息，期末不强制平仓；分红和拆并份额不单独入账。",
        "总资产耗尽或非正时停止并报错，不伪造零资产下的持仓权重或完整绩效。",
    ])
    if factor.adjustment == "none":
        messages.append("未复权价格未处理公司行动；本模型资产变化不是含分红总回报。")
    else:
        messages.append("复权价不是真实历史成交价；复权坐标中的份额与整手仅为模型单位，不等于历史真实可买份额。")
    return list(dict.fromkeys(messages))


def _config_metadata(config: PortfolioConfig) -> dict:
    """以标准 JSON 数值记录配置，兼容校验允许的 NumPy 数值标量。"""
    backtest = asdict(config.backtest)
    return {
        "top_n": int(config.top_n), "rebalance_interval": int(config.rebalance_interval),
        "ascending": config.ascending,
        "backtest": {key: int(value) if key == "lot_size" else float(value)
                     for key, value in backtest.items()},
    }


def run_factor_portfolio(
    preview: ResearchSetPreview, *, factor_name: str, config: PortfolioConfig,
) -> PortfolioBacktestResult:
    """纯内存运行，不回读来源文件、不修复身份、不接受外部因子或注册表。

    成功返回完整曲线、独立输入快照和 portfolio-run-v1 身份；失败只抛出
    带上下文的错误，不修改输入、不保存部分结果。返回表格本身仍可修改。
    """
    if not isinstance(config, PortfolioConfig):
        raise PortfolioBacktestError("组合配置必须为 PortfolioConfig", reason="invalid_config")
    # 冻结容器并非防篡改认证；重新校验嵌套交易参数，并隔离调用方对象。
    active_config = PortfolioConfig(
        top_n=config.top_n, rebalance_interval=config.rebalance_interval,
        ascending=config.ascending, backtest=deepcopy(config.backtest),
    )
    if not isinstance(preview, ResearchSetPreview):
        raise PortfolioBacktestError("组合研究输入必须为 ResearchSetPreview", reason="invalid_research_input")
    if not isinstance(factor_name, str) or not factor_name or factor_name != factor_name.strip():
        raise PortfolioBacktestError("因子名称必须为非空且无两端空白的字符串", reason="invalid_factor")
    snapshot = deepcopy(preview)
    try:
        # 此入口默认新建内置注册表；同时完整校验来源、网格、警告与已有指纹。
        # run_research_factor 自己处理无身份意义的行标签，不会修补损坏的行值。
        factor = run_research_factor(snapshot, factor_name)
    except KeyError as error:
        raise PortfolioBacktestError(f"组合因子选择失败：{error}", reason="invalid_factor") from error
    except (ValueError, TypeError, OverflowError) as error:
        raise PortfolioBacktestError(f"组合研究输入校验失败：{error}", reason="invalid_research_input") from error

    # 仅对已通过身份校验的副本重置行标签，保留 has_observation 与所有原始行值。
    market = snapshot.alignment.data.copy(deep=True)
    market.index = market.index.set_names([None] * market.index.nlevels)
    market = market.reset_index(drop=True)
    factor_data = factor.data.copy(deep=True)
    schedule = build_rebalance_schedule(factor, config=active_config)
    try:
        identity = create_metadata(market, active_config.backtest, strategy_name="single_factor_top_n")
    except ValueError as error:
        raise PortfolioBacktestError(f"组合运行身份校验失败：{error}", reason="invalid_run_identity") from error
    # 执行错误已有日期、代码和稳定原因，直接保留，不改写成输入错误。
    simulated = simulate_portfolio(market, factor.calendar, schedule, config=active_config)
    calendar = asdict(factor.calendar)
    calendar["dates"] = [day.isoformat() for day in factor.calendar.dates]
    for field in ("coverage_start", "coverage_end"):
        calendar[field] = calendar[field].isoformat()
    metadata = {
        "schema_version": "portfolio-run-v1", "status": "complete",
        "run_id": identity.run_id, "run_at": identity.run_at,
        "code_sha256": identity.code_sha256, "environment": deepcopy(identity.environment),
        "strategy": {
            "name": "single_factor_top_n", "version": "1", "config": _config_metadata(active_config),
            "resolved_ascending": schedule.ascending,
            "lookback_periods": factor.definition.lookback_periods,
            "signal_anchor_position": factor.definition.lookback_periods,
            "signal_rule": "L+k*K", "execution_rule": "next_research_date_open",
        },
        "factor_definition": asdict(factor.definition),
        "data": {
            "preview_sha256": factor.input_preview_sha256,
            "alignment_content_sha256": factor.alignment_content_sha256,
            "market_sha256": dataframe_sha256(market), "factor_sha256": dataframe_sha256(factor_data),
            "sources": deepcopy(list(factor.sources)),
            "start": factor.start.isoformat(), "end": factor.end.isoformat(),
            "adjustment": factor.adjustment, "calendar": calendar,
            "row_count": len(market), "symbols": [source["symbol"] for source in factor.sources],
        },
        "warnings": _warnings(factor), "analytics_config": None,
    }
    return replace(simulated, metadata=metadata, input_data=market, factor_data=factor_data)
