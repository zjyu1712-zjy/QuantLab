"""只决定何时选取哪些 ETF；不计算成交、持仓、费用或组合收益。"""

from datetime import date
from itertools import product
import re

import numpy as np
import pandas as pd

from quantlab.data.calendar import CalendarAxis
from quantlab.factors import FactorDefinition, FactorRunResult
from quantlab.portfolio.types import (
    DECISION_DTYPES, TARGET_DTYPES, PortfolioBacktestError, PortfolioConfig, RebalanceSchedule,
)
FACTOR_COLUMNS = ["date", "symbol", "factor_value", "is_valid", "status"]


def _invalid(message: str) -> None:
    raise PortfolioBacktestError(message, reason="invalid_factor_result")


def _checked_frame(result: FactorRunResult) -> tuple[pd.DataFrame, tuple[date, ...], tuple[str, ...]]:
    """检查内部数值表契约，不认证公式来源或重算/修补研究输入指纹。"""
    if not isinstance(result, FactorRunResult):
        _invalid("调仓输入必须为 FactorRunResult")
    definition = result.definition
    if not isinstance(definition, FactorDefinition):
        _invalid("因子结果缺少有效定义")
    if definition.availability != "after_close":
        _invalid("组合调仓只支持收盘后可用的因子")
    if definition.direction not in ("higher_is_stronger", "higher_is_more_volatile"):
        _invalid("未知因子方向，不能猜测排序规则")

    axis = result.calendar
    if not isinstance(axis, CalendarAxis) or not isinstance(axis.dates, tuple) or not axis.dates:
        _invalid("研究日期轴必须为非空 CalendarAxis")
    dates = axis.dates
    if any(type(day) is not date for day in dates) or dates != tuple(sorted(set(dates))):
        _invalid("研究日期轴必须为升序、无重复的纯日期")
    if (type(result.start) is not date or type(result.end) is not date
            or not result.start <= dates[0] <= dates[-1] <= result.end):
        _invalid("因子日期轴必须位于声明的研究区间内")

    if (not isinstance(result.sources, tuple) or len(result.sources) < 2
            or any(not isinstance(source, dict) for source in result.sources)):
        _invalid("因子结果必须包含至少两只 ETF 的来源代码")
    symbols = tuple(source.get("symbol") for source in result.sources)
    if (any(not isinstance(symbol, str) or not re.fullmatch(r"[0-9]{6}", symbol) for symbol in symbols)
            or symbols != tuple(sorted(set(symbols)))):
        _invalid("来源 ETF 代码必须为升序、唯一的六位字符串")

    if not isinstance(result.data, pd.DataFrame) or result.data.columns.tolist() != FACTOR_COLUMNS:
        _invalid("因子表必须包含规范的日期、代码、数值、有效标记和状态列")
    # 行索引不是数据身份；只重置自己的副本，不依靠标签对齐或修改原表。
    frame = result.data.copy(deep=True).reset_index(drop=True)
    if (frame.date.dtype != np.dtype("datetime64[ns]") or frame.date.isna().any()
            or not frame.date.eq(frame.date.dt.normalize()).all()):
        _invalid("因子日期必须为无空值、无时区的午夜 datetime64[ns]")
    if any(not isinstance(symbol, str) for symbol in frame.symbol):
        _invalid("因子代码必须为非空字符串")
    if list(zip(frame.date.dt.date, frame.symbol)) != list(product(dates, symbols)):
        _invalid("因子表必须保留有序、无重复的完整日期与代码网格")
    if frame.factor_value.dtype != np.dtype("float64"):
        _invalid("因子数值必须为 float64")
    if frame.is_valid.dtype != np.dtype("bool") or frame.is_valid.isna().any():
        _invalid("因子有效标记必须为无空值布尔列")
    if (not isinstance(frame.status.dtype, pd.StringDtype) or frame.status.isna().any()
            or not frame.status.isin(["warmup", "missing_window", "numeric_range", "valid"]).all()
            or not frame.is_valid.eq(frame.status.eq("valid")).all()):
        _invalid("因子状态与有效标记不一致或含未知状态")
    if (not np.isfinite(frame.loc[frame.is_valid, "factor_value"]).all()
            or not frame.loc[~frame.is_valid, "factor_value"].isna().all()):
        _invalid("有效因子值必须有限，不可用值必须为空")
    return frame, dates, symbols


def build_rebalance_schedule(result: FactorRunResult, *, config: PortfolioConfig) -> RebalanceSchedule:
    """热身后按固定轴位置生成目标；不足 N 跳过，末日足 N 过期。"""
    if not isinstance(config, PortfolioConfig):
        raise PortfolioBacktestError("组合配置必须为 PortfolioConfig", reason="invalid_config")
    frame, dates, symbols = _checked_frame(result)
    if config.top_n > len(symbols):
        raise PortfolioBacktestError("选择数量 N 不能超过研究集合的 ETF 数量", reason="invalid_config")
    ascending = (result.definition.direction == "higher_is_more_volatile"
                 if config.ascending is None else config.ascending)
    decisions, targets = [], []
    for index in range(result.definition.lookback_periods, len(dates), config.rebalance_interval):
        day = pd.Timestamp(dates[index])
        valid = frame.loc[frame.date.eq(day) & frame.is_valid].sort_values(
            ["factor_value", "symbol"], ascending=[ascending, True],
        ).reset_index(drop=True)
        count = len(valid)
        enough = count >= config.top_n
        has_next = index + 1 < len(dates)
        decision_id = f"PD{len(decisions) + 1:06d}"
        cutoff_tie = bool(enough and count > config.top_n
                          and valid.factor_value.iloc[config.top_n - 1] == valid.factor_value.iloc[config.top_n])
        if not enough:
            status, reason = "SKIPPED_INSUFFICIENT_FACTORS", f"有效因子 {count} 只，不足 N={config.top_n}；跳过本次调仓"
        elif not has_next:
            status, reason = "EXPIRED", "已生成目标，但没有下一研究日期可供执行"
        else:
            status, reason = "PENDING", "收盘后生成目标，待下一研究日期开盘执行"
        if cutoff_tie:
            reason += "；选取边界同值，按 ETF 代码升序确定席位，不表示投资优劣"
        decisions.append({
            "decision_id": decision_id, "signal_date": day,
            "execution_date": pd.Timestamp(dates[index + 1]) if has_next else pd.NaT,
            "valid_count": count, "top_n": config.top_n, "status": status,
            "reason": reason, "cutoff_tie": cutoff_tie,
        })
        if enough:
            for row in valid.iloc[:config.top_n].itertuples(index=False):
                targets.append({"decision_id": decision_id, "signal_date": day, "symbol": row.symbol,
                                "factor_value": row.factor_value, "target_weight": 1.0 / config.top_n})
    return RebalanceSchedule(
        decisions=pd.DataFrame(decisions, columns=list(DECISION_DTYPES)).astype(DECISION_DTYPES),
        targets=pd.DataFrame(targets, columns=list(TARGET_DTYPES)).astype(TARGET_DTYPES),
        ascending=ascending,
    )
