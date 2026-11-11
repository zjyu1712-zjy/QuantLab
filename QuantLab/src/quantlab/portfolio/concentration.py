"""从完整资金曲线观察正向资产增量的集中程度，不重跑或评价策略。"""

from dataclasses import dataclass
from math import fsum, isfinite

import numpy as np
import pandas as pd

from quantlab.portfolio.types import PortfolioBacktestError


@dataclass(frozen=True)
class GainConcentration:
    interval_count: int
    positive_count: int
    selected_count: int
    gross_positive_gain: float
    top_gain: float
    top5_share: float | None
    status: str
    intervals: pd.DataFrame


def equity_gain_concentration(curve: pd.DataFrame) -> GainConcentration:
    """相邻资产差取正数，最高至多五项 / 全部正数之和；并列取较早区间。

    仅针对无外部出入金的既有回测资金曲线；不补日期，不将观测区间称作交易日。
    不改变输入或持久化状态；金额已包含曲线中的费用和估值变动，非已实现交易盈利。
    """
    def invalid():
        raise PortfolioBacktestError("集中度需要升序、无缺失的正资产曲线", reason="invalid_portfolio_result")

    if (not isinstance(curve, pd.DataFrame) or curve.empty or not curve.columns.is_unique
            or not {"date", "equity"}.issubset(curve.columns)):
        invalid()
    if curve.date.dtype != np.dtype("datetime64[ns]") or curve.equity.dtype != np.dtype("float64"):
        invalid()
    dates = pd.DatetimeIndex(curve.date)
    values = curve.equity.to_numpy(copy=True)
    if (dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing
            or not dates.equals(dates.normalize()) or not np.isfinite(values).all() or (values <= 0).any()):
        invalid()
    with np.errstate(over="ignore", invalid="ignore"):
        changes = np.diff(values)
    if not np.isfinite(changes).all():
        raise PortfolioBacktestError("资产增量超出可靠范围", reason="numeric_range")
    # 保留原始位置关系，不能让 DataFrame 的外部行标签影响起止日期。
    intervals = pd.DataFrame({"start_date": dates[:-1], "end_date": dates[1:], "asset_gain": changes})
    positives = intervals.loc[intervals.asset_gain.gt(0)]
    selected = positives.sort_values(["asset_gain", "end_date"], ascending=[False, True]).head(5).copy()
    selected = selected.reset_index(drop=True)
    try:
        gross = fsum(positives.asset_gain)
        top = fsum(selected.asset_gain)
    except OverflowError as error:
        raise PortfolioBacktestError("正向资产增量合计超出可靠范围", reason="numeric_range") from error
    share = top / gross if gross else None
    with np.errstate(under="ignore", invalid="ignore", divide="ignore"):
        selected["positive_gain_share"] = selected.asset_gain / gross if gross else pd.Series(dtype="float64")
    if (not isfinite(gross) or not isfinite(top) or (share is not None and not 0 < share <= 1)
            or not np.isfinite(selected.positive_gain_share).all()
            or selected.positive_gain_share.le(0).any()):
        raise PortfolioBacktestError("集中度占比超出可靠范围", reason="numeric_range")
    status = "no_return_interval" if len(changes) == 0 else ("available" if gross else "no_positive_gain")
    return GainConcentration(len(changes), len(positives), len(selected), gross, top, share, status, selected)
