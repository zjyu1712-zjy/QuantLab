"""从已有资金曲线派生年度片段统计；不重跑交易、不改原指标、不写档案。"""

import numpy as np
import pandas as pd

from quantlab.portfolio.types import PortfolioBacktestError


def _invalid():
    raise PortfolioBacktestError("年度统计需要升序、无缺失的正资产曲线", reason="invalid_portfolio_result")


def yearly_equity_statistics(curve: pd.DataFrame) -> pd.DataFrame:
    """按收益区间结束年份归属；后续年份借用上一资产点，绝不补日或年化。

    输入只是既有曲线，不认证数据来源或真实交易日完整性。无观测年份保留空行；
    跨整年缺口仍展示实际观测跨度收益，但明确标记它并非单年收益。
    """
    if (not isinstance(curve, pd.DataFrame) or curve.empty or not curve.columns.is_unique
            or not {"date", "equity"}.issubset(curve.columns)):
        _invalid()
    if curve.date.dtype != np.dtype("datetime64[ns]"):
        _invalid()
    dates = pd.DatetimeIndex(curve.date)
    if (dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing
            or not dates.equals(dates.normalize())):
        _invalid()
    # 曲线来自引擎的 float64 资产；不接受文本、布尔或复数的静默转换。
    if curve.equity.dtype != np.dtype("float64"):
        _invalid()
    values = curve.equity.to_numpy(copy=True)
    if not np.isfinite(values).all() or (values <= 0).any():
        _invalid()

    rows = []
    for year in range(dates[0].year, dates[-1].year + 1):
        positions = np.flatnonzero(dates.year == year)
        row = dict(year=year, base_date=pd.NaT, first_date=pd.NaT, last_date=pd.NaT,
                   observation_count=0, return_count=0, period_return=np.nan,
                   max_drawdown=np.nan, status="no_observations")
        if positions.size:
            first, last = int(positions[0]), int(positions[-1])
            base = max(0, first - 1)
            row.update(base_date=dates[base], first_date=dates[first], last_date=dates[last],
                       observation_count=len(positions), return_count=last - base,
                       status="observed_segment" if last > base else "no_return_interval")
            if last > base:
                segment = values[base:last + 1]
                with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                    period_return = segment[-1] / segment[0] - 1.0
                    drawdowns = segment / np.maximum.accumulate(segment) - 1.0
                if (not np.isfinite(period_return) or period_return <= -1
                        or not np.isfinite(drawdowns).all() or (drawdowns <= -1).any()):
                    raise PortfolioBacktestError("年度收益或回撤超出可靠数值范围", reason="numeric_range")
                row.update(period_return=float(period_return), max_drawdown=float(drawdowns.min()))
                if dates[base].year < year - 1:
                    row["status"] = "gap_spanning_years"
        rows.append(row)
    return pd.DataFrame(rows).astype({"base_date": "datetime64[ns]", "first_date": "datetime64[ns]",
                                     "last_date": "datetime64[ns]"})
