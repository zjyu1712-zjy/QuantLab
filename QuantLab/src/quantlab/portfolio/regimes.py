"""依据同区间研究池基准涨跌做事后分组，不识别牛熊、不重跑策略。"""

from math import fsum, isfinite

import numpy as np
import pandas as pd

from quantlab.portfolio.types import PortfolioBacktestError


def _invalid():
    raise PortfolioBacktestError("分组需要完整、同轴的一次买入持有基准与策略曲线", reason="invalid_comparison")


def _numeric():
    raise PortfolioBacktestError("分组收益或均值超出可靠数值范围", reason="numeric_range")


def _curve_values(curve):
    if (not isinstance(curve, pd.DataFrame) or curve.empty or not curve.columns.is_unique
            or not {"date", "equity"}.issubset(curve.columns)
            or curve.date.dtype != np.dtype("datetime64[ns]")
            or curve.equity.dtype != np.dtype("float64")):
        _invalid()
    dates = pd.DatetimeIndex(curve.date)
    values = curve.equity.to_numpy(copy=True)
    if (dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing
            or not dates.equals(dates.normalize()) or not np.isfinite(values).all() or (values <= 0).any()):
        _invalid()
    return dates, values


def _returns(values):
    # 差额/期初资产与比值减一等价，但避免极小的非零变动被先舍入为比值 1。
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        result = np.diff(values) / values[:-1]
    if (not np.isfinite(result).all() or (result <= -1).any()
            or ((values[1:] != values[:-1]) & (result == 0)).any()):
        _numeric()
    return result


def _mean(values):
    try:
        total = fsum(values)
        result = total / len(values)
    except OverflowError:
        _numeric()
    if not isfinite(result) or (total != 0 and result == 0):
        _numeric()
    return result


def benchmark_regime_statistics(strategy: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """只统计持仓→持仓区间；现金及首次建仓单列计数、均值留空。

    两条曲线必须已经通过外层研究身份核对；本函数再验证日期、资产与基准持仓。
    输入从初始现金开始；基准只允许零或一次建仓，拒绝退出/再次建仓。
    不补日期、不丢坏值、不拼接非连续片段，计数覆盖所有原始相邻观测区间。
    """
    dates, strategy_values = _curve_values(strategy)
    benchmark_dates, benchmark_values = _curve_values(benchmark)
    if not dates.equals(benchmark_dates):
        _invalid()
    if "position_value" not in benchmark or benchmark.position_value.dtype != np.dtype("float64"):
        _invalid()
    positions = benchmark.position_value.to_numpy(copy=True)
    if (not np.isfinite(positions).all() or (positions < 0).any()
            or (positions > benchmark_values).any() or positions[0] != 0):
        _invalid()
    held = positions > 0
    if (held[:-1] & ~held[1:]).any():
        _invalid()
    strategy_returns, benchmark_returns = _returns(strategy_values), _returns(benchmark_values)
    eligible = held[:-1] & held[1:]
    masks = {
        "up": eligible & (benchmark_returns > 0),
        "down": eligible & (benchmark_returns < 0),
        "flat": eligible & (benchmark_returns == 0),
        "cash": ~held[:-1] & ~held[1:],
        "entry": ~held[:-1] & held[1:],
    }
    rows = []
    for group, mask in masks.items():
        count = int(mask.sum())
        row = dict(group=group, interval_count=count, strategy_mean_return=np.nan,
                   benchmark_mean_return=np.nan, mean_difference_pp=np.nan)
        if group in ("up", "down", "flat") and count:
            left, right = strategy_returns[mask], benchmark_returns[mask]
            with np.errstate(over="ignore", invalid="ignore"):
                differences = left - right
            if not np.isfinite(differences).all():
                _numeric()
            difference_pp = _mean(differences) * 100
            if not isfinite(difference_pp):
                _numeric()
            row.update(strategy_mean_return=_mean(left), benchmark_mean_return=_mean(right),
                       mean_difference_pp=difference_pp)
        rows.append(row)
    return pd.DataFrame(rows)
