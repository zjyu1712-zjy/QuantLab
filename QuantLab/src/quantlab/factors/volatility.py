"""20/60 期滚动年化波动率：简单收益率样本标准差，固定每年 252 期。"""

from math import sqrt

import numpy as np
import pandas as pd

from quantlab.factors.base import BaseFactor, FactorDefinition, FactorEvaluation
from quantlab.factors.validation import validate_price_input


def _volatility_definition(periods: int) -> FactorDefinition:
    return FactorDefinition(
        name=f"volatility_{periods}", description=f"{periods} 期年化波动率",
        formula=(f"r[t] = close[t] / close[t-1] - 1; "
                 f"std(r[t-{periods - 1}:t+1], ddof=1) * sqrt(252)"),
        version="1.0", required_fields=("close", "has_observation"),
        lookback_periods=periods, window_observations=periods + 1,
        direction="higher_is_more_volatile", missing_policy="strict_window",
        availability="after_close",
    )


class _VolatilityFactor(BaseFactor):
    """固定周期共用算法；不把缺失、数值异常当成零波动。"""

    def evaluate(self, market_data: pd.DataFrame) -> FactorEvaluation:
        validate_price_input(market_data)
        periods = self.definition.lookback_periods
        window = self.definition.window_observations
        close = market_data.close.to_numpy(dtype="float64", copy=True)
        values = np.full(len(close), np.nan, dtype="float64")
        status = np.full(len(close), "warmup", dtype=object)
        status[periods:] = "missing_window"
        complete = (
            market_data.has_observation.astype("int64")
            .rolling(window, min_periods=window).sum().eq(window).to_numpy()
        )
        positions = np.flatnonzero(complete)
        if positions.size:
            # N 个相邻收益率必须来自同一个完整的 N+1 价格窗口。
            windows = np.lib.stride_tricks.sliding_window_view(close, window)[positions - periods]
            status[positions] = "numeric_range"
            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                returns = windows[:, 1:] / windows[:, :-1] - 1.0
                reliable = (np.isfinite(returns) & (returns > -1.0)).all(axis=1)
                returns = returns[reliable]
                positions = positions[reliable]
                # 缩放后求标准差，避免收益率平方溢出；不是裁剪或截断收益率。
                scale = np.max(np.abs(returns), axis=1)
                scaled = np.divide(returns, scale[:, None], out=np.zeros_like(returns),
                                   where=scale[:, None] != 0)
                candidate = np.std(scaled, axis=1, ddof=1) * scale * sqrt(252)
            valid = np.isfinite(candidate) & (candidate >= 0)
            status[positions[valid]] = "valid"
            values[positions[valid]] = candidate[valid]
        return FactorEvaluation(
            values=pd.Series(values, index=market_data.index.copy(), name=self.name, dtype="float64"),
            status=pd.Series(status, index=market_data.index.copy(), name="status", dtype="string"),
        )


class Volatility20Factor(_VolatilityFactor):
    """20 个收益率、21 个收盘价；含当日收盘，仅描述历史波动。"""

    name = "volatility_20"
    definition = _volatility_definition(20)


class Volatility60Factor(_VolatilityFactor):
    """60 个收益率、61 个收盘价；含当日收盘，仅描述历史波动。"""

    name = "volatility_60"
    definition = _volatility_definition(60)
