"""固定 20/60/120 期价格动量：按完整日期轴计算，缺失不填充、不跳过。"""

import numpy as np
import pandas as pd

from quantlab.factors.base import BaseFactor, FactorDefinition, FactorEvaluation
from quantlab.factors.validation import validate_price_input


def _momentum_definition(periods: int) -> FactorDefinition:
    """由固定周期生成说明，避免公式、窗口和名称互相矛盾。"""
    return FactorDefinition(
        name=f"momentum_{periods}",
        description=f"{periods} 期价格动量",
        formula=f"close[t] / close[t-{periods}] - 1",
        version="1.0",
        required_fields=("close", "has_observation"),
        lookback_periods=periods,
        window_observations=periods + 1,
        direction="higher_is_stronger",
        missing_policy="strict_window",
        availability="after_close",
    )


class _MomentumFactor(BaseFactor):
    """三个固定周期共用的算法；子类只声明周期，不开放任意公式输入。"""

    def evaluate(self, market_data: pd.DataFrame) -> FactorEvaluation:
        """返回独立数值与状态；历史不足、窗口缺失、数值异常依次判定。"""
        validate_price_input(market_data)
        periods = self.definition.lookback_periods
        window = self.definition.window_observations
        close = market_data["close"].to_numpy(dtype="float64", copy=True)
        count = len(close)
        values = np.full(count, np.nan, dtype="float64")
        status = np.full(count, "warmup", dtype=object)
        status[periods:] = "missing_window"

        # 不只检查首尾：窗口中任一缺失都必须保留，不能跳过该行压缩日期轴。
        complete = (
            market_data["has_observation"].astype("int64")
            .rolling(window, min_periods=window).sum().eq(window)
        )
        positions = np.flatnonzero(complete.to_numpy())
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            candidate = close[positions] / close[positions - periods] - 1.0

        # 正价格的理论结果应大于 -1；极端比值舍入成 -1 时不伪装成精确 -100%。
        representable = np.isfinite(candidate) & (candidate > -1.0)
        status[positions] = "numeric_range"
        status[positions[representable]] = "valid"
        values[positions[representable]] = candidate[representable]

        return FactorEvaluation(
            values=pd.Series(
                values, index=market_data.index.copy(), name=self.name, dtype="float64",
            ),
            status=pd.Series(
                status, index=market_data.index.copy(), name="status", dtype="string",
            ),
        )


class Momentum20Factor(_MomentumFactor):
    """20 期动量，含当前期的 21 条观测须完整；保留原有名称与 1.0 口径。"""

    name = "momentum_20"
    definition = _momentum_definition(20)


class Momentum60Factor(_MomentumFactor):
    """60 期动量，含当前期的 61 条观测须完整。"""

    name = "momentum_60"
    definition = _momentum_definition(60)


class Momentum120Factor(_MomentumFactor):
    """120 期动量，含当前期的 121 条观测须完整。"""

    name = "momentum_120"
    definition = _momentum_definition(120)
