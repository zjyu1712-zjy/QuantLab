"""固定20条价格窗口内的最大回撤；与组合全区间绩效指标独立。"""

import numpy as np
import pandas as pd

from quantlab.factors.base import BaseFactor, FactorDefinition, FactorEvaluation
from quantlab.factors.validation import validate_price_input


class MaxDrawdown20Factor(BaseFactor):
    """每个窗口从第一条价格重新起峰，保留窗口内最深的峰谷回落。"""

    name = "max_drawdown_20"
    definition = FactorDefinition(
        name=name, description="20 期价格最大回撤",
        formula="min(close[t-19:t+1] / cummax(close[t-19:t+1]) - 1)",
        version="1.0", required_fields=("close", "has_observation"),
        lookback_periods=19, window_observations=20,
        direction="higher_is_less_drawdown", missing_policy="strict_window",
        availability="after_close",
    )

    def evaluate(self, market_data: pd.DataFrame) -> FactorEvaluation:
        """只计算完整窗口；历史不足、窗口缺失、数值异常分别保留状态。"""
        validate_price_input(market_data)
        window = self.definition.window_observations
        lookback = self.definition.lookback_periods
        close = market_data["close"].to_numpy(dtype="float64", copy=True)
        values = np.full(len(close), np.nan, dtype="float64")
        status = np.full(len(close), "warmup", dtype=object)
        status[lookback:] = "missing_window"
        complete = (
            market_data["has_observation"].astype("int64")
            .rolling(window, min_periods=window).sum().eq(window)
        )

        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            for position in np.flatnonzero(complete.to_numpy()):
                prices = close[position - lookback:position + 1]
                # 每个窗口重新累计峰值，既不继承窗口外高点，也不提前使用后面的高点。
                peaks = np.maximum.accumulate(prices)
                # 比值不超过1，无需求和或整体缩放；避免小价格缩放为0后产生0/0。
                candidate = np.min(prices / peaks - 1.0)
                status[position] = "numeric_range"
                # 正价格不应有精确-100%回撤；中途出现不可表示的深跌也不能被末期反弹掩盖。
                if np.isfinite(candidate) and -1.0 < candidate <= 0.0:
                    values[position] = candidate
                    status[position] = "valid"

        return FactorEvaluation(
            values=pd.Series(values, index=market_data.index.copy(), name=self.name, dtype="float64"),
            status=pd.Series(status, index=market_data.index.copy(), name="status", dtype="string"),
        )
