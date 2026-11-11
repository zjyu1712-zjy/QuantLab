"""20期带方向的价格路径效率；并非收益率或未来趋势预测。"""

from fractions import Fraction
import math

import numpy as np
import pandas as pd

from quantlab.factors.base import BaseFactor, FactorDefinition, FactorEvaluation
from quantlab.factors.validation import validate_price_input


def _window_efficiency(prices: np.ndarray) -> float:
    """用等价的净位移/逆向路程计算，避免总路程溢出及结果越界。"""
    net = float(prices[-1] - prices[0])
    if net == 0.0:
        # 恒价按定义为0；震荡回到原点的分子本来就是0。
        return 0.0
    changes = np.diff(prices)
    opposite = -changes[changes < 0] if net > 0 else changes[changes > 0]
    scale = max(abs(net), float(np.max(opposite)) if opposite.size else 0.0)
    scaled_net = abs(net) / scale
    # 总路程 = |净位移| + 2 × 逆向路程；先算价差再缩放，保留相邻大价格的小变动。
    distance = scaled_net + 2.0 * math.fsum(opposite / scale)
    candidate = math.copysign(scaled_net / distance, net)
    if abs(candidate) < np.finfo("float64").tiny:
        # 极小比值可能因多次舍入丢成0；仅在此边界用输入浮点值的精确有理数重算。
        points = [Fraction(float(value)) for value in prices]
        exact_distance = sum(abs(right - left) for left, right in zip(points, points[1:]))
        candidate = float((points[-1] - points[0]) / exact_distance)
        if candidate == 0.0:
            return math.nan  # 非零净位移仍不可表示，不能冒充真零效率。
    return candidate


class TrendEfficiency20Factor(BaseFactor):
    """有符号的净价格变化/总价格路程，恒价为0，向上连贯性高者默认靠前。"""

    name = "trend_efficiency_20"
    definition = FactorDefinition(
        name=name, description="20 期趋势效率",
        formula="0 if all prices equal else (close[t] - close[t-20]) / sum(abs(diff(close[t-20:t+1])))",
        version="1.0", required_fields=("close", "has_observation"),
        lookback_periods=20, window_observations=21,
        direction="higher_is_more_upward_efficient", missing_policy="strict_window",
        availability="after_close",
    )

    def evaluate(self, market_data: pd.DataFrame) -> FactorEvaluation:
        """检查完整21条价格窗口；缺失、热身、无法可靠表示的结果分别留空。"""
        validate_price_input(market_data)
        lookback = self.definition.lookback_periods
        window = self.definition.window_observations
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
                candidate = _window_efficiency(close[position - lookback:position + 1])
                status[position] = "numeric_range"
                if np.isfinite(candidate) and -1.0 <= candidate <= 1.0:
                    values[position] = candidate
                    status[position] = "valid"
        return FactorEvaluation(
            values=pd.Series(values, index=market_data.index.copy(), name=self.name, dtype="float64"),
            status=pd.Series(status, index=market_data.index.copy(), name="status", dtype="string"),
        )
