"""固定 20 条观测的均线偏离；描述价格位置，不预测反转或未来收益。"""

from math import fsum

import numpy as np
import pandas as pd

from quantlab.factors.base import BaseFactor, FactorDefinition, FactorEvaluation
from quantlab.factors.validation import validate_price_input


class Bias20Factor(BaseFactor):
    """当前收盘价相对含当期 20 条收盘价简单均值的偏离比例。"""

    name = "bias_20"
    definition = FactorDefinition(
        name=name, description="20 期均线偏离",
        formula="close[t] / mean(close[t-19:t+1]) - 1",
        version="1.0", required_fields=("close", "has_observation"),
        lookback_periods=19, window_observations=20,
        direction="higher_is_more_above_ma", missing_policy="strict_window",
        availability="after_close",
    )

    def evaluate(self, market_data: pd.DataFrame) -> FactorEvaluation:
        """验证输入后逐完整窗口计算；缺失不填充，热身期不借用区间外数据。"""
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
                # 同比例缩放不改变偏离率，避免先求原始价格总和时溢出。
                # 至少一个缩放值为 1，均值不会下溢；微小历史值下溢不独立判错。
                scaled = prices / prices.max()
                mean_scaled = fsum(scaled) / window
                candidate = scaled[-1] / mean_scaled - 1.0
                status[position] = "numeric_range"
                # 正价格的结果应大于 -1；不将浮点舍入误报为精确 -100%。
                if np.isfinite(candidate) and candidate > -1.0:
                    values[position] = candidate
                    status[position] = "valid"

        return FactorEvaluation(
            values=pd.Series(values, index=market_data.index.copy(), name=self.name, dtype="float64"),
            status=pd.Series(status, index=market_data.index.copy(), name="status", dtype="string"),
        )
