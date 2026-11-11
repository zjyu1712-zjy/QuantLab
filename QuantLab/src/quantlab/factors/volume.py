"""当期成交量相对此前20期均量的变化；历史均量不包含当期。"""

import math

import numpy as np
import pandas as pd

from quantlab.factors.base import BaseFactor, FactorDefinition, FactorEvaluation
from quantlab.factors.validation import validate_volume_input


class VolumeChange20Factor(BaseFactor):
    """衡量相对放量/缩量，不解释为资金流向或未来收益。"""

    name = "volume_change_20"
    definition = FactorDefinition(
        name=name, description="20 期成交量变化",
        formula="volume[t] / mean(volume[t-20:t]) - 1",
        version="1.0", required_fields=("volume", "has_observation"),
        lookback_periods=20, window_observations=21,
        direction="higher_is_more_volume", missing_policy="strict_window",
        availability="after_close",
    )

    def evaluate(self, market_data: pd.DataFrame) -> FactorEvaluation:
        """校验完整窗口、按比例缩放计算比值、分清零分母与真实零成交量。"""
        validate_volume_input(market_data)
        lookback = self.definition.lookback_periods
        window = self.definition.window_observations
        volume = market_data["volume"].to_numpy(dtype="float64", copy=True)
        values = np.full(len(volume), np.nan, dtype="float64")
        status = np.full(len(volume), "warmup", dtype=object)
        status[lookback:] = "missing_window"
        complete = (
            market_data["has_observation"].astype("int64")
            .rolling(window, min_periods=window).sum().eq(window)
        )

        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            for position in np.flatnonzero(complete.to_numpy()):
                prior = volume[position - lookback:position]  # 不包含当期
                current = volume[position]
                scale = np.max(prior)
                status[position] = "numeric_range"
                if scale == 0.0:
                    # 非负历史全为0时分母为0；包括0/0，均不可计算。
                    continue
                # 不先计算原始总量或还原均量，避免溢出或极小均量下溢。
                scaled_mean = math.fsum(prior / scale) / lookback
                candidate = (current / scale) / scaled_mean - 1.0
                # 真零量可以是-100%；正量若舍入为-100%则不能伪装成真零。
                if np.isfinite(candidate) and (candidate > -1.0 or current == 0.0):
                    values[position] = candidate
                    status[position] = "valid"

        return FactorEvaluation(
            values=pd.Series(values, index=market_data.index.copy(), name=self.name, dtype="float64"),
            status=pd.Series(status, index=market_data.index.copy(), name="status", dtype="string"),
        )
