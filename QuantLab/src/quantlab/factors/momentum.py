"""固定 20/60/120 期价格动量：按完整日期轴计算，缺失不填充、不跳过。"""

import re

import numpy as np
import pandas as pd

from quantlab.factors.base import BaseFactor, FactorDefinition, FactorEvaluation


def _validate_input(market_data: pd.DataFrame) -> None:
    """只验证一只 ETF 的规范输入；不排序、解析日期或修补价格。"""
    if not isinstance(market_data, pd.DataFrame) or market_data.empty:
        raise ValueError("因子行情输入必须为非空数据表")
    if not market_data.columns.is_unique:
        raise ValueError("因子行情输入不能有重复字段")
    required = ("date", "symbol", "close", "has_observation")
    missing = [field for field in required if field not in market_data.columns]
    if missing:
        raise ValueError(f"因子行情输入缺少字段：{', '.join(missing)}")
    if not market_data.index.is_unique:
        raise ValueError("单 ETF 因子输入的行索引必须唯一")

    symbols = market_data["symbol"]
    if not symbols.map(
        lambda value: isinstance(value, str) and re.fullmatch(r"[0-9]{6}", value) is not None
    ).all() or symbols.nunique() != 1:
        raise ValueError("因子输入必须是一只六位字符串代码的 ETF")

    dates = market_data["date"]
    if dates.dtype != np.dtype("datetime64[ns]"):
        raise ValueError("因子日期必须是无时区的 datetime64[ns] 日线")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("因子日期必须非空、唯一且升序")
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("因子日期不能包含盘中时间")

    observed = market_data["has_observation"]
    if not pd.api.types.is_bool_dtype(observed.dtype) or observed.isna().any():
        raise ValueError("观测标记必须为无空值布尔列")
    close = market_data["close"]
    if close.dtype != np.dtype("float64"):
        raise ValueError("收盘价必须保持规范 float64 类型")
    present = close.loc[observed].to_numpy(dtype="float64")
    if not np.isfinite(present).all() or (present <= 0).any():
        raise ValueError("观测行收盘价必须为有限正数")
    if not close.loc[~observed].isna().all():
        raise ValueError("缺失行收盘价必须为 NaN，不能填充")


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
        _validate_input(market_data)
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
