"""价格类因子共用的单 ETF 输入检查；不排序、填充或转换数据。"""

import re

import numpy as np
import pandas as pd


def validate_price_input(market_data: pd.DataFrame) -> None:
    """检查规范日线及观测标记，不改变调用方的表格。"""
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
