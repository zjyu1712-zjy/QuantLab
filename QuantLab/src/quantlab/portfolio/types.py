"""组合研究的配置与数值目标类型；当前不包含成交或资产曲线。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite
from numbers import Integral, Real

import pandas as pd

from quantlab.config.settings import BacktestConfig, DEFAULT_BACKTEST_CONFIG


class PortfolioBacktestError(ValueError):
    """带稳定原因和可选日期/代码的错误，不携带部分回测结果。"""

    def __init__(self, message: str, *, date: date | None = None,
                 symbols: tuple[str, ...] = (), reason: str) -> None:
        super().__init__(message)
        self.date = date
        self.symbols = symbols
        self.reason = reason


def _invalid_config(message: str) -> None:
    raise PortfolioBacktestError(message, reason="invalid_config")


@dataclass(frozen=True)
class PortfolioConfig:
    """固定间隔单因子等权配置；N 的资产数量上限在研究边界校验。"""

    top_n: int
    rebalance_interval: int = 20
    ascending: bool | None = None
    backtest: BacktestConfig = DEFAULT_BACKTEST_CONFIG

    def __post_init__(self) -> None:
        for label, value in (("选择数量 N", self.top_n), ("调仓间隔", self.rebalance_interval)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                _invalid_config(f"{label}必须为正整数，不能为布尔值")
        if self.ascending is not None and type(self.ascending) is not bool:
            _invalid_config("排序方向必须为布尔值或默认值 None")
        if not isinstance(self.backtest, BacktestConfig):
            _invalid_config("交易配置必须为 BacktestConfig")
        lot_size = self.backtest.lot_size
        if isinstance(lot_size, bool) or not isinstance(lot_size, Integral) or lot_size <= 0:
            _invalid_config("每手份额必须为正整数，不能为布尔值")
        for name, label in (("initial_cash", "初始现金"), ("commission_rate", "佣金率"),
                            ("minimum_commission", "最低佣金"), ("slippage_rate", "滑点率"),
                            ("sell_tax_rate", "卖出税率")):
            value = getattr(self.backtest, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                _invalid_config(f"{label}必须为有限数值，不能为布尔值或文本")
            try:
                finite = isfinite(value)
            except OverflowError:
                finite = False
            if not finite or value < 0:
                _invalid_config(f"{label}必须为非负有限数值")
        if self.backtest.initial_cash == 0:
            _invalid_config("初始现金必须大于零")
        if self.backtest.slippage_rate >= 1:
            _invalid_config("滑点率必须小于 1")


@dataclass(frozen=True)
class RebalanceSchedule:
    """内部调仓安排；两张表为独立副本，但仍是可变 DataFrame。"""

    decisions: pd.DataFrame
    targets: pd.DataFrame
    ascending: bool
