"""组合研究的配置、内部结果与规范表结构。"""

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


DECISION_DTYPES = {
    "decision_id": "string", "signal_date": "datetime64[ns]", "execution_date": "datetime64[ns]",
    "valid_count": "int64", "top_n": "int64", "status": "string", "reason": "string", "cutoff_tie": "bool",
}
TARGET_DTYPES = {
    "decision_id": "string", "signal_date": "datetime64[ns]", "symbol": "string",
    "factor_value": "float64", "target_weight": "float64",
}
ORDER_DTYPES = {
    "order_id": "string", "decision_id": "string", "symbol": "string", "signal_date": "datetime64[ns]",
    "execution_date": "datetime64[ns]", "side": "string", "requested_quantity": "int64",
    "filled_quantity": "int64", "unfilled_quantity": "int64", "status": "string", "reason": "string", "trade_id": "string",
}
TRADE_DTYPES = {
    "trade_id": "string", "order_id": "string", "decision_id": "string", "symbol": "string",
    "signal_date": "datetime64[ns]", "execution_date": "datetime64[ns]", "side": "string",
    "quantity": "int64", "open_price": "float64", "execution_price": "float64", "notional": "float64",
    "commission": "float64", "tax": "float64", "slippage_cost": "float64", "cash_after": "float64", "position_after": "int64",
}
POSITION_DTYPES = {
    "date": "datetime64[ns]", "symbol": "string", "quantity": "int64", "close": "float64",
    "position_value": "float64", "weight": "float64",
}
EQUITY_DTYPES = {
    "date": "datetime64[ns]", "cash": "float64", "position_value": "float64", "equity": "float64",
    "net_value": "float64", "cumulative_return": "float64", "cumulative_fees_taxes": "float64",
    "cumulative_slippage": "float64",
}


@dataclass(frozen=True)
class PortfolioBacktestResult:
    """内部引擎结果；公开研究入口仍需附加身份及独立输入快照。"""

    decisions: pd.DataFrame
    targets: pd.DataFrame
    orders: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    equity_curve: pd.DataFrame
    metadata: dict | None = None
    input_data: pd.DataFrame | None = None
    factor_data: pd.DataFrame | None = None


@dataclass(frozen=True)
class PortfolioPerformance:
    """组合指标；None 必须有明确原因，不表示零收益或失败回测。"""

    daily_returns: pd.Series
    net_value: pd.Series
    drawdown: pd.Series
    cumulative_return: float | None
    annualized_return: float | None
    annualized_volatility: float | None
    max_drawdown: float
    sharpe_ratio: float | None
    unavailable_reasons: dict[str, str]


@dataclass(frozen=True)
class PortfolioAnalysis:
    """独立的回测副本及基于完整资产曲线的分析，不修改原运行身份。"""

    backtest: PortfolioBacktestResult
    performance: PortfolioPerformance
