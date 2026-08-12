"""第一阶段统一使用的默认参数。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalyticsConfig:
    """收益和风险指标配置。"""

    periods_per_year: int = 252
    annual_risk_free_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.periods_per_year <= 0:
            raise ValueError("每年期数必须大于0")

        if self.annual_risk_free_rate <= -1:
            raise ValueError("年化无风险利率必须大于-100%")


@dataclass(frozen=True)
class BacktestConfig:
    """第一阶段单 ETF 回测的默认配置。"""

    initial_cash: float = 100_000.0
    commission_rate: float = 0.0003
    minimum_commission: float = 0.0
    slippage_rate: float = 0.0001
    sell_tax_rate: float = 0.0
    lot_size: int = 100

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("初始资金必须大于0")

        non_negative_values = {
            "手续费率": self.commission_rate,
            "最低手续费": self.minimum_commission,
            "滑点率": self.slippage_rate,
            "卖出税率": self.sell_tax_rate,
        }

        for name, value in non_negative_values.items():
            if value < 0:
                raise ValueError(f"{name}不能为负数")

        if self.lot_size <= 0:
            raise ValueError("每手数量必须大于0")


DEFAULT_ANALYTICS_CONFIG = AnalyticsConfig()
DEFAULT_BACKTEST_CONFIG = BacktestConfig()
