"""因子说明及统一计算接口；不执行文本公式，不读取行情文件。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import pandas as pd


@dataclass(frozen=True)
class FactorDefinition:
    """稳定的因子说明书；公式是说明文本，不是待执行的代码。"""

    name: str
    description: str
    formula: str
    version: str
    required_fields: tuple[str, ...]
    lookback_periods: int
    window_observations: int
    direction: str
    missing_policy: str
    availability: str

    def __post_init__(self) -> None:
        """拒绝有歧义的说明，保留原值，不静默去空格或转换类型。"""
        for label, value in (
            ("名称", self.name),
            ("描述", self.description),
            ("公式", self.formula),
            ("版本", self.version),
            ("方向", self.direction),
            ("缺失规则", self.missing_policy),
            ("可用时点", self.availability),
        ):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"因子{label}必须为非空且无两端空白的字符串")

        if not isinstance(self.required_fields, tuple) or not self.required_fields:
            raise ValueError("因子所需字段必须为非空元组")
        for field in self.required_fields:
            if not isinstance(field, str) or not field or field != field.strip():
                raise ValueError("因子所需字段必须为非空且无两端空白的字符串")
        if len(set(self.required_fields)) != len(self.required_fields):
            raise ValueError("因子所需字段不能重复")

        for label, value in (
            ("回看周期", self.lookback_periods),
            ("窗口观测数", self.window_observations),
        ):
            # bool 是 int 的子类，不能用 isinstance(value, int) 接受 True/False。
            if type(value) is not int or value <= 0:
                raise ValueError(f"因子{label}必须为正整数，不能为布尔值")
        if (
            self.missing_policy == "strict_window"
            and self.window_observations != self.lookback_periods + 1
        ):
            raise ValueError("严格窗口观测数必须包含当前期和回看周期")


@dataclass(frozen=True)
class FactorEvaluation:
    """数值与逐行状态，索引应与输入一致。

    冻结字段不意味着其中的 Series 不可修改。具体因子须返回独立序列，
    研究入口会进一步核对输出契约；此容器不转换或修补计算结果。
    """

    values: pd.Series
    status: pd.Series


class BaseFactor(ABC):
    """具体因子只实现 evaluate，compute 复用同一套计算逻辑。"""

    name: ClassVar[str]
    definition: ClassVar[FactorDefinition]

    def compute(self, market_data: pd.DataFrame) -> pd.Series:
        """保留纯数值调用方式，返回独立的数值序列副本。"""
        return self.evaluate(market_data).values.copy(deep=True)

    @abstractmethod
    def evaluate(self, market_data: pd.DataFrame) -> FactorEvaluation:
        """返回同索引数值和状态；不可用值为 NaN，不修改输入。"""
        raise NotImplementedError
