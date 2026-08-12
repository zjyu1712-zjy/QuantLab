"""因子计算的最小抽象接口。"""

from abc import ABC, abstractmethod

import pandas as pd


class BaseFactor(ABC):
    """所有第二阶段因子需要遵循的统一接口。"""

    name: str

    @abstractmethod
    def compute(self, market_data: pd.DataFrame) -> pd.Series:
        """根据标准化行情计算因子值。"""
