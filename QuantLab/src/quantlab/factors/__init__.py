"""第二阶段因子研究的扩展接口。

第一阶段模块不得依赖此包；这里暂不包含任何具体因子实现。
"""

from quantlab.factors.base import BaseFactor
from quantlab.factors.registry import FactorRegistry

__all__ = ["BaseFactor", "FactorRegistry"]
