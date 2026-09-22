"""第二阶段因子研究的统一说明、计算与显式注册接口。

第一阶段模块不得依赖此包；这里暂不包含任何具体因子实现。
"""

from quantlab.factors.base import BaseFactor, FactorDefinition, FactorEvaluation
from quantlab.factors.registry import FactorRegistry

__all__ = ["BaseFactor", "FactorDefinition", "FactorEvaluation", "FactorRegistry"]
