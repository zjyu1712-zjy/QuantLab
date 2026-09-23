"""第二阶段因子研究的统一说明、计算与显式注册接口。

第一阶段模块不得依赖此包；导入不会运行计算或建立全局注册表。
"""

from quantlab.factors.base import BaseFactor, FactorDefinition, FactorEvaluation
from quantlab.factors.momentum import Momentum20Factor
from quantlab.factors.registry import FactorRegistry, create_default_registry
from quantlab.factors.research import FactorRunResult, run_research_factor

__all__ = [
    "BaseFactor", "FactorDefinition", "FactorEvaluation", "FactorRegistry",
    "Momentum20Factor", "create_default_registry",
    "FactorRunResult", "run_research_factor",
]
