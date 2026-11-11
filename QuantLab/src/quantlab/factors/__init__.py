"""第二阶段因子研究的统一说明、计算与显式注册接口。

第一阶段模块不得依赖此包；导入不会运行计算或建立全局注册表。
"""

from quantlab.factors.base import BaseFactor, FactorDefinition, FactorEvaluation
from quantlab.factors.bias import Bias20Factor
from quantlab.factors.drawdown import MaxDrawdown20Factor
from quantlab.factors.momentum import Momentum20Factor, Momentum60Factor, Momentum120Factor
from quantlab.factors.volatility import Volatility20Factor, Volatility60Factor
from quantlab.factors.volume import VolumeChange20Factor
from quantlab.factors.trend import TrendEfficiency20Factor
from quantlab.factors.registry import FactorRegistry, create_default_registry
from quantlab.factors.research import FactorRunResult, run_research_factor
from quantlab.factors.scoring import cross_sectional_rank_scores
from quantlab.factors.dual import DualFactorPreview, run_dual_factor_preview

__all__ = [
    "BaseFactor", "FactorDefinition", "FactorEvaluation", "FactorRegistry",
    "Momentum20Factor", "Momentum60Factor", "Momentum120Factor", "create_default_registry",
    "Volatility20Factor", "Volatility60Factor",
    "Bias20Factor",
    "MaxDrawdown20Factor",
    "VolumeChange20Factor",
    "TrendEfficiency20Factor",
    "FactorRunResult", "run_research_factor",
    "cross_sectional_rank_scores",
    "DualFactorPreview", "run_dual_factor_preview",
]
