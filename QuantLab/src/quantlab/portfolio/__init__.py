"""固定研究快照上的单因子组合回测；不联网、不自动交易或保存档案。"""

from quantlab.portfolio.research import run_factor_portfolio
from quantlab.portfolio.types import PortfolioBacktestError, PortfolioBacktestResult, PortfolioConfig

__all__ = ["PortfolioBacktestError", "PortfolioBacktestResult", "PortfolioConfig", "run_factor_portfolio"]
