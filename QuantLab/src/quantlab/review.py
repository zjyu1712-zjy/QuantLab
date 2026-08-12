"""三方比较结果的结论与交易贡献分析。"""

from dataclasses import dataclass

import pandas as pd

from quantlab.comparison import ThreeWayComparison


@dataclass(frozen=True)
class ComparisonReview:
    """比较结论、成本拆解和主要盈亏交易。"""

    conclusions: dict[str, object]
    cost_attribution: pd.DataFrame
    top_winners: pd.DataFrame
    top_losers: pd.DataFrame


def _cost_row(
    comparison_name: str,
    observed_return_gap: float,
    left_cost: float,
    right_cost: float,
    initial_cash: float,
) -> dict[str, object]:
    extra_cost_rate = (left_cost - right_cost) / initial_cash
    estimated_cost_effect = -extra_cost_rate
    return {
        "comparison": comparison_name,
        "observed_return_gap": observed_return_gap,
        "extra_cost_rate": extra_cost_rate,
        "estimated_cost_effect": estimated_cost_effect,
        "remaining_gap": (
            observed_return_gap - estimated_cost_effect
        ),
    }


def _rank_personal_trades(
    matches: pd.DataFrame,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = matches.copy()
    ranked["return_on_cost"] = (
        ranked["realized_pnl"] / ranked["cost_basis"]
    )

    winners = (
        ranked.loc[ranked["realized_pnl"] > 0]
        .nlargest(top_n, "realized_pnl")
        .reset_index(drop=True)
    )
    losers = (
        ranked.loc[ranked["realized_pnl"] < 0]
        .nsmallest(top_n, "realized_pnl")
        .reset_index(drop=True)
    )
    return winners, losers


def review_comparison(
    comparison: ThreeWayComparison,
    *,
    top_n: int = 5,
) -> ComparisonReview:
    """回答核心比较问题，并列出主要盈利和亏损交易。"""
    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("展示交易数量必须是正整数")

    metrics = comparison.metrics.set_index("approach")
    strategy_gap = comparison.return_gaps[
        "strategy_minus_buy_and_hold"
    ]
    personal_gap = comparison.return_gaps[
        "personal_minus_strategy"
    ]

    conclusions = {
        "strategy_beats_buy_and_hold": strategy_gap > 0,
        "strategy_return_advantage": strategy_gap,
        "personal_beats_strategy": personal_gap > 0,
        "personal_return_advantage": personal_gap,
        "lowest_cost_approach": str(
            metrics["total_costs"].idxmin()
        ),
        "interpretation": (
            "成本影响为近似拆解；剩余差异还包含择时、仓位、"
            "成交价格和主观操作，需结合模型信号与交易备注判断。"
        ),
    }

    cost_attribution = pd.DataFrame(
        [
            _cost_row(
                "strategy_vs_buy_and_hold",
                strategy_gap,
                float(
                    metrics.loc[
                        "dual_moving_average",
                        "total_costs",
                    ]
                ),
                float(
                    metrics.loc["buy_and_hold", "total_costs"]
                ),
                comparison.initial_cash,
            ),
            _cost_row(
                "personal_vs_strategy",
                personal_gap,
                float(
                    metrics.loc["personal_trades", "total_costs"]
                ),
                float(
                    metrics.loc[
                        "dual_moving_average",
                        "total_costs",
                    ]
                ),
                comparison.initial_cash,
            ),
        ]
    )

    top_winners, top_losers = _rank_personal_trades(
        comparison.personal.fifo.matches,
        top_n,
    )

    return ComparisonReview(
        conclusions=conclusions,
        cost_attribution=cost_attribution,
        top_winners=top_winners,
        top_losers=top_losers,
    )
