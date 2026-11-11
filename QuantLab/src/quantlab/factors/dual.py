"""固定动量/低波动双因子观察；不输出调仓目标，不写入研究档案。"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import date

import pandas as pd

from quantlab.data.research_sets import ResearchSetPreview
from quantlab.factors.research import run_research_factor
from quantlab.factors.scoring import cross_sectional_rank_scores


@dataclass(frozen=True)
class DualFactorPreview:
    """单日观察结果，独立于单因子及组合输入契约；表格是可修改的副本。"""

    data: pd.DataFrame
    selected_date: date
    common_count: int
    input_preview_sha256: str
    warnings: tuple[str, ...]


def run_dual_factor_preview(preview: ResearchSetPreview, selected_date: date) -> DualFactorPreview:
    """同一快照 → 固定两项因子 → 精确日期共同样本 → 各自评分 → 等权平均。

    20 期动量固定高值在前，20 期波动率固定低值在前；50/50 是因子评分
    权重，不是资产仓位。未进入共同样本的 ETF 仍保留原值与逐项原因，
    但两项评分及综合分均留空。少于两只共同有效 ETF 时不评分。
    """
    if type(selected_date) is not date:
        raise ValueError("查看日期必须为纯日期")
    snapshot = deepcopy(preview)
    # 沿用现有入口的快照、数值和指纹检查，不接受外部拼接的异源因子结果。
    momentum = run_research_factor(snapshot, "momentum_20")
    if selected_date not in momentum.calendar.dates:
        raise ValueError("查看日期不在研究日期轴内")
    volatility = run_research_factor(snapshot, "volatility_20")
    day = pd.Timestamp(selected_date)
    left = momentum.data.loc[momentum.data.date.eq(day), ["symbol", "factor_value", "status", "is_valid"]]
    right = volatility.data.loc[volatility.data.date.eq(day), ["symbol", "factor_value", "status", "is_valid"]]
    frame = left.rename(columns={"factor_value": "momentum_value", "status": "momentum_status",
                                 "is_valid": "momentum_valid"}).merge(
        right.rename(columns={"factor_value": "volatility_value", "status": "volatility_status",
                              "is_valid": "volatility_valid"}),
        on="symbol", how="inner", validate="one_to_one",
    )
    frame["is_common"] = frame.momentum_valid & frame.volatility_valid
    common_count = int(frame.is_common.sum())
    # 先取交集再分别排名，不能把两个不同有效池中的旧评分直接平均。
    frame["momentum_score"] = cross_sectional_rank_scores(
        frame.momentum_value.where(frame.is_common), ascending=False,
    )
    frame["volatility_score"] = cross_sectional_rank_scores(
        frame.volatility_value.where(frame.is_common), ascending=True,
    )
    frame["combined_score"] = .5 * frame.momentum_score + .5 * frame.volatility_score
    frame["rank"] = frame.combined_score.rank(method="min", ascending=False)
    frame = frame.drop(columns=["momentum_valid", "volatility_valid"]).sort_values(
        ["rank", "symbol"], na_position="last",
    ).reset_index(drop=True)
    return DualFactorPreview(
        data=frame, selected_date=selected_date, common_count=common_count,
        input_preview_sha256=momentum.input_preview_sha256,
        warnings=tuple(dict.fromkeys((*momentum.warnings, *volatility.warnings))),
    )
