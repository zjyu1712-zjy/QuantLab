"""单日单因子横截面排名分；不计算因子、不读数据、不参与组合调仓。"""

import numpy as np
import pandas as pd


def cross_sectional_rank_scores(values: pd.Series, *, ascending: bool) -> pd.Series:
    """将同一横截面的原值转换为 0–100 分，保留索引及缺失位置。

    输入沿用因子结果的 float64 契约：有限值参与、NaN 排除，调用方须先
    选定日期并屏蔽无效项。方向必须显式传入，不能由这里猜测因子含义。
    至少两个有效值才评分；并列取平均名次，全部相同自然得到 50 分。
    评分只表示所选方向的相对位置，不表示价差大小、收益率或上涨概率。
    """
    if not isinstance(values, pd.Series) or values.dtype != np.dtype("float64"):
        raise ValueError("横截面原值必须为 float64 Series")
    if type(ascending) is not bool:
        raise ValueError("评分方向必须为布尔值")
    if np.isinf(values.to_numpy()).any():
        raise ValueError("横截面原值只允许有限值或 NaN")
    count = int(values.notna().sum())
    if count < 2:
        return pd.Series(np.nan, index=values.index, dtype="float64", name="relative_score")
    # 不复用页面的最小并列名次，否则全同值会错误地全得 100 分。
    average_rank = values.rank(method="average", ascending=ascending)
    scores = 100.0 * (count - average_rank) / (count - 1)
    return scores.rename("relative_score")
