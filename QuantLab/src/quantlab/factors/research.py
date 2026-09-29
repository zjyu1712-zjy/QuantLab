"""在已验证的研究快照上计算因子；纯内存运行，不回读或改写行情。"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from quantlab.data.calendar import CalendarAxis
from quantlab.data.research_sets import (
    SOURCE_FIELDS, ResearchSetPreview, validate_research_preview,
)
from quantlab.factors.base import FactorDefinition, FactorEvaluation
from quantlab.factors.registry import FactorRegistry, create_default_registry


@dataclass(frozen=True)
class FactorRunResult:
    """因子表和本次输入身份；表格/字典是独立副本，但并非不可修改。"""

    data: pd.DataFrame
    definition: FactorDefinition
    input_preview_sha256: str
    alignment_content_sha256: str
    start: date
    end: date
    calendar: CalendarAxis
    adjustment: str
    sources: tuple[dict[str, str], ...]
    warnings: tuple[str, ...]


def _validate_evaluation(evaluation: FactorEvaluation, expected_index: pd.Index) -> None:
    """拒绝损坏的因子输出，不能依靠 pandas 自动对齐来修补错行。"""
    if not isinstance(evaluation, FactorEvaluation):
        raise ValueError("因子输出必须包含数值和状态")
    values, status = evaluation.values, evaluation.status
    if not isinstance(values, pd.Series) or not isinstance(status, pd.Series):
        raise ValueError("因子输出数值与状态必须为 Series")
    if not values.index.identical(expected_index) or not status.index.identical(expected_index):
        raise ValueError("因子输出索引、顺序或长度与输入不一致")
    if values.dtype != np.dtype("float64"):
        raise ValueError("因子输出数值必须为 float64")
    allowed = {"warmup", "missing_window", "numeric_range", "valid"}
    if (not isinstance(status.dtype, pd.StringDtype)
            or status.isna().any() or not status.isin(allowed).all()):
        raise ValueError("因子输出状态无效")
    valid = status.eq("valid")
    if not np.isfinite(values.loc[valid].to_numpy(dtype="float64")).all():
        raise ValueError("因子输出有效行必须是有限数值")
    if not values.loc[~valid].isna().all():
        raise ValueError("因子输出不可用行必须为 NaN")


def _factor_warnings(
    preview: ResearchSetPreview, definition: FactorDefinition, has_valid: bool,
) -> tuple[str, ...]:
    """保留原输入限制，并解释本次动量周期的适用边界。"""
    periods = definition.lookback_periods
    messages = list(preview.warnings)
    messages.extend([
        f"{periods} 期按研究日期轴位置计算，不代表 {periods} 个自然日；并集可能漏掉全体缺失日。",
        f"前 {periods} 个日期位置为热身期；窗口内任一缺失都留空，不自动读取区间外历史。",
        "因子含当日收盘价，只能在收盘数据可用后计算，不是交易指令。",
        "未使用未来行不等于输入在历史当时可得；本结果是固定数据版本上的回顾性研究。",
        "同一复权标签不保证来源算法相同；口径声明不等于认证，未在此进行复权转换。",
    ])
    if preview.adjustment == "none":
        messages.append("未复权价格可能受分红、拆并份额影响，价格动量不能当作总回报。")
    if not has_valid:
        messages.append("本研究区间没有可用因子值，请核对历史长度和窗口内缺失情况。")
    return tuple(dict.fromkeys(messages))


def run_research_factor(
    preview: ResearchSetPreview, factor_name: str = "momentum_20", *,
    registry: FactorRegistry | None = None,
) -> FactorRunResult:
    """核验输入 → 逐 ETF 计算 → 按日期/代码合并 → 返回带追溯的副本。

    接受已载入快照或尚未保存的有效预览；不读取来源文件、不扩展历史，
    不更新已有指纹。结果只在内存中返回，不写入原研究集合。
    """
    if not isinstance(preview, ResearchSetPreview):
        raise ValueError("研究输入必须是 ResearchSetPreview")
    if registry is not None and not isinstance(registry, FactorRegistry):
        raise ValueError("因子注册表类型无效")
    snapshot = deepcopy(preview)
    # 索引名称不属于数据身份，但与 date 列同名会干扰旧校验器的 groupby。
    # 只在副本上清除这些名称；所有行值、顺序和既有指纹仍须通过完整校验。
    snapshot_data = getattr(snapshot.alignment, "data", None)
    if isinstance(snapshot_data, pd.DataFrame):
        snapshot_data.index = snapshot_data.index.set_names([None] * snapshot_data.index.nlevels)
    validate_research_preview(snapshot)
    active = create_default_registry() if registry is None else registry
    factor_type = active.get(factor_name)
    definition = deepcopy(factor_type.definition)
    factor = factor_type()

    # 行标签不属于研究数据身份；只重置副本，不改变调用方索引或任何行值。
    frame = snapshot.alignment.data.reset_index(drop=True)
    keys = frame[["date", "symbol"]].copy(deep=True)
    parts = []
    for source in snapshot.sources:
        asset = frame.loc[frame.symbol.eq(source["symbol"])].copy(deep=True)
        # 单独保留用于合并的键，避免具体因子改写它收到的计算表。
        asset_keys = asset[["date", "symbol"]].copy(deep=True)
        evaluation = factor.evaluate(asset.copy(deep=True))
        _validate_evaluation(evaluation, asset.index)
        asset_keys["factor_value"] = evaluation.values.copy(deep=True)
        asset_keys["status"] = evaluation.status.copy(deep=True)
        parts.append(asset_keys)

    calculated = pd.concat(parts, ignore_index=True)
    combined = keys.merge(
        calculated, on=["date", "symbol"], how="left", sort=False, validate="one_to_one",
    )
    if not combined[["date", "symbol"]].equals(keys) or combined.status.isna().any():
        raise ValueError("因子输出无法与原研究网格逐行对应")
    combined["is_valid"] = combined.status.eq("valid").astype(bool)
    data = combined[["date", "symbol", "factor_value", "is_valid", "status"]].copy(deep=True)
    sources = tuple(
        {key: source[key] for key in SOURCE_FIELDS if key != "processed_path"}
        for source in snapshot.sources
    )
    return FactorRunResult(
        data=data, definition=definition,
        input_preview_sha256=snapshot.preview_sha256,
        alignment_content_sha256=snapshot.alignment.content_sha256,
        start=snapshot.start, end=snapshot.end, calendar=deepcopy(snapshot.alignment.calendar),
        adjustment=snapshot.adjustment, sources=deepcopy(sources),
        warnings=_factor_warnings(snapshot, definition, bool(data.is_valid.any())),
    )
