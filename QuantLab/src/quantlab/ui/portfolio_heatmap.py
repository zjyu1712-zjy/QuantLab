"""只将现有参数汇总转换成热力图，不运行回测或计算绩效。"""

from decimal import Decimal
from math import isfinite
from numbers import Integral, Real

import altair as alt
import pandas as pd

from quantlab.portfolio.sensitivity import INTERVAL_CANDIDATES, TOP_N_CANDIDATES
from quantlab.portfolio.types import PortfolioBacktestError
from quantlab.ui.portfolio_display import format_metric, portfolio_error_message


METRIC_LABELS = {
    "cumulative_return": "累计收益率", "annualized_return": "年化收益率",
    "annualized_volatility": "年化波动率", "max_drawdown": "最大回撤", "sharpe_ratio": "夏普比率",
}


def _heatmap_rows(result, metric):
    """每个 N/K 对应一个独立格子；不聚合、不补组、不修改原始数值。"""
    required = {"top_n", "rebalance_interval", "status", "trade_count", "failure_reason", "unavailable_reasons", metric}
    if (not isinstance(result, pd.DataFrame) or result.empty or not result.columns.is_unique
            or not required.issubset(result.columns)):
        raise ValueError("参数汇总缺少完整字段")
    for column, allowed in (("top_n", TOP_N_CANDIDATES), ("rebalance_interval", INTERVAL_CANDIDATES)):
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v not in allowed for v in result[column]):
            raise ValueError("参数网格不在已支持范围")
    ns = sorted(result.top_n.unique().tolist())
    expected = {(n, k) for n in ns for k in INTERVAL_CANDIDATES}
    pairs = list(zip(result.top_n, result.rebalance_interval))
    if len(set(pairs)) != len(pairs) or set(pairs) != expected:
        raise ValueError("参数网格有重复或缺失组合")
    rows = []
    for row in result.sort_values(["top_n", "rebalance_interval"]).itertuples(index=False):
        if row.status not in {"complete", "failed"}:
            raise ValueError("参数组状态无效")
        raw = getattr(row, metric)
        valid = (row.status == "complete" and isinstance(raw, Real) and not isinstance(raw, bool)
                 and isfinite(raw))
        value = float(raw) if valid else None
        if row.status == "failed":
            label, state = "失败", "运行失败"
            reason = portfolio_error_message(PortfolioBacktestError("参数组未完成", reason=row.failure_reason))
        elif not valid:
            label, state = "不可用", "指标不可用"
            reasons = row.unavailable_reasons if isinstance(row.unavailable_reasons, dict) else {}
            reason = reasons.get(metric, "此指标缺失或非有限，不能着色。")
        else:
            label = format_metric(value, percent=metric != "sharpe_ratio")
            # 极大数值用同单位科学计数法，避免格内标签溢出；底层值保持不变。
            if len(label) > 10 or "inf" in label.lower():
                label = (f"{Decimal(str(value)) * 100:.2E}%" if metric != "sharpe_ratio" else f"{value:.2e}")
            state = "完成（零成交）" if row.trade_count == 0 else "完成"
            reason = "没有发生成交；该指标仍有定义。" if row.trade_count == 0 else "—"
        rows.append(dict(top_n=int(row.top_n), rebalance_interval=int(row.rebalance_interval),
                         value=value, valid=valid, label=label, state=state, reason=reason))
    return rows, ns


def build_parameter_heatmap(result: pd.DataFrame, *, metric: str) -> alt.LayerChart:
    """单一蓝色色阶仅编码数值大小，灰格和文字区分失败/不可用。"""
    if not isinstance(metric, str) or metric not in METRIC_LABELS:
        raise ValueError("不支持的热力图指标")
    rows, ns = _heatmap_rows(result, metric)
    values = [row["value"] for row in rows if row["valid"]]
    # 含零尺度避免放大微小差异；全零/全缺失仍使用有限、非退化色域。
    low, high = min([0., *values]), max([0., *values])
    if low == high:
        high = 1.
    text_threshold = low * .3 + high * .7
    for row in rows:
        row["text_color"] = "#ffffff" if row["valid"] and row["value"] >= text_threshold else "#000000"
    number_format = ".2f" if metric == "sharpe_ratio" else ".2%"
    title = METRIC_LABELS[metric] + ("（无单位）" if metric == "sharpe_ratio" else "（%）")
    base = alt.Chart(alt.Data(values=rows)).encode(
        x=alt.X("rebalance_interval:O", title="调仓间隔（期）", sort=list(INTERVAL_CANDIDATES),
                axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=13),
                scale=alt.Scale(paddingInner=0, paddingOuter=0)),
        y=alt.Y("top_n:O", title="选择数量 N", sort=ns,
                axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=13),
                scale=alt.Scale(paddingInner=0, paddingOuter=0)),
        tooltip=[alt.Tooltip("top_n:O", title="选择数量 N"),
                 alt.Tooltip("rebalance_interval:O", title="调仓间隔（期）"),
                 alt.Tooltip("label:N", title=title),
                 alt.Tooltip("state:N", title="状态"), alt.Tooltip("reason:N", title="说明")],
    )
    tiles = base.mark_rect(stroke="#ffffff", strokeWidth=1, invalid="show").encode(
        color=alt.condition(
            "datum.valid", alt.Color("value:Q", title=title,
                scale=alt.Scale(domain=[low, high], range=["#e6f0fa", "#0b4778"], clamp=True, interpolate="rgb"),
                legend=alt.Legend(orient="bottom", format=number_format, gradientLength=180,
                                  titleLimit=240, labelLimit=110)), alt.value("#e5e7eb")),
    )
    labels = base.mark_text(fontSize=14).encode(
        text="label:N", color=alt.Color("text_color:N", scale=None, legend=None))
    # 只自适应宽度，避免 Streamlit 默认 fit 把图例/坐标轴挤进格子高度。
    return (tiles + labels).resolve_scale(color="independent").properties(
        width="container", height=max(140, 68 * len(ns)),
        autosize=alt.AutoSizeParams(type="fit-x", contains="padding"),
        title=f"参数热力图 · {METRIC_LABELS[metric]}",
    ).configure_view(stroke=None).configure_scale(invalid={"color": {"value": "#e5e7eb"}})
