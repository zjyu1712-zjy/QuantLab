"""因子结果的显示转换；不计算公式、不访问磁盘、不修改研究结果。"""

from dataclasses import dataclass
from datetime import date
from numbers import Integral

import altair as alt
import pandas as pd

from quantlab.factors import FactorDefinition, FactorRunResult, cross_sectional_rank_scores


STATUS_LABELS = {
    "valid": "可计算",
    "missing_window": "窗口内缺失行情",
    "numeric_range": "数值超出可靠范围",
}


def factor_label(definition: FactorDefinition) -> str:
    """使用定义中的名称，保留原动量表格的简写。"""
    return definition.description.replace("价格动量", "动量")


def factor_explanation(definition: FactorDefinition) -> str:
    """说明数值的方向，不把历史波动解释成涨幅或交易优劣。"""
    if definition.direction == "higher_is_more_volatile":
        return "数值越大表示历史收益率波动越大；低波动不保证安全，也不代表收益更高。"
    if definition.direction == "higher_is_stronger":
        return "数值越大只表示过去涨幅更高。"
    if definition.direction == "higher_is_more_above_ma":
        return "正值在均线上方，负值在下方，零表示相等；数值越大只表示相对均线位置越高，不代表未来收益或投资优劣。"
    if definition.direction == "higher_is_less_drawdown":
        return "数值为零或负数，越接近 0 表示窗口内价格最大回撤越浅；不代表未来更安全，也不是组合全区间回撤。"
    if definition.direction == "higher_is_more_volume":
        return "正值表示成交量高于此前均量，负值表示低于、零表示相等；数值越大仅表示相对放量，不是资金净流入，也不代表未来收益更高。"
    if definition.direction == "higher_is_more_upward_efficient":
        return "数值在 -1 到 1 之间；越接近 1 向上越连贯、越接近 -1 向下越连贯，首尾相等或恒价为 0。这是带方向的趋势效率，不是收益率或未来走势预测。"
    return "数值含义请核对完整因子定义。"


def _display_table(data: pd.DataFrame, definition: FactorDefinition) -> pd.DataFrame:
    """只格式化副本；趋势效率显示数字，其余因子显示百分比，空值不冒充零。"""
    periods = definition.lookback_periods
    labels = {**STATUS_LABELS, "warmup": f"历史不足（前 {periods} 个日期位置）"}
    if definition.direction == "higher_is_more_volume":
        labels["numeric_range"] = "历史均量为 0 或数值超出可靠范围"
    number_format = ".2f" if definition.direction == "higher_is_more_upward_efficient" else ".2%"
    return pd.DataFrame({
        "日期": data.date.dt.date,
        "ETF 代码": data.symbol,
        factor_label(definition): data.factor_value.map(lambda value: "—" if pd.isna(value) else format(value, number_format)),
        "状态 / 原因": data.status.map(labels),
    }).reset_index(drop=True)


def _daily_rows(result: FactorRunResult, selected_date: date) -> pd.DataFrame:
    """只复制指定日期，不回退到之前的有效值，也不修改计算结果。"""
    if selected_date not in result.calendar.dates:
        raise ValueError("查看日期不在研究日期轴内")
    return result.data.loc[result.data.date.dt.date.eq(selected_date)].copy(deep=True).reset_index(drop=True)


def daily_factor_table(result: FactorRunResult, selected_date: date) -> pd.DataFrame:
    """按 ETF 代码显示指定日期的全部 ETF，不排名。"""
    rows = _daily_rows(result, selected_date).sort_values("symbol")
    return _display_table(rows, result.definition).drop(columns="日期")


def _ranked_daily_rows(
    result: FactorRunResult, selected_date: date, *, ascending: bool | None = None,
) -> pd.DataFrame:
    """保留原始值的共用排序，供排名和筛选使用，不能从显示字符串反推。"""
    if ascending is None:
        ascending = result.definition.direction == "higher_is_more_volatile"
    rows = _daily_rows(result, selected_date)
    # run_research_factor 已保证有效值有限、不可用值为 NaN。
    # 先排名再格式化，避免百分比舍入造成假并列；有效的 0 正常参与。
    rows["rank"] = rows.factor_value.where(rows.is_valid).rank(method="min", ascending=ascending)
    return rows.sort_values(["rank", "symbol"], na_position="last").reset_index(drop=True)


def ranked_factor_table(
    result: FactorRunResult, selected_date: date, *, ascending: bool | None = None,
) -> pd.DataFrame:
    """单日单因子排名与相对评分；评分只附加展示，不改变排序和筛选。"""
    if ascending is None:
        ascending = result.definition.direction == "higher_is_more_volatile"
    rows = _ranked_daily_rows(result, selected_date, ascending=ascending)
    scores = cross_sectional_rank_scores(rows.factor_value.where(rows.is_valid), ascending=ascending)
    table = _display_table(rows, result.definition).drop(columns="日期")
    table.insert(0, "名次", rows["rank"].map(lambda rank: "—" if pd.isna(rank) else str(int(rank))))
    table.insert(3, "相对评分（0–100）", scores.map(lambda value: "—" if pd.isna(value) else f"{value:.2f}"))
    return table


@dataclass(frozen=True)
class TopNPreview:
    """只供当前页面展示的目标配置，不是持仓、订单或组合回测输入契约。"""

    table: pd.DataFrame
    valid_count: int
    cutoff_tie: bool


def top_n_factor_preview(
    result: FactorRunResult, selected_date: date, n: int, *, ascending: bool | None = None,
) -> TopNPreview:
    """恰好取 N 个有效席位；数量不足时返回空表，权重保留未舍入小数。"""
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("选择数量 N 必须为正整数")
    rows = _ranked_daily_rows(result, selected_date, ascending=ascending)
    valid = rows.loc[rows.is_valid].reset_index(drop=True)
    enough = len(valid) >= n
    selected = valid.iloc[:n if enough else 0].copy()
    # 同值的并列名次可跳号；截取的是排好序的席位，不是 rank <= N。
    cutoff_tie = bool(enough and len(valid) > n and
                      valid.factor_value.iloc[n - 1] == valid.factor_value.iloc[n])
    table = _display_table(selected, result.definition).drop(columns=["日期", "状态 / 原因"])
    table.insert(0, "名次", selected["rank"].map(lambda rank: str(int(rank))))
    table["目标权重"] = 1.0 / n
    return TopNPreview(table=table, valid_count=len(valid), cutoff_tie=cutoff_tie)


def factor_history(result: FactorRunResult, symbol: str) -> pd.DataFrame:
    """保留全部日期和空值，并给连续有效区间分段，阻止曲线跨缺口连接。"""
    data = result.data.loc[result.data.symbol.eq(symbol)].copy(deep=True).reset_index(drop=True)
    if data.empty:
        raise ValueError("研究集合中没有所选 ETF")
    data["segment"] = (~data.is_valid).cumsum()
    data.attrs["factor_definition"] = result.definition
    return data


def _history_definition(
    history: pd.DataFrame, definition: FactorDefinition | None,
) -> FactorDefinition:
    """从历史恢复不可变的因子说明；缺失或冲突时拒绝猜测周期。"""
    stored = history.attrs.get("factor_definition")
    if definition is None:
        definition = stored
    if not isinstance(definition, FactorDefinition):
        raise ValueError("历史数据缺少因子说明，请使用 factor_history() 或显式提供说明")
    if stored is not None and stored != definition:
        raise ValueError("显式因子说明与历史数据携带的说明不一致")
    return definition


def history_factor_table(
    history: pd.DataFrame, definition: FactorDefinition | None = None,
) -> pd.DataFrame:
    """图下的精确值与原因明细，和曲线使用同一只 ETF 的完整历史。"""
    return _display_table(history, _history_definition(history, definition)).drop(columns="ETF 代码")


def build_factor_chart(
    history: pd.DataFrame, definition: FactorDefinition | None = None,
) -> alt.LayerChart:
    """真实日期横轴，按因子含义格式化纵轴；保留缺口和单点区间。"""
    definition = _history_definition(history, definition)
    label = factor_label(definition)
    is_efficiency = definition.direction == "higher_is_more_upward_efficient"
    rows = [
        {"date": row.date.isoformat(),
         "factor_value": float(row.factor_value) if row.is_valid else None,
         "segment": int(row.segment)}
        for row in history.itertuples(index=False)
    ]
    # 显式传递可序列化记录，避免长区间受到 Altair 默认 5000 行阈值限制。
    data = alt.Data(values=rows)
    domain = [rows[0]["date"], rows[-1]["date"]]
    base = alt.Chart(data)
    line = base.mark_line(color="#4b8df8", strokeWidth=2, point=alt.OverlayMarkDef(size=26)).encode(
        x=alt.X("date:T", title="日期", scale=alt.Scale(domain=domain),
                axis=alt.Axis(format="%Y-%m-%d", labelAngle=0, tickCount=4)),
        y=alt.Y("factor_value:Q", title=label, axis=alt.Axis(format=".2f" if is_efficiency else ".1%")),
        detail="segment:N",
        tooltip=[alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"),
                 alt.Tooltip("factor_value:Q", title=label, format=".2f" if is_efficiency else ".2%")],
    )
    zero = alt.Chart(alt.Data(values=[{"zero": 0}])).mark_rule(
        color="#9a9da4", strokeDash=[4, 4], strokeWidth=1,
    ).encode(y="zero:Q")
    return alt.layer(line, zero, data=data).properties(height=280)
