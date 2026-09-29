"""因子结果的显示转换；不计算公式、不访问磁盘、不修改研究结果。"""

from datetime import date

import altair as alt
import pandas as pd

from quantlab.factors import FactorDefinition, FactorRunResult


STATUS_LABELS = {
    "valid": "可计算",
    "missing_window": "窗口内缺失行情",
    "numeric_range": "数值超出可靠范围",
}


def _display_table(data: pd.DataFrame, definition: FactorDefinition) -> pd.DataFrame:
    """只格式化显示副本；NaN 显示为破折号，不能冒充 0%。"""
    periods = definition.lookback_periods
    labels = {**STATUS_LABELS, "warmup": f"历史不足（前 {periods} 个日期位置）"}
    return pd.DataFrame({
        "日期": data.date.dt.date,
        "ETF 代码": data.symbol,
        f"{periods} 期动量": data.factor_value.map(lambda value: "—" if pd.isna(value) else f"{value:.2%}"),
        "状态 / 原因": data.status.map(labels),
    }).reset_index(drop=True)


def daily_factor_table(result: FactorRunResult, selected_date: date) -> pd.DataFrame:
    """只取指定日期的全部 ETF，不查找各自最近一次有效值或按动量排名。"""
    if selected_date not in result.calendar.dates:
        raise ValueError("查看日期不在研究日期轴内")
    rows = result.data.loc[result.data.date.dt.date.eq(selected_date)]
    return _display_table(rows, result.definition).drop(columns="日期")


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
    """真实日期横轴、百分比纵轴；分组连线且保留单点区间。"""
    label = f"{_history_definition(history, definition).lookback_periods} 期动量"
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
        y=alt.Y("factor_value:Q", title=label, axis=alt.Axis(format=".1%")),
        detail="segment:N",
        tooltip=[alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"),
                 alt.Tooltip("factor_value:Q", title=label, format=".2%")],
    )
    zero = alt.Chart(alt.Data(values=[{"zero": 0}])).mark_rule(
        color="#9a9da4", strokeDash=[4, 4], strokeWidth=1,
    ).encode(y="zero:Q")
    return alt.layer(line, zero, data=data).properties(height=280)
