"""组合档案 v1 的纯数据编解码及可移植校验；不读源文件、不重跑研究。"""

from copy import deepcopy
from dataclasses import fields
from datetime import date, datetime, timezone
from math import fsum, isfinite
import re
from uuid import UUID

import numpy as np
import pandas as pd

from quantlab.backtest.provenance import dataframe_records, dataframe_sha256
from quantlab.config.settings import AnalyticsConfig, BacktestConfig
from quantlab.data.research_sets import SOURCE_FIELDS
from quantlab.factors.base import FactorDefinition
from quantlab.portfolio.types import (
    DECISION_DTYPES, TARGET_DTYPES, ORDER_DTYPES, TRADE_DTYPES, POSITION_DTYPES, EQUITY_DTYPES,
    PortfolioAnalysis, PortfolioBacktestResult, PortfolioConfig, PortfolioPerformance,
)


TABLES = {
    "decisions": DECISION_DTYPES, "targets": TARGET_DTYPES, "orders": ORDER_DTYPES,
    "trades": TRADE_DTYPES, "positions": POSITION_DTYPES, "equity_curve": EQUITY_DTYPES,
    "input_data": {"date": "datetime64[ns]", "symbol": "text", "open": "float64", "high": "float64",
                   "low": "float64", "close": "float64", "volume": "float64", "has_observation": "bool"},
    "factor_data": {"date": "datetime64[ns]", "symbol": "text", "factor_value": "float64",
                    "is_valid": "bool", "status": "string"},
}
METRICS = ("cumulative_return", "annualized_return", "annualized_volatility", "max_drawdown", "sharpe_ratio")
SERIES = {"daily_returns": "daily_return", "net_value": "net_value", "drawdown": "drawdown"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(value, expected):
    require(isinstance(value, dict) and set(value) == set(expected), "档案字段不符合 v1 白名单")


def canonical_id(value):
    require(isinstance(value, str), "运行编号必须是规范 UUID")
    try:
        require(str(UUID(value)) == value, "运行编号必须是规范 UUID")
    except (ValueError, AttributeError) as error:
        raise ValueError("运行编号必须是规范 UUID") from error
    return value


def utc_time(value):
    require(isinstance(value, str), "时间必须是带 UTC 时区的 ISO 文本")
    try:
        parsed = datetime.fromisoformat(value)
        require(parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed),
                "时间必须记录 UTC 时区")
    except (ValueError, TypeError) as error:
        raise ValueError("时间必须是带 UTC 时区的 ISO 文本") from error
    return parsed


def text(value):
    require(isinstance(value, str) and bool(value.strip()), "文本字段必须非空")


def number(value):
    require(type(value) in (int, float), "数值字段不接受文本或布尔值")
    try:
        require(isfinite(value), "数值必须有限")
    except OverflowError as error:
        raise ValueError("数值超出支持范围") from error


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "整数参数无效")


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "内容指纹格式无效")


def iso_date(value):
    require(isinstance(value, str), "日期必须是 ISO 日期文本")
    try:
        require(date.fromisoformat(value).isoformat() == value, "日期必须规范化")
    except ValueError as error:
        raise ValueError("日期必须是 ISO 日期文本") from error
    return value


def _daily_timestamp(value):
    require(isinstance(value, str), "表格日期必须是 ISO 文本")
    try:
        result = pd.Timestamp(value)
        require(not pd.isna(result) and result.tzinfo is None and result == result.normalize()
                and result.isoformat() == value, "表格日期无效或不是规范午夜时间")
        return result
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("表格日期超出支持范围或格式无效") from error


def encode_table(frame, schema):
    require(isinstance(frame, pd.DataFrame) and frame.columns.tolist() == list(schema), "数据表字段或顺序无效")
    require(frame.index.equals(pd.RangeIndex(len(frame))) and frame.index.name is None, "数据表必须保留规范行索引")
    dtypes = {key: str(frame[key].dtype) for key in schema}
    result = {"columns": list(schema), "dtypes": dtypes, "rows": dataframe_records(frame)}
    # 解码路径同时检查类型和数值；避免把 True、分数份额等强制转换成有效值。
    decode_table(result, schema)
    return result


def decode_table(payload, schema):
    keys(payload, ("columns", "dtypes", "rows"))
    require(payload["columns"] == list(schema), "数据表字段或顺序无效")
    keys(payload["dtypes"], schema)
    require(isinstance(payload["rows"], list), "数据表记录必须是列表")
    for key, expected in schema.items():
        actual = payload["dtypes"][key]
        require(actual in ("object", "str", "string") if expected == "text" else actual == expected,
                "数据表类型与规范不一致")
    columns = {key: [] for key in schema}
    for row in payload["rows"]:
        keys(row, schema)
        for key, value in row.items():
            dtype = payload["dtypes"][key]
            if dtype == "datetime64[ns]":
                value = pd.NaT if value is None else _daily_timestamp(value)
            elif dtype == "float64":
                if value is not None:
                    number(value)
                value = np.nan if value is None else value
            elif dtype == "int64":
                integer(value)
                require(value <= np.iinfo(np.int64).max, "整数超过 int64 范围")
            elif dtype == "bool":
                require(type(value) is bool, "布尔字段必须为 true/false")
            else:
                require(value is None or isinstance(value, str), "文本列不接受其他类型")
            columns[key].append(value)
    try:
        return pd.DataFrame({key: pd.Series(values, dtype=payload["dtypes"][key]) for key, values in columns.items()})
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("数据表不能按保存的类型恢复") from error


def _encode_performance(performance):
    require(isinstance(performance, PortfolioPerformance), "必须保存完整组合绩效")
    series = {}
    for key, name in SERIES.items():
        item = getattr(performance, key)
        require(isinstance(item, pd.Series) and str(item.dtype) == "float64"
                and isinstance(item.index, pd.DatetimeIndex) and str(item.index.dtype) == "datetime64[ns]"
                and item.index.name == "date" and item.name == name, "绩效序列类型、名称或索引无效")
        series[key] = {"dates": [day.isoformat() for day in item.index], "values": item.tolist(),
                       "name": name, "index_name": "date", "dtype": "float64"}
    return {"series": series, "metrics": {key: getattr(performance, key) for key in METRICS},
            "unavailable_reasons": deepcopy(performance.unavailable_reasons)}


def _decode_performance(payload):
    keys(payload, ("series", "metrics", "unavailable_reasons"))
    keys(payload["series"], SERIES)
    keys(payload["metrics"], METRICS)
    series = {}
    for key, name in SERIES.items():
        item = payload["series"][key]
        keys(item, ("dates", "values", "name", "index_name", "dtype"))
        require(item["name"] == name and item["index_name"] == "date" and item["dtype"] == "float64",
                "绩效序列类型或名称无效")
        require(isinstance(item["dates"], list) and isinstance(item["values"], list)
                and len(item["dates"]) == len(item["values"]), "绩效序列日期与数值数量不一致")
        for value in item["values"]:
            number(value)
        series[key] = pd.Series(item["values"], name=name, dtype="float64",
                                index=pd.DatetimeIndex([_daily_timestamp(day) for day in item["dates"]],
                                                       name="date", dtype="datetime64[ns]"))
    reasons = payload["unavailable_reasons"]
    require(isinstance(reasons, dict), "未定义指标必须记录原因")
    require(set(reasons) == {key for key, value in payload["metrics"].items() if value is None},
            "未定义指标与原因清单不一致")
    for reason in reasons.values():
        text(reason)
    for value in payload["metrics"].values():
        if value is not None:
            number(value)
    require(payload["metrics"]["max_drawdown"] is not None, "最大回撤不能缺失")
    return PortfolioPerformance(**series, **payload["metrics"], unavailable_reasons=deepcopy(reasons))


def _metadata(meta, benchmark):
    keys(meta, {"schema_version", "status", "run_id", "run_at", "code_sha256", "environment", "strategy",
                "data", "warnings", "analytics_config"} | (set() if benchmark else {"factor_definition"}))
    require(meta["schema_version"] == ("portfolio-benchmark-v1" if benchmark else "portfolio-run-v1")
            and meta["status"] == "complete", "只能保存对应类别的完整成功结果")
    canonical_id(meta["run_id"])
    utc_time(meta["run_at"])
    digest(meta["code_sha256"])
    keys(meta["environment"], ("python", "pandas", "numpy", "pyarrow"))
    for value in meta["environment"].values():
        text(value)
    require(isinstance(meta["warnings"], list), "警告必须保留为列表")
    for value in meta["warnings"]:
        text(value)
    analytics = meta["analytics_config"]
    keys(analytics, ("periods_per_year", "annual_risk_free_rate"))
    integer(analytics["periods_per_year"], 1)
    number(analytics["annual_risk_free_rate"])
    require(analytics["annual_risk_free_rate"] > -1, "无风险年利率必须大于 -100%")
    AnalyticsConfig(**analytics)
    strategy = meta["strategy"]
    common = {"name", "version", "config", "signal_anchor_position", "execution_rule"}
    keys(strategy, common | ({"anchor_factor", "planned_execution_date", "target_universe", "target_weight"}
                            if benchmark else {"resolved_ascending", "lookback_periods", "signal_rule"}))
    require(strategy["name"] == ("equal_weight_buy_and_hold" if benchmark else "single_factor_top_n")
            and strategy["version"] == "1", "不支持的组合策略身份")
    integer(strategy["signal_anchor_position"], 1)
    config = strategy["config"]
    keys(config, ("backtest",) if benchmark else ("top_n", "rebalance_interval", "ascending", "backtest"))
    keys(config["backtest"], (field.name for field in fields(BacktestConfig)))
    PortfolioConfig(**({"top_n": 1} if benchmark else {key: config[key] for key in ("top_n", "rebalance_interval", "ascending")}),
                    backtest=BacktestConfig(**config["backtest"]))
    if benchmark:
        keys(strategy["anchor_factor"], ("name", "version"))
        for value in strategy["anchor_factor"].values():
            text(value)
        if strategy["planned_execution_date"] is not None:
            iso_date(strategy["planned_execution_date"])
        require(strategy["execution_rule"] == "L+1_open_once" and strategy["target_universe"] == "all_snapshot_symbols",
                "基准规则无效")
        number(strategy["target_weight"])
    else:
        require(type(strategy["resolved_ascending"]) is bool, "排序方向无效")
        keys(meta["factor_definition"], (field.name for field in fields(FactorDefinition)))
        definition = meta["factor_definition"]
        require(isinstance(definition["required_fields"], (tuple, list)), "因子字段清单无效")
        definition["required_fields"] = tuple(definition["required_fields"])
        FactorDefinition(**definition)
        require(strategy["lookback_periods"] == definition["lookback_periods"] == strategy["signal_anchor_position"]
                and strategy["signal_rule"] == "L+k*K" and strategy["execution_rule"] == "next_research_date_open",
                "因子窗口和执行规则不一致")
    data = meta["data"]
    keys(data, {"preview_sha256", "alignment_content_sha256", "market_sha256", "sources", "start", "end",
                "adjustment", "calendar", "row_count", "symbols"} | (set() if benchmark else {"factor_sha256"}))
    for key in ("preview_sha256", "alignment_content_sha256", "market_sha256") + (() if benchmark else ("factor_sha256",)):
        digest(data[key])
    iso_date(data["start"])
    iso_date(data["end"])
    require(data["start"] <= data["end"] and data["adjustment"] in ("none", "qfq", "hfq"), "研究区间或复权口径无效")
    symbols = data["symbols"]
    require(isinstance(symbols, list) and len(symbols) >= 2
            and all(isinstance(s, str) and re.fullmatch(r"[0-9]{6}", s) for s in symbols), "ETF 集合无效")
    require(symbols == sorted(set(symbols)), "ETF 集合须排序且不重复")
    integer(data["row_count"], 1)
    require(isinstance(data["sources"], list) and len(data["sources"]) == len(symbols), "来源清单不完整")
    for source, symbol in zip(data["sources"], symbols):
        keys(source, set(SOURCE_FIELDS) - {"processed_path"})
        for value in source.values():
            text(value)
        require(source["symbol"] == symbol and source["adjustment"] == data["adjustment"]
                and source["adjustment_basis"] in ("provider", "user_declared"), "来源代码或复权记录不一致")
        for key in ("file_sha256", "metadata_sha256", "input_content_sha256"):
            digest(source[key])
    calendar = data["calendar"]
    keys(calendar, ("dates", "mode", "source", "coverage_start", "coverage_end", "file_sha256", "input_dates_sha256"))
    require(calendar["mode"] in ("observed_union", "supplied"), "日期轴类型无效")
    text(calendar["source"])
    for key in ("coverage_start", "coverage_end"):
        iso_date(calendar[key])
    digest(calendar["input_dates_sha256"])
    if calendar["file_sha256"] is not None:
        digest(calendar["file_sha256"])
    dates = calendar["dates"]
    require(isinstance(dates, list) and bool(dates), "研究日期轴不能为空")
    for value in dates:
        iso_date(value)
    require(dates == sorted(set(dates)) and data["start"] <= dates[0] <= dates[-1] <= data["end"]
            and calendar["coverage_start"] <= dates[0] <= dates[-1] <= calendar["coverage_end"], "日期轴或覆盖区间不一致")
    require(data["row_count"] == len(dates) * len(symbols), "输入行数与研究网格不一致")
    if benchmark:
        anchor = strategy["signal_anchor_position"]
        planned = dates[anchor + 1] if anchor + 1 < len(dates) else None
        require(strategy["planned_execution_date"] == planned and strategy["target_weight"] == 1 / len(symbols), "基准计划不一致")
    else:
        require(config["top_n"] <= len(symbols), "选取数量超过 ETF 池")
    return pd.DatetimeIndex(dates, dtype="datetime64[ns]", name="date"), symbols


def _close(left, right, message):
    require(np.allclose(left, right, rtol=1e-12, atol=1e-9, equal_nan=False), message)


def _validate_analysis(analysis, benchmark):
    result, performance = analysis.backtest, analysis.performance
    dates, symbols = _metadata(result.metadata, benchmark)
    meta = result.metadata
    grid = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
    for name in ("input_data", "positions") + (() if benchmark else ("factor_data",)):
        table = getattr(result, name)
        require(pd.MultiIndex.from_frame(table[["date", "symbol"]]).equals(grid), "数据表必须保留完整日期 × ETF 网格")
    require(dataframe_sha256(result.input_data) == meta["data"]["market_sha256"], "行情快照指纹不一致")
    market = result.input_data
    price_columns = ["open", "high", "low", "close", "volume"]
    observed = market.has_observation
    require(market.loc[~observed, price_columns].isna().all().all(), "缺失行情行不能含替代价格")
    values = market.loc[observed, price_columns]
    require(np.isfinite(values.to_numpy()).all() and (values[["open", "high", "low", "close"]] > 0).all().all()
            and (values.volume >= 0).all(), "有效行情必须为有限正价格和非负成交量")
    if benchmark:
        require(result.factor_data is None and result.targets.factor_value.isna().all(), "基准不能携带伪造因子数据")
    else:
        factor = result.factor_data
        require(dataframe_sha256(factor) == meta["data"]["factor_sha256"], "因子快照指纹不一致")
        require(factor.status.notna().all() and factor.is_valid.equals(factor.status.eq("valid").astype(bool))
                and np.isfinite(factor.loc[factor.is_valid, "factor_value"]).all()
                and factor.loc[~factor.is_valid, "factor_value"].isna().all(), "因子有效标记与值不一致")
        require(result.targets.factor_value.notna().all(), "策略目标因子值不能缺失")
    # 允许缺失的地方仅限未到执行日、无关联成交、空仓缺行情及因子缺失。
    nullable = {"decisions": {"execution_date"}, "orders": {"trade_id"}, "positions": {"close"},
                "targets": {"factor_value"} if benchmark else set()}
    for name in ("decisions", "targets", "orders", "trades", "positions", "equity_curve"):
        table = getattr(result, name)
        for column in table:
            if column not in nullable.get(name, set()):
                require(table[column].notna().all(), "业务表含不允许的缺失值")
            if str(table[column].dtype) == "float64":
                require(not np.isinf(table[column].to_numpy()).any(), "业务表不能含无穷值")
            if str(table[column].dtype) == "datetime64[ns]":
                require(table[column].dropna().isin(dates).all(), "业务日期不在原日期轴内")
        if "symbol" in table:
            require(table.symbol.isin(symbols).all(), "业务表含研究池之外的 ETF")
    curve, positions = result.equity_curve, result.positions
    require(pd.DatetimeIndex(curve.date, name="date").equals(dates), "资金曲线必须保留原完整日期轴")
    initial = meta["strategy"]["config"]["backtest"]["initial_cash"]
    require((curve.equity > 0).all() and (curve.cash >= 0).all() and curve.equity.iloc[0] == initial,
            "资产必须有限正数、现金非负且从初始资金开始")
    _close(curve.equity, curve.cash + curve.position_value, "总资产与现金、持仓市值不符")
    _close(curve.net_value, curve.equity / initial, "资金曲线净值不一致")
    _close(curve.cumulative_return, curve.net_value - 1, "资金曲线累计收益不一致")
    held = positions.quantity > 0
    require(positions.loc[held, "close"].notna().all(), "正持仓不能缺估值价格")
    require((positions.quantity % meta["strategy"]["config"]["backtest"]["lot_size"] == 0).all(), "持仓不符合整手规则")
    _close(positions.position_value, positions.quantity.mul(positions.close).where(held, 0), "持仓市值不一致")
    _close(positions.groupby("date").position_value.sum(), curve.position_value, "逐日持仓市值汇总不一致")
    _close(positions.weight, positions.position_value / np.repeat(curve.equity.to_numpy(), len(symbols)), "持仓权重不一致")
    _ledgers(result)
    for key in SERIES:
        series = getattr(performance, key)
        require(series.index.equals(dates[1:] if key == "daily_returns" else dates), "绩效序列日期轴不一致")
    _close(performance.net_value, curve.net_value, "保存绩效净值与资产曲线不一致")
    require((performance.drawdown <= 0).all() and (performance.drawdown >= -1).all(), "回撤序列超出范围")
    require(-1 <= performance.max_drawdown <= 0, "最大回撤超出范围")


def _ledgers(result):
    """检查主外键和委托成交对应，不把这些检查描述为交易引擎重放认证。"""
    decisions, orders, trades, targets = result.decisions, result.orders, result.trades, result.targets
    for table, key in ((decisions, "decision_id"), (orders, "order_id"), (trades, "trade_id")):
        require(table[key].is_unique and table[key].str.len().gt(0).all(), "业务编号为空或重复")
    has_execution = decisions.execution_date.notna()
    require((decisions.loc[has_execution, "execution_date"] > decisions.loc[has_execution, "signal_date"]).all(),
            "计划执行日期必须晚于信号日期")
    require(not targets.duplicated(["decision_id", "symbol"]).any(), "目标记录重复")
    for table in (targets, orders, trades):
        require(table.decision_id.isin(decisions.decision_id).all(), "明细引用了不存在的调仓决策")
        if not table.empty:
            mapped = decisions.set_index("decision_id").signal_date
            require(table.signal_date.equals(table.decision_id.map(mapped)), "明细信号日期与决策不一致")
            if "execution_date" in table:
                executions = decisions.set_index("decision_id").execution_date
                require(table.execution_date.equals(table.decision_id.map(executions)), "明细执行日期与决策不一致")
    require((targets.target_weight > 0).all() and (targets.target_weight <= 1).all(), "目标权重无效")
    if not targets.empty:
        _close(targets.groupby("decision_id").target_weight.sum(), 1, "目标权重和不等于 1")
    require(orders.side.isin(["BUY", "SELL"]).all() and trades.side.isin(["BUY", "SELL"]).all(), "买卖方向无效")
    require((orders.requested_quantity > 0).all()
            and ((orders.filled_quantity + orders.unfilled_quantity) == orders.requested_quantity).all(), "委托份额对账失败")
    require(trades.order_id.is_unique and trades.order_id.isin(orders.order_id).all(), "成交引用了不存在或重复的委托")
    filled = orders.loc[orders.filled_quantity.gt(0)].set_index("order_id")
    require(set(filled.index) == set(trades.order_id), "成交与已成交委托数量不一致")
    for trade in trades.itertuples(index=False):
        order = filled.loc[trade.order_id]
        require(trade.quantity > 0 and trade.quantity == order.filled_quantity and trade.trade_id == order.trade_id
                and all(getattr(trade, key) == order[key] for key in ("symbol", "decision_id", "side", "signal_date", "execution_date")),
                "成交与对应委托不一致")
    require(orders.loc[orders.filled_quantity.eq(0), "trade_id"].isna().all(), "未成交委托不能引用成交记录")
    _accounting(result)


def _accounting(result):
    """只对保存的流水作账务核对；不生成信号、委托或新的绩效。"""
    trades, curve = result.trades, result.equity_curve
    require(trades.execution_date.is_monotonic_increasing, "成交必须按执行时间排序")
    require((trades[["open_price", "execution_price", "notional"]] > 0).all().all()
            and (trades[["commission", "tax", "slippage_cost", "cash_after"]] >= 0).all().all(),
            "成交价格、金额、成本或现金余额无效")
    _close(trades.notional, trades.quantity * trades.execution_price, "成交额与份额及价格不一致")
    _close(trades.slippage_cost, trades.quantity * (trades.execution_price - trades.open_price).abs(), "滑点金额不一致")
    cash = result.metadata["strategy"]["config"]["backtest"]["initial_cash"]
    holdings = {symbol: 0 for symbol in result.metadata["data"]["symbols"]}
    fees = slippage = 0.0
    by_date = {day: group for day, group in trades.groupby("execution_date", sort=False)}
    positions = result.positions.set_index(["date", "symbol"])
    for row in curve.itertuples(index=False):
        daily = by_date.get(row.date)
        if daily is not None:
            for trade in daily.itertuples(index=False):
                signed_quantity = trade.quantity if trade.side == "BUY" else -trade.quantity
                holdings[trade.symbol] += signed_quantity
                require(holdings[trade.symbol] >= 0 and holdings[trade.symbol] == trade.position_after, "成交后的份额对账失败")
                # v1 账本使用稳定求和：大额现金减去接近全仓的支出时，普通加减
                # 会损失小额余额。保留原买卖分组及 1e-9 微负现金归整约定。
                if trade.side == "BUY":
                    require(trade.tax == 0, "买入成交不能含卖出税")
                    cash = fsum((cash, -trade.notional, -trade.commission))
                    require(cash >= -1e-9, "买入后的现金不能为负")
                    cash = max(cash, 0.0)
                else:
                    proceeds = fsum((trade.notional, -trade.commission, -trade.tax))
                    require(proceeds >= 0, "卖出净收入不能为负")
                    cash = fsum((cash, proceeds))
                fees = fsum((fees, fsum((trade.commission, trade.tax))))
                slippage = fsum((slippage, trade.slippage_cost))
                _close(cash, trade.cash_after, "成交后的现金对账失败")
        _close(row.cash, cash, "每日现金与成交流水不一致")
        _close(row.cumulative_fees_taxes, fees, "累计费用与成交流水不一致")
        _close(row.cumulative_slippage, slippage, "累计滑点与成交流水不一致")
        require(all(positions.loc[(row.date, symbol), "quantity"] == quantity for symbol, quantity in holdings.items()),
                "每日持仓份额与成交流水不一致")


def encode_analysis(analysis, *, benchmark=False):
    require(isinstance(analysis, PortfolioAnalysis) and isinstance(analysis.backtest, PortfolioBacktestResult),
            "输入必须为完整 PortfolioAnalysis")
    result = analysis.backtest
    payload = {"metadata": deepcopy(result.metadata),
               "tables": {key: (None if benchmark and key == "factor_data" and result.factor_data is None
                                else encode_table(getattr(result, key), schema)) for key, schema in TABLES.items()},
               "performance": _encode_performance(analysis.performance)}
    decode_analysis(payload, benchmark=benchmark)
    return payload


def decode_analysis(payload, *, benchmark=False):
    keys(payload, ("metadata", "tables", "performance"))
    keys(payload["tables"], TABLES)
    tables = {key: (None if benchmark and key == "factor_data" and payload["tables"][key] is None
                    else decode_table(payload["tables"][key], schema)) for key, schema in TABLES.items()}
    result = PortfolioBacktestResult(**tables, metadata=deepcopy(payload["metadata"]))
    analysis = PortfolioAnalysis(backtest=result, performance=_decode_performance(payload["performance"]))
    _validate_analysis(analysis, benchmark)
    return analysis


def validate_pair(strategy, benchmark):
    left, right = strategy.backtest.metadata, benchmark.backtest.metadata
    for key in ("preview_sha256", "alignment_content_sha256", "market_sha256", "sources", "start", "end", "adjustment", "calendar", "symbols", "row_count"):
        require(left["data"][key] == right["data"][key], "策略与基准的数据身份或日期轴不同")
    require(left["strategy"]["config"]["backtest"] == right["strategy"]["config"]["backtest"]
            and left["analytics_config"] == right["analytics_config"]
            and left["strategy"]["signal_anchor_position"] == right["strategy"]["signal_anchor_position"],
            "策略与基准的成本、指标参数或热身锚点不同")
    require(right["strategy"]["anchor_factor"] == {key: left["factor_definition"][key] for key in ("name", "version")},
            "策略与基准的锚点因子不同")


def verify_current_performance(analysis):
    """只在保存当前代码运行结果时复核；历史解码永远不调用。"""
    from quantlab.portfolio.analysis import _performance_from_equity

    curve = analysis.backtest.equity_curve
    config = analysis.backtest.metadata["analytics_config"]
    equity = pd.Series(curve.equity.to_numpy(), index=pd.DatetimeIndex(curve.date, name="date"), name="equity")
    expected = _performance_from_equity(equity, periods=config["periods_per_year"], risk_free_rate=config["annual_risk_free_rate"])
    # 相同实现、相同输入必须一致；不为接近但不同的结果自动修补数值。
    require(_encode_performance(expected) == _encode_performance(analysis.performance), "绩效与记录的资产曲线或指标参数不一致")
