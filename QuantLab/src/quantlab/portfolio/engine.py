"""逐日模拟先卖后买与收盘估值；仅内部入口，不持久化或提供不完整结果。"""

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

from quantlab.data.alignment import ALIGNED_COLUMNS, validate_aligned_data
from quantlab.data.calendar import CalendarAxis
from quantlab.portfolio.execution import allocate_buys, commission, execution_price
from quantlab.portfolio.types import (
    DECISION_DTYPES, TARGET_DTYPES, ORDER_DTYPES, TRADE_DTYPES, POSITION_DTYPES, EQUITY_DTYPES,
    PortfolioBacktestError, PortfolioBacktestResult, PortfolioConfig, RebalanceSchedule,
)


MONEY_ATOL = 1e-9  # 只归整运算后的微小负余额，不增加买入预算。
MAX_QUANTITY = np.iinfo(np.int64).max


def _fail(message, reason, day=None, codes=()):
    context = f"{day.isoformat()} " if day is not None else ""
    context += f"ETF {', '.join(codes)} " if codes else ""
    raise PortfolioBacktestError(context + message, date=day, symbols=tuple(codes), reason=reason)


def _finite(value, label, day, codes=(), *, positive=False):
    if not math.isfinite(value) or (positive and value <= 0):
        _fail(f"{label}超出可可靠表示的数值范围", "numeric_range", day, codes)
    return value


def _sum(values, label, day, codes=()):
    try:
        result = math.fsum(values)
    except OverflowError:
        _fail(f"{label}求和溢出", "numeric_range", day, codes)
    return _finite(result, label, day, codes)


def _cash(value, day):
    _finite(value, "现金", day)
    if value < -MONEY_ATOL:
        _fail("现金出现负余额，不能融资", "negative_cash", day)
    return 0.0 if value < 0 else value


def _check_quantity(quantity, lot, day, code):
    if quantity < 0 or quantity > MAX_QUANTITY or quantity % lot:
        _fail("份额须为非负整手且在 int64 范围内", "numeric_range", day, (code,))


def _checked_market(data, calendar):
    if not isinstance(data, pd.DataFrame) or tuple(data.columns) != ALIGNED_COLUMNS:
        _fail("组合行情必须为规范的八列对齐表", "invalid_market_data")
    frame = data.copy(deep=True).reset_index(drop=True)
    if frame.date.dtype != np.dtype("datetime64[ns]") or any(not isinstance(s, str) for s in frame.symbol):
        _fail("组合行情的日期或代码类型无效", "invalid_market_data")
    symbols = tuple(sorted(set(frame.symbol)))
    try:
        validate_aligned_data(frame, calendar, symbols)
    except (ValueError, TypeError, OverflowError) as error:
        _fail(f"组合行情校验失败：{error}", "invalid_market_data")
    return frame, symbols


def _schedule_table(table, schema):
    if not isinstance(table, pd.DataFrame) or table.columns.tolist() != list(schema):
        _fail("内部调仓安排缺少规范字段", "invalid_schedule")
    result = table.copy(deep=True).reset_index(drop=True)
    for name, dtype in schema.items():
        correct = (isinstance(result[name].dtype, pd.StringDtype) if dtype == "string"
                   else result[name].dtype == np.dtype(dtype))
        if not correct:
            _fail(f"调仓安排字段 {name} 类型无效", "invalid_schedule")
        if name != "execution_date" and result[name].isna().any():
            _fail(f"调仓安排字段 {name} 不能为空", "invalid_schedule")
    return result


def _checked_schedule(schedule, calendar, symbols, config):
    if not isinstance(schedule, RebalanceSchedule) or type(schedule.ascending) is not bool:
        _fail("内部调仓安排或排序方向无效", "invalid_schedule")
    decisions = _schedule_table(schedule.decisions, DECISION_DTYPES)
    targets = _schedule_table(schedule.targets, TARGET_DTYPES)
    dates = pd.DatetimeIndex(calendar.dates, dtype="datetime64[ns]")
    if (decisions.decision_id.tolist() != [f"PD{i + 1:06d}" for i in range(len(decisions))]
            or not decisions.signal_date.is_monotonic_increasing or decisions.signal_date.duplicated().any()
            or not decisions.signal_date.isin(dates).all()
            or not decisions.top_n.eq(config.top_n).all()
            or not decisions.valid_count.between(0, len(symbols)).all()
            or not decisions.reason.str.len().gt(0).all()):
        _fail("决策日期、身份、有效数量或 N 不符合研究安排", "invalid_schedule")
    if (not targets.decision_id.isin(decisions.decision_id).all()
            or not targets.symbol.isin(symbols).all()
            or targets.duplicated(["decision_id", "symbol"]).any()
            or not np.isfinite(targets[["factor_value", "target_weight"]].to_numpy()).all()):
        _fail("目标代码、身份或数值无效，不能重复持有目标席位", "invalid_schedule")
    for row in decisions.itertuples():
        index = dates.get_loc(row.signal_date)
        has_next = index + 1 < len(dates)
        if ((has_next and (pd.isna(row.execution_date) or row.execution_date != dates[index + 1]))
                or (not has_next and not pd.isna(row.execution_date))):
            _fail("计划执行日必须为信号日紧邻的下一研究日期", "invalid_schedule")
        selected = targets.loc[targets.decision_id.eq(row.decision_id)]
        enough = row.valid_count >= config.top_n
        status = "SKIPPED_INSUFFICIENT_FACTORS" if not enough else ("PENDING" if has_next else "EXPIRED")
        if row.status != status or (row.cutoff_tie and row.valid_count <= config.top_n):
            _fail("决策状态或边界并列标记与有效数量/执行日期不一致", "invalid_schedule")
        if enough:
            if (len(selected) != config.top_n or not selected.signal_date.eq(row.signal_date).all()
                    or not selected.target_weight.eq(1.0 / config.top_n).all()):
                _fail("每次有效调仓须有恰好 N 个同日等权目标", "invalid_schedule")
        elif not selected.empty:
            _fail("跳过调仓不能含有目标配置", "invalid_schedule")
    return decisions, targets.sort_values(["signal_date", "symbol"]).reset_index(drop=True)


@dataclass
class _DayState:
    """当日暂存账本；完成收盘估值前，不接入已接受的每日结果。"""

    cash: float
    holdings: dict[str, int]
    fees: float
    slippage: float
    orders: list = field(default_factory=list)
    trades: list = field(default_factory=list)


def _fill_order(state, row, code, side, requested, filled, market, config, day, order_offset, trade_offset):
    order_id = f"PO{order_offset + len(state.orders) + 1:06d}"
    trade_id = pd.NA
    if filled:
        if market["volume"] == 0:
            _fail("实际成交所需的当日成交量为零，不支持本次成交假设", "zero_volume", day, (code,))
        price = execution_price(market["open"], side, config.backtest)
        _check_quantity(filled, int(config.backtest.lot_size), day, code)
        if int(float(filled)) != filled:
            _fail("成交份额无法精确参与金额运算", "numeric_range", day, (code,))
        notional = _finite(filled * price, "成交额", day, (code,), positive=True)
        fee = commission(notional, config.backtest)
        tax_rate = float(config.backtest.sell_tax_rate) if side == "SELL" else 0.0
        tax = _finite(notional * tax_rate, "卖出税", day, (code,), positive=tax_rate > 0)
        costs = _sum((fee, tax), "费用税额", day, (code,))
        slip = _finite(filled * abs(price - market["open"]), "滑点金额", day, (code,))
        if side == "SELL":
            proceeds = _sum((notional, -fee, -tax), "卖出净收入", day, (code,))
            if proceeds < 0:
                _fail("卖出费用超过成交额，净收入为负", "negative_sell_proceeds", day, (code,))
            if costs > 0 and proceeds >= notional:
                _fail("卖出金额精度不足，无法保留正佣金或税费", "numeric_range", day, (code,))
            state.cash = _sum((state.cash, proceeds), "卖出后现金", day, (code,))
            state.holdings[code] -= filled
        else:
            state.cash = _cash(_sum((state.cash, -notional, -fee), "买入后现金", day, (code,)), day)
            state.holdings[code] += filled
        _check_quantity(state.holdings[code], int(config.backtest.lot_size), day, code)
        state.fees = _sum((state.fees, costs), "累计佣金税费", day, (code,))
        state.slippage = _sum((state.slippage, slip), "累计滑点", day, (code,))
        trade_id = f"PT{trade_offset + len(state.trades) + 1:06d}"
        state.trades.append({
            "trade_id": trade_id, "order_id": order_id, "decision_id": row.decision_id, "symbol": code,
            "signal_date": row.signal_date, "execution_date": pd.Timestamp(day), "side": side,
            "quantity": filled, "open_price": market["open"], "execution_price": price, "notional": notional,
            "commission": fee, "tax": tax, "slippage_cost": slip, "cash_after": state.cash,
            "position_after": state.holdings[code],
        })
    status, reason = ("FILLED", "按申请份额完成模拟成交") if filled == requested else (
        ("PARTIALLY_FILLED", "含费独立预算不足，仅成交部分整手份额") if filled else
        ("REJECTED", "含费独立预算不足一手，未成交也未收取费用"))
    state.orders.append({
        "order_id": order_id, "decision_id": row.decision_id, "symbol": code,
        "signal_date": row.signal_date, "execution_date": pd.Timestamp(day), "side": side,
        "requested_quantity": requested, "filled_quantity": filled, "unfilled_quantity": requested - filled,
        "status": status, "reason": reason, "trade_id": trade_id,
    })


def _rebalance(state, row, selected, market, config, day, order_offset, trade_offset):
    needed = tuple(sorted(set(selected) | {s for s, q in state.holdings.items() if q > 0}))
    missing = tuple(s for s in needed if not market[s]["has_observation"])
    if missing:
        _fail("调仓所需开盘价缺失，整批交易不能执行", "missing_open", day, missing)
    try:
        opening_equity = _sum(
            [state.cash] + [_finite(q * market[s]["open"], "开盘持仓市值", day, (s,))
                            for s, q in state.holdings.items() if q], "开盘总资产", day)
        lot = int(config.backtest.lot_size)
        targets = dict.fromkeys(state.holdings, 0)
        for code in selected:
            lot_value = _finite(market[code]["open"] * lot, "每手开盘金额", day, (code,), positive=True)
            target_lots = _finite((opening_equity / config.top_n) / lot_value, "目标手数", day, (code,))
            targets[code] = math.floor(target_lots) * lot
            _check_quantity(targets[code], lot, day, code)
        # 目标只按交易前资产计算一次，不随扣费重新追求精确等权。
        for code in sorted(targets):
            difference = state.holdings[code] - targets[code]
            if difference > 0:
                _fill_order(state, row, code, "SELL", difference, difference, market[code],
                            config, day, order_offset, trade_offset)
        requests = {s: targets[s] - state.holdings[s] for s in sorted(targets) if targets[s] > state.holdings[s]}
        filled = allocate_buys(state.cash, requests, {s: market[s]["open"] for s in requests}, config.backtest)
        for code in requests:
            _fill_order(state, row, code, "BUY", requests[code], filled[code], market[code],
                        config, day, order_offset, trade_offset)
    except PortfolioBacktestError as error:
        if error.date is not None:
            raise
        _fail(str(error), error.reason, day, error.symbols or needed)
    except OverflowError:
        _fail("组合交易数值运算溢出", "numeric_range", day, needed)


def _value_day(state, market, symbols, config, day):
    missing = tuple(s for s in symbols if state.holdings[s] and not market[s]["has_observation"])
    if missing:
        _fail("正持仓的收盘价缺失，无法完成当日估值", "missing_close", day, missing)
    positions = []
    for code in symbols:
        quantity = state.holdings[code]
        _check_quantity(quantity, int(config.backtest.lot_size), day, code)
        close = market[code]["close"]
        # 空仓缺失保留 NaN；不要计算 0 * NaN。
        value = _finite(quantity * close, "收盘持仓市值", day, (code,), positive=True) if quantity else 0.0
        positions.append({"date": pd.Timestamp(day), "symbol": code, "quantity": quantity,
                          "close": close, "position_value": value})
    position_value = _sum((r["position_value"] for r in positions), "收盘总市值", day)
    state.cash = _cash(state.cash, day)
    equity = _sum((state.cash, position_value), "收盘总资产", day)
    if equity <= 0:
        _fail("总资产已耗尽，无法定义持仓权重或继续完整绩效研究", "non_positive_equity", day)
    for position in positions:
        position["weight"] = position["position_value"] / equity
    weights = _sum([r["weight"] for r in positions] + [state.cash / equity], "资产权重", day)
    if not math.isclose(weights, 1.0, rel_tol=0, abs_tol=1e-12):
        _fail("持仓与现金权重无法对账", "numeric_range", day)
    net_value = _finite(equity / float(config.backtest.initial_cash), "净值", day, positive=True)
    curve = {"date": pd.Timestamp(day), "cash": state.cash, "position_value": position_value,
             "equity": equity, "net_value": net_value, "cumulative_return": net_value - 1,
             "cumulative_fees_taxes": state.fees, "cumulative_slippage": state.slippage}
    return positions, curve


def _table(rows, schema):
    return pd.DataFrame(rows, columns=list(schema)).astype(schema)


def simulate_portfolio(data: pd.DataFrame, calendar: CalendarAxis, schedule: RebalanceSchedule,
                       *, config: PortfolioConfig) -> PortfolioBacktestResult:
    """返回完整期间内部账本；失败仅抛结构化错误，不返回截断曲线或修改输入。"""
    if not isinstance(config, PortfolioConfig):
        _fail("组合配置必须为 PortfolioConfig", "invalid_config")
    # 冻结容器内部仍可能被程序篡改；在运行边界重验，不依赖早前构造成功。
    PortfolioConfig(config.top_n, config.rebalance_interval, config.ascending, config.backtest)
    frame, symbols = _checked_market(data, calendar)
    if config.top_n > len(symbols):
        _fail("N 不能超过行情资产数量", "invalid_config")
    decisions, targets = _checked_schedule(schedule, calendar, symbols, config)
    pending = {row.execution_date: row for row in decisions.itertuples() if row.status == "PENDING"}
    markets = {day: group.set_index("symbol").to_dict("index") for day, group in frame.groupby("date", sort=True)}
    cash, holdings, fees, slippage = float(config.backtest.initial_cash), dict.fromkeys(symbols, 0), 0.0, 0.0
    orders, trades, positions, curves = [], [], [], []
    for day in calendar.dates:
        stamp = pd.Timestamp(day)
        state = _DayState(cash, holdings.copy(), fees, slippage)
        row = pending.get(stamp)
        if row is not None:
            selected = targets.loc[targets.decision_id.eq(row.decision_id), "symbol"].tolist()
            _rebalance(state, row, selected, markets[stamp], config, day, len(orders), len(trades))
        day_positions, curve = _value_day(state, markets[stamp], symbols, config, day)
        # 仅在整天交易/估值/不变量均通过后接受状态与结果行。
        cash, holdings, fees, slippage = state.cash, state.holdings, state.fees, state.slippage
        orders.extend(state.orders)
        trades.extend(state.trades)
        positions.extend(day_positions)
        curves.append(curve)
        if row is not None:
            decisions.loc[row.Index, "status"] = "EXECUTED"
            decisions.loc[row.Index, "reason"] += "；执行流程完成，实际成交情况见订单和成交表"
    return PortfolioBacktestResult(
        decisions=decisions, targets=targets, orders=_table(orders, ORDER_DTYPES), trades=_table(trades, TRADE_DTYPES),
        positions=_table(positions, POSITION_DTYPES), equity_curve=_table(curves, EQUITY_DTYPES),
    )
