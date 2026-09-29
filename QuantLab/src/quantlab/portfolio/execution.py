"""纯费用与买入预算；不读行情、不改变现金或持仓、不实际提交订单。"""

import math
from numbers import Integral, Real
import re

from quantlab.config.settings import BacktestConfig
from quantlab.portfolio.types import PortfolioBacktestError, PortfolioConfig


def _invalid(message: str) -> None:
    raise PortfolioBacktestError(message, reason="invalid_execution_input")


def _range_error(message: str) -> None:
    raise PortfolioBacktestError(message, reason="numeric_range")


def _validate_config(config: BacktestConfig) -> None:
    # 复用组合边界的严格检查，不修改第一阶段配置类的行为。
    PortfolioConfig(top_n=1, backtest=config)


def _money(value: float, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        _invalid(f"{label}必须为有限数值，不能为布尔值或文本")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        _invalid(f"{label}超出数值范围")
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        _invalid(f"{label}必须为{'正' if positive else '非负'}有限数值")
    return result


def _quantity(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        _invalid("申请份额必须为非负整数，不能为布尔值")
    return int(value)


def _finite(value: float, label: str, *, positive: bool = False) -> float:
    if not math.isfinite(value) or (positive and value <= 0):
        _range_error(f"{label}超出可可靠表示的数值范围")
    return value


def _sum(values, label: str) -> float:
    try:
        result = math.fsum(values)
    except OverflowError:
        _range_error(f"{label}求和溢出")
    return _finite(result, label)


def _commission(notional: float, config: BacktestConfig) -> float:
    if notional == 0:
        return 0.0
    rate = float(config.commission_rate)
    proportional = _finite(notional * rate, "比例佣金", positive=rate > 0)
    return max(proportional, float(config.minimum_commission))


def commission(notional: float, config: BacktestConfig) -> float:
    """实际成交额的佣金；零成交额不收最低佣金，不包括卖出税。"""
    _validate_config(config)
    return _commission(_money(notional, "成交额"), config)


def execution_price(open_price: float, side: str, config: BacktestConfig) -> float:
    """买入上调、卖出下调开盘价；滑点已在价格中，不再重复扣现金。"""
    _validate_config(config)
    price = _money(open_price, "开盘价", positive=True)
    if not isinstance(side, str) or side not in ("BUY", "SELL"):
        _invalid("买卖方向只能为 BUY 或 SELL")
    slip = float(config.slippage_rate)
    result = _finite(price * (1 + slip if side == "BUY" else 1 - slip), "含滑点价", positive=True)
    if slip > 0 and result == price:
        _range_error("价格精度不足，无法保留非零滑点")
    return result


def _buy_cost(quantity: int, price: float, config: BacktestConfig) -> float:
    if quantity == 0:
        return 0.0
    try:
        notional = quantity * price
    except OverflowError:
        _range_error("买入成交额溢出")
    notional = _finite(notional, "买入成交额", positive=True)
    fee = _commission(notional, config)
    cost = _sum((notional, fee), "含费买入成本")
    if (fee > 0 and cost <= notional) or cost <= fee:
        _range_error("金额精度不足，无法保留成交额及正佣金")
    return cost


def _filled_cost(quantity: int, price: float, config: BacktestConfig) -> float:
    # 不把大整数转换损失的一份当作已支付份额；只核对最终成交数量，
    # 巨大但未获资金支持的上限仍可通过二分迅速缩小。
    try:
        exact_quantity = int(float(quantity)) == quantity
    except (OverflowError, ValueError):
        exact_quantity = False
    if not exact_quantity:
        _range_error("成交份额无法在金额运算中精确表示")
    return _buy_cost(quantity, price, config)


def affordable_lot_quantity(budget: float, price: float, max_quantity: int,
                            config: BacktestConfig) -> int:
    """在独立含费预算内寻找最大整手数量；price 已含滑点，不再调整。"""
    _validate_config(config)
    available = _money(budget, "买入预算")
    price = _money(price, "成交价", positive=True)
    ceiling = _quantity(max_quantity)
    lot = int(config.lot_size)
    low, high = 0, ceiling // lot
    if high == 0 or available == 0:
        return 0
    full_cost = _buy_cost(high * lot, price, config)
    if full_cost <= available:
        low = high
    else:
        # 在整数手数上二分，不从最大申请逐手扣减；预算比较无容差融资。
        while low < high:
            middle = (low + high + 1) // 2
            if _buy_cost(middle * lot, price, config) <= available:
                low = middle
            else:
                high = middle - 1
    quantity = low * lot
    _filled_cost(quantity, price, config)
    return quantity


def allocate_buys(cash: float, requests: dict[str, int], opens: dict[str, float],
                  config: BacktestConfig) -> dict[str, int]:
    """同时按完整含费需求分预算，余款留现金；返回包括零成交的申请代码。"""
    _validate_config(config)
    cash = _money(cash, "可用现金")
    if not isinstance(requests, dict) or not isinstance(opens, dict):
        _invalid("买入申请与开盘价必须为按 ETF 代码索引的字典")
    if any(not isinstance(code, str) or not re.fullmatch(r"[0-9]{6}", code) for code in requests):
        _invalid("买入申请的 ETF 代码必须为六位字符串")
    quantities, prices, needs = {}, {}, {}
    for code in sorted(requests):
        quantity = _quantity(requests[code])
        if quantity % int(config.lot_size):
            _invalid("买入申请必须已按整手取整")
        quantities[code] = quantity
        if quantity == 0:
            continue
        if code not in opens:
            raise PortfolioBacktestError("买入申请缺少开盘价", symbols=(code,), reason="invalid_execution_input")
        prices[code] = execution_price(opens[code], "BUY", config)
        needs[code] = _buy_cost(quantity, prices[code], config)
    total_need = _sum(needs.values(), "全部买入需求")
    if not needs:
        return quantities

    # 先确定每笔预算，再计算成交。除法先行避免 cash * need 的中间溢出。
    budgets = {code: need if total_need <= cash else cash * (need / total_need)
               for code, need in needs.items()}
    result = {code: 0 for code in quantities}
    for code, budget in budgets.items():
        result[code] = affordable_lot_quantity(budget, prices[code], quantities[code], config)
    spent = _sum((_filled_cost(result[code], prices[code], config) for code in needs), "实际买入支出")
    if spent > cash:
        _range_error("独立预算的数值舍入使总支出超过现金，不能融资或偏向某只 ETF")
    return result
