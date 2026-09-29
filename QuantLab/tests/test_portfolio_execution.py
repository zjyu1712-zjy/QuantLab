"""手算费用和预算；禁止抢占现金、余款再分配、容差融资。"""

from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from quantlab.config.settings import BacktestConfig
from quantlab.portfolio import PortfolioBacktestError


@pytest.fixture
def execution():
    # 缺失模块时逐项失败，避免在收集阶段掩盖测试。
    from quantlab.portfolio import execution
    return execution


def fees(**kwargs):
    return BacktestConfig(**{"commission_rate": 0, "minimum_commission": 0,
                             "slippage_rate": 0, **kwargs})


def test_minimum_fees_reduce_both_buys_without_code_priority(execution):
    config = fees(minimum_commission=5)
    actual = execution.allocate_buys(100000, {"510300": 5000, "159915": 5000},
                                     {"510300": 10, "159915": 10}, config)
    assert actual == {"159915": 4900, "510300": 4900}
    assert 100000 - sum(q * 10 + 5 for q in actual.values()) == 1990


@pytest.mark.parametrize("notional,expected", [(0, 0), (100, 5), (500, 5), (600, 6), (909, 9.09)])
def test_commission_switches_at_minimum_and_zero_is_free(execution, notional, expected):
    assert execution.commission(notional, fees(commission_rate=0.01, minimum_commission=5)) == pytest.approx(expected)


@pytest.mark.parametrize("side,open_price,expected", [("BUY", 10, 10.1), ("SELL", 12, 11.88)])
def test_slippage_adjusts_price_in_each_direction(execution, side, open_price, expected):
    assert execution.execution_price(open_price, side, fees(slippage_rate=0.01)) == pytest.approx(expected)


def test_no_rounding_to_cents(execution):
    assert execution.execution_price(1.234567, "BUY", fees(slippage_rate=0.001)) == pytest.approx(1.235801567)
    assert execution.commission(1.234567, fees(commission_rate=0.001)) == pytest.approx(0.001234567)


def test_affordability_uses_already_slipped_price_once(execution):
    config = fees(commission_rate=0.01, minimum_commission=2, slippage_rate=0.01, lot_size=10)
    assert execution.affordable_lot_quantity(1000, 10.1, 100, config) == 90
    assert execution.allocate_buys(1000, {"510300": 100}, {"510300": 10}, config) == {"510300": 90}


@pytest.mark.parametrize("budget,expected", [(0, 0), (104.99, 0), (105, 10),
                                            (505, 50), (605, 50), (606, 60), (5000, 100)])
def test_lot_budget_includes_piecewise_fee(execution, budget, expected):
    assert execution.affordable_lot_quantity(
        budget, 10, 100, fees(commission_rate=0.01, minimum_commission=5, lot_size=10)
    ) == expected


def test_max_quantity_is_a_ceiling_not_an_instruction_to_round_up(execution):
    assert execution.affordable_lot_quantity(10000, 10, 249, fees()) == 200
    assert execution.affordable_lot_quantity(10000, 10, 99, fees()) == 0


@pytest.mark.parametrize("budget,expected", [(1005, 100), (math.nextafter(1005, -math.inf), 0)])
def test_even_one_ulp_short_cannot_borrow(execution, budget, expected):
    config = fees(minimum_commission=5)
    assert execution.affordable_lot_quantity(budget, 10, 100, config) == expected
    assert execution.allocate_buys(budget, {"510300": 100}, {"510300": 10}, config) == {"510300": expected}


def test_relative_tolerance_cannot_finance_a_large_order(execution):
    assert execution.affordable_lot_quantity(1e12 - 0.001, 1e12, 1, fees(lot_size=1)) == 0


def test_unaffordable_minimum_fee_means_no_trade_and_no_fee(execution):
    config = fees(minimum_commission=2000)
    assert execution.affordable_lot_quantity(1000, 1, 100, config) == 0
    assert execution.allocate_buys(1000, {"510300": 100}, {"510300": 1}, config) == {"510300": 0}


def test_zero_quantity_has_no_minimum_commission(execution):
    config = fees(minimum_commission=5)
    assert execution.affordable_lot_quantity(1000, 10, 0, config) == 0
    # 没有买入申请的代码无需价格，未用价格不做额外校验。
    assert execution.allocate_buys(1000, {"510300": 0}, {}, config) == {"510300": 0}


def test_empty_requests_or_zero_cash_cannot_buy(execution):
    assert execution.allocate_buys(1000, {}, {}, fees()) == {}
    assert execution.allocate_buys(0, {"510300": 100}, {"510300": 10}, fees()) == {"510300": 0}


def test_sufficient_cash_fills_all_without_forced_full_investment(execution):
    assert execution.allocate_buys(5000, {"510300": 100, "159915": 200},
                                     {"510300": 10, "159915": 5}, fees(minimum_commission=5)) == {
                                         "159915": 200, "510300": 100}


def test_each_budget_is_based_on_cost_including_fees(execution):
    # 需求 1005/3005，现金 2015；预算约 505.006/1509.994，不能按无费金额分配。
    result = execution.allocate_buys(2015, {"159915": 1000, "510300": 1000},
                                      {"159915": 1, "510300": 3}, fees(minimum_commission=5))
    assert result == {"159915": 500, "510300": 500}
    assert 2015 - (500 * 1 + 5) - (500 * 3 + 5) == 5


def test_leftover_cash_is_not_redistributed(execution):
    result = execution.allocate_buys(3500, {"159915": 200, "510300": 100},
                                      {"159915": 10, "510300": 20}, fees(minimum_commission=5))
    assert result == {"159915": 100, "510300": 0}
    # 剩 2495 元，虽能买第二只的一手，也不二次分配。
    assert 3500 - (100 * 10 + 5) == 2495


def test_dictionary_order_cannot_change_budgets_or_mutate_inputs(execution):
    requests = {"510300": 1000, "159915": 1000, "588000": 0}
    opens = {"510300": 3, "159915": 1, "588000": None}
    before = deepcopy((requests, opens))
    config = fees(minimum_commission=5)
    left = execution.allocate_buys(2015, requests, opens, config)
    right = execution.allocate_buys(2015, dict(reversed(list(requests.items()))),
                                    dict(reversed(list(opens.items()))), config)
    assert left == right == {"159915": 500, "510300": 500, "588000": 0}
    assert list(left) == list(right) == ["159915", "510300", "588000"]
    left["159915"] = 0
    assert (requests, opens) == before


def test_buying_does_not_charge_sell_tax(execution):
    config = fees(sell_tax_rate=0.5)
    assert execution.allocate_buys(1000, {"510300": 100}, {"510300": 10}, config) == {"510300": 100}


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_positive_slippage_cannot_disappear_in_float_rounding(execution, side):
    # 1e20 * 1e-20 应代表非零滑点，不能因 1 +/- 滑点舍入回 1 而变成免费。
    with pytest.raises(PortfolioBacktestError) as caught:
        execution.execution_price(1e20, side, fees(slippage_rate=1e-20))
    assert caught.value.reason == "numeric_range"


def test_aggregate_rounding_cannot_finance_even_a_tiny_overrun(execution):
    # 每个独立预算分别舍入到可买一份，但三笔合计 13.92 已超过现金。
    with pytest.raises(PortfolioBacktestError) as caught:
        execution.allocate_buys(13.919999999999998,
                               {"159915": 1, "510300": 1, "588000": 1},
                               {"159915": 9.23, "510300": 0.13, "588000": 4.56}, fees(lot_size=1))
    assert caught.value.reason == "numeric_range"


def test_numpy_numbers_are_supported_without_quantity_overflow(execution):
    assert execution.allocate_buys(np.float64(1000), {"510300": np.int64(100)},
                                     {"510300": np.float64(10)}, fees(lot_size=np.int64(100))) == {"510300": 100}


def test_large_allocation_avoids_intermediate_multiplication_overflow(execution):
    result = execution.allocate_buys(1e300, {"159915": 1, "510300": 1},
                                      {"159915": 6e299, "510300": 6e299}, fees(lot_size=1))
    assert result == {"159915": 0, "510300": 0}


def test_huge_request_uses_bounded_search(execution):
    # 子进程超时保护：误改为逐手递减不能卡住整套测试。
    result = subprocess.run(
        [sys.executable, "-c", "from quantlab.portfolio.execution import affordable_lot_quantity; "
         "from quantlab.config.settings import BacktestConfig; "
         "print(affordable_lot_quantity(10000, 1, 10**18, "
         "BacktestConfig(commission_rate=0, slippage_rate=0)))"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "10000"


@pytest.mark.parametrize("bad", [-1, math.nan, math.inf, -math.inf, True, "100", None, 10**1000])
@pytest.mark.parametrize("field", ["notional", "budget", "cash", "price", "open_price"])
def test_invalid_money_and_prices_fail_with_structured_error(execution, bad, field):
    with pytest.raises(PortfolioBacktestError):
        if field == "notional":
            execution.commission(bad, fees())
        elif field == "budget":
            execution.affordable_lot_quantity(bad, 10, 100, fees())
        elif field == "cash":
            execution.allocate_buys(bad, {}, {}, fees())
        elif field == "price":
            execution.affordable_lot_quantity(1000, bad, 100, fees())
        else:
            execution.execution_price(bad, "BUY", fees())


@pytest.mark.parametrize("call", ["price", "open_price", "allocation_price"])
def test_zero_price_cannot_create_free_shares(execution, call):
    with pytest.raises(PortfolioBacktestError):
        if call == "price":
            execution.affordable_lot_quantity(1000, 0, 100, fees())
        elif call == "open_price":
            execution.execution_price(0, "BUY", fees())
        else:
            execution.allocate_buys(1000, {"510300": 100}, {"510300": 0}, fees())


@pytest.mark.parametrize("bad", [-1, 1.5, True, "100", None, math.inf])
def test_invalid_quantities_fail_without_silent_conversion(execution, bad):
    with pytest.raises(PortfolioBacktestError):
        execution.affordable_lot_quantity(1000, 10, bad, fees())
    with pytest.raises(PortfolioBacktestError):
        execution.allocate_buys(1000, {"510300": bad}, {"510300": 10}, fees())


def test_buy_requests_must_already_be_whole_lots(execution):
    with pytest.raises(PortfolioBacktestError):
        execution.allocate_buys(1000, {"510300": 99}, {"510300": 10}, fees())


@pytest.mark.parametrize("side", ["buy", "HOLD", "", 1, None, []])
def test_unknown_side_is_not_silently_treated_as_sell(execution, side):
    with pytest.raises(PortfolioBacktestError):
        execution.execution_price(10, side, fees())


@pytest.mark.parametrize("requests,opens", [([], {}), ({}, []), ({510300: 100}, {510300: 10}),
                                            ({"bad": 100}, {"bad": 10}), ({"510300": 100}, {})])
def test_invalid_mapping_or_missing_required_open_is_clear(execution, requests, opens):
    with pytest.raises(PortfolioBacktestError):
        execution.allocate_buys(1000, requests, opens, fees())


@pytest.mark.parametrize("field,bad", [("commission_rate", math.nan), ("minimum_commission", math.inf),
                                      ("slippage_rate", 1), ("lot_size", True), ("lot_size", 1.5)])
def test_each_entry_revalidates_config(execution, field, bad):
    config = fees()
    object.__setattr__(config, field, bad)
    for call in (lambda: execution.commission(0, config),
                 lambda: execution.execution_price(10, "BUY", config),
                 lambda: execution.affordable_lot_quantity(0, 10, 0, config),
                 lambda: execution.allocate_buys(0, {}, {}, config)):
        with pytest.raises(PortfolioBacktestError) as caught:
            call()
        assert caught.value.reason == "invalid_config"


def test_wrong_config_type_is_rejected(execution):
    with pytest.raises(PortfolioBacktestError):
        execution.allocate_buys(1000, {}, {}, None)


@pytest.mark.parametrize("case", ["price", "fee", "notional", "cost", "sum", "zero_underflow", "erased_fee", "quantity_precision"])
def test_numeric_range_failures_do_not_return_nan_or_partial_allocations(execution, case):
    with pytest.raises(PortfolioBacktestError) as caught:
        if case == "price":
            execution.execution_price(1.7e308, "BUY", fees(slippage_rate=0.9))
        elif case == "fee":
            execution.commission(1e308, fees(commission_rate=2))
        elif case == "notional":
            execution.affordable_lot_quantity(1e308, 1e308, 100, fees())
        elif case == "cost":
            execution.affordable_lot_quantity(1e308, 1e308, 1, fees(minimum_commission=1e308, lot_size=1))
        elif case == "sum":
            execution.allocate_buys(1e308, {"159915": 1, "510300": 1},
                                    {"159915": 1e308, "510300": 1e308}, fees(lot_size=1))
        elif case == "zero_underflow":
            execution.execution_price(math.ulp(0.0), "SELL", fees(slippage_rate=0.9))
        elif case == "erased_fee":
            execution.affordable_lot_quantity(1e20, 1e20, 1, fees(minimum_commission=5, lot_size=1))
        else:
            execution.affordable_lot_quantity(float(2**53), 1, 2**53 + 1, fees(lot_size=1))
    assert caught.value.reason == "numeric_range"
    assert caught.value.date is None
    assert isinstance(caught.value.symbols, tuple)
