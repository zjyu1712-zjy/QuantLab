"""用合成行情和独立手算账本验证组合引擎；手动安排仅用于内部单测。"""

from copy import deepcopy
from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd
import pytest

from quantlab.config.settings import BacktestConfig
from quantlab.data.alignment import align_market_data
from quantlab.data.calendar import parse_calendar_csv
from quantlab.factors import run_research_factor
from quantlab.portfolio import PortfolioBacktestError, PortfolioConfig
from quantlab.portfolio.selection import DECISION_DTYPES, TARGET_DTYPES, build_rebalance_schedule
from quantlab.portfolio.types import RebalanceSchedule


A, B, C = "510300", "159915", "512000"


def simulate(case):
    from quantlab.portfolio.engine import simulate_portfolio
    data, axis, schedule, config = case
    return simulate_portfolio(data, axis, schedule, config=config)


def arrange(axis, signals, top_n):
    decisions, targets = [], []
    for number, (index, selected) in enumerate(signals, 1):
        identity = f"PD{number:06d}"
        day = pd.Timestamp(axis.dates[index])
        next_day = pd.Timestamp(axis.dates[index + 1]) if index + 1 < len(axis.dates) else pd.NaT
        status = "SKIPPED_INSUFFICIENT_FACTORS" if not selected else (
            "EXPIRED" if pd.isna(next_day) else "PENDING")
        decisions.append((identity, day, next_day, len(selected), top_n, status, "内部手算安排", False))
        for code in selected:
            targets.append((identity, day, code, 0.1, 1 / top_n))
    return RebalanceSchedule(
        pd.DataFrame(decisions, columns=list(DECISION_DTYPES)).astype(DECISION_DTYPES),
        pd.DataFrame(targets, columns=list(TARGET_DTYPES)).astype(TARGET_DTYPES), False,
    )


def make_case(*, prices=None, signals=((0, (A, B)), (1, (B, C))), top_n=2,
              backtest=None, gaps=None, zeros=None, supplied=False):
    prices = prices or {A: [(10, 10), (10, 11), (12, 12)],
                        B: [(20, 20), (20, 18), (20, 20)], C: [(5, 5)] * 3}
    dates = pd.date_range("2025-01-02", periods=len(next(iter(prices.values()))), freq="B")
    datasets = {}
    for code, values in prices.items():
        datasets[code] = pd.DataFrame([
            (day, code, opening, max(opening, close), min(opening, close), close,
             0 if index in (zeros or {}).get(code, []) else 1000)
            for index, (day, (opening, close)) in enumerate(zip(dates, values))
            if index not in (gaps or {}).get(code, [])
        ], columns=["date", "symbol", "open", "high", "low", "close", "volume"])
    calendar = None
    if supplied:
        calendar = parse_calendar_csv(
            ("date\n" + "\n".join(day.date().isoformat() for day in dates)).encode(),
            source="引擎手算日期轴", coverage_start=dates[0].date(), coverage_end=dates[-1].date())
    aligned = align_market_data(datasets, start=dates[0].date(), end=dates[-1].date(), calendar=calendar)
    config = PortfolioConfig(top_n=top_n, rebalance_interval=1, backtest=backtest or BacktestConfig(
        initial_cash=1000, lot_size=10, commission_rate=0, slippage_rate=0))
    return aligned.data, aligned.calendar, arrange(aligned.calendar, signals, top_n), config


@pytest.fixture
def manual_case():
    return make_case()


def test_cash_and_holdings_match_manual_ledger(manual_case):
    result = simulate(manual_case)
    assert result.equity_curve.cash.tolist() == [1000.0, 100.0, 150.0]
    assert result.equity_curve.equity.tolist() == [1000.0, 1010.0, 1100.0]
    assert result.equity_curve.position_value.tolist() == [0.0, 910.0, 950.0]
    assert result.equity_curve.net_value.tolist() == pytest.approx([1, 1.01, 1.1])
    assert result.equity_curve.cumulative_return.tolist() == pytest.approx([0, 0.01, 0.1])
    last = result.positions.loc[result.positions.date.eq(pd.Timestamp(manual_case[1].dates[2]))]
    assert dict(zip(last.symbol, last.quantity)) == {A: 0, B: 20, C: 110}
    assert result.trades.side.tolist() == ["BUY", "BUY", "SELL", "BUY"]
    assert result.trades.symbol.tolist() == [B, A, A, C]
    assert result.trades.quantity.tolist() == [20, 50, 50, 110]
    assert result.trades.cash_after.tolist() == [600.0, 100.0, 700.0, 150.0]
    assert result.trades.position_after.tolist() == [20, 50, 0, 110]
    assert result.decisions.status.tolist() == ["EXECUTED", "EXECUTED"]
    assert result.trades.trade_id.tolist() == [f"PT{i:06d}" for i in range(1, 5)]
    assert result.orders.order_id.tolist() == [f"PO{i:06d}" for i in range(1, 5)]
    assert result.orders.status.eq("FILLED").all()
    assert result.orders.unfilled_quantity.eq(0).all()
    assert result.orders.trade_id.tolist() == result.trades.trade_id.tolist()


def test_costs_are_not_double_counted():
    config = BacktestConfig(initial_cash=1000, lot_size=10, commission_rate=0.01,
                            minimum_commission=2, slippage_rate=0.01, sell_tax_rate=0.005)
    case = make_case(prices={A: [(10, 10), (10, 10), (12, 12)], B: [(1000, 1000)] * 3},
                     signals=((0, (A,)), (1, (B,))), top_n=1, backtest=config)
    result = simulate(case)
    assert result.trades.quantity.tolist() == [90, 90]
    assert result.trades.execution_price.tolist() == pytest.approx([10.1, 11.88])
    assert result.trades.notional.tolist() == pytest.approx([909, 1069.2])
    assert result.trades.commission.tolist() == pytest.approx([9.09, 10.692])
    assert result.trades.tax.tolist() == pytest.approx([0, 5.346])
    assert result.trades.slippage_cost.tolist() == pytest.approx([9, 10.8])
    assert result.equity_curve.cash.tolist() == pytest.approx([1000, 81.91, 1135.072])
    assert result.equity_curve.equity.tolist() == pytest.approx([1000, 981.91, 1135.072])
    assert result.equity_curve.cumulative_fees_taxes.tolist() == pytest.approx([0, 9.09, 25.128])
    assert result.equity_curve.cumulative_slippage.tolist() == pytest.approx([0, 9, 19.8])


def test_retained_assets_trade_only_differences_and_sell_before_lower_code_buy():
    case = make_case(prices={A: [(10, 10), (10, 10), (20, 20)], B: [(20, 20)] * 3},
                     signals=((0, (A, B)), (1, (A, B))))
    result = simulate(case)
    last = result.trades.loc[result.trades.execution_date.eq(pd.Timestamp(case[1].dates[2]))]
    assert last.symbol.tolist() == [A, B]  # 卖 510300 必须早于买 159915，不能全局按代码重排。
    assert last.side.tolist() == ["SELL", "BUY"]
    assert last.quantity.tolist() == [20, 10]
    assert last.position_after.tolist() == [30, 30]
    assert last.cash_after.tolist() == [500, 300]
    assert result.equity_curve.equity.tolist() == [1000, 1000, 1500]


@pytest.mark.parametrize("cost_kind", ["commission", "tax"])
def test_positive_sell_cost_cannot_be_swallowed_by_large_notional(cost_kind):
    case = make_case(prices={A: [(10, 10), (10, 10), (1e18, 1e18)], B: [(1e21, 1e21)] * 3},
                     signals=((0, (A,)), (1, (B,))), top_n=1,
                     backtest=BacktestConfig(initial_cash=1000, lot_size=1, commission_rate=0, slippage_rate=0,
                                            minimum_commission=5 if cost_kind == "commission" else 0,
                                            sell_tax_rate=1e-20 if cost_kind == "tax" else 0))
    with pytest.raises(PortfolioBacktestError) as caught:
        simulate(case)
    assert caught.value.reason == "numeric_range"
    assert caught.value.date == case[1].dates[2]
    assert caught.value.symbols == (A,)


def test_partial_and_rejected_orders_keep_requests_and_reasons():
    case = make_case(prices={A: [(10, 10)] * 3, B: [(50, 50)] * 3},
                     signals=((0, (A, B)),),
                     backtest=BacktestConfig(initial_cash=1000, lot_size=10,
                                            commission_rate=0, minimum_commission=5, slippage_rate=0))
    result = simulate(case)
    assert result.orders.symbol.tolist() == [B, A]
    assert result.orders.requested_quantity.tolist() == [10, 50]
    assert result.orders.filled_quantity.tolist() == [0, 40]
    assert result.orders.unfilled_quantity.tolist() == [10, 10]
    assert result.orders.status.tolist() == ["REJECTED", "PARTIALLY_FILLED"]
    assert result.orders.reason.str.len().gt(0).all()
    assert pd.isna(result.orders.trade_id.iloc[0])
    assert result.orders.trade_id.iloc[1] == "PT000001"
    assert result.trades.symbol.tolist() == [A]
    assert result.trades.commission.tolist() == [5.0]
    assert result.equity_curve.cash.tolist() == [1000, 595, 595]
    assert result.equity_curve.equity.tolist() == [1000, 995, 995]


@pytest.mark.parametrize("mode", ["unchanged", "rejected"])
def test_zero_volume_without_fill_is_allowed(mode):
    case = make_case(prices={A: [(10, 10)] * 3, B: [(50, 50)] * 3},
                     signals=((0, (A, B)), (1, (A, B))),
                     zeros={A: [2], B: [1, 2]} if mode == "rejected" else {A: [2], B: [2]},
                     backtest=BacktestConfig(initial_cash=1000, lot_size=10,
                                            commission_rate=0, minimum_commission=5, slippage_rate=0))
    result = simulate(case)
    assert len(result.orders) == 2 and len(result.trades) == 1
    assert result.decisions.status.eq("EXECUTED").all()
    assert result.equity_curve.cumulative_fees_taxes.tolist() == [0, 5, 5]


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_actual_fill_with_zero_volume_fails(side):
    index = 1 if side == "BUY" else 2
    case = make_case(signals=((0, (A,)), (1, (B,))), top_n=1, zeros={A: [index]})
    with pytest.raises(PortfolioBacktestError) as caught:
        simulate(case)
    assert caught.value.reason == "zero_volume"
    assert caught.value.date == case[1].dates[index]
    assert caught.value.symbols == (A,)


def test_skipped_rebalance_still_values_existing_positions():
    case = make_case(signals=((0, (A,)), (1, ())), top_n=1, gaps={A: [2]})
    with pytest.raises(PortfolioBacktestError) as caught:
        simulate(case)
    assert caught.value.reason == "missing_close"
    assert caught.value.date == case[1].dates[2] and caught.value.symbols == (A,)
    assert "收盘" in str(caught.value)


def test_skipped_rebalance_does_not_liquidate_or_require_unheld_prices():
    case = make_case(signals=((0, (A,)), (1, ())), top_n=1, gaps={B: [2]})
    result = simulate(case)
    assert result.trades.side.tolist() == ["BUY"]
    assert result.equity_curve.cash.tolist() == [1000, 0, 0]
    assert result.equity_curve.equity.tolist() == [1000, 1100, 1200]
    assert result.decisions.status.tolist() == ["EXECUTED", "SKIPPED_INSUFFICIENT_FACTORS"]


def test_required_open_is_checked_before_any_batch_trade():
    case = make_case(signals=((0, (A,)), (1, (B,))), top_n=1, gaps={B: [2]})
    before = deepcopy(case)
    with pytest.raises(PortfolioBacktestError) as caught:
        simulate(case)
    assert caught.value.reason == "missing_open"
    assert caught.value.date == case[1].dates[2] and caught.value.symbols == (B,)
    assert "开盘" in str(caught.value)
    pd.testing.assert_frame_equal(case[0], before[0])
    pd.testing.assert_frame_equal(case[2].decisions, before[2].decisions)
    assert not hasattr(caught.value, "result")


def test_all_required_open_gaps_are_reported_together():
    case = make_case(signals=((0, (A,)), (1, (B,))), top_n=1, gaps={A: [2], B: [2]})
    with pytest.raises(PortfolioBacktestError) as caught:
        simulate(case)
    assert caught.value.reason == "missing_open"
    assert caught.value.symbols == (B, A)


def test_expired_signal_never_executes():
    case = make_case(signals=((2, (A, B)),))
    result = simulate(case)
    assert result.trades.empty and result.orders.empty
    assert result.decisions.status.tolist() == ["EXPIRED"]
    assert len(result.targets) == 2
    assert result.equity_curve.equity.tolist() == [1000, 1000, 1000]


def test_sold_asset_later_gap_is_allowed():
    case = make_case(prices={A: [(10, 10)] * 4, B: [(20, 20)] * 4}, top_n=1,
                     signals=((0, (A,)), (1, (B,))), gaps={A: [3]})
    result = simulate(case)
    assert result.equity_curve.equity.tolist() == [1000] * 4
    missing = result.positions.loc[result.positions.date.eq(pd.Timestamp(case[1].dates[3])) & result.positions.symbol.eq(A)].iloc[0]
    assert pd.isna(missing.close) and missing.quantity == 0 and missing.position_value == 0 and missing.weight == 0


def test_same_target_no_orders_or_fees():
    case = make_case(prices={A: [(12, 12)] * 3, B: [(12, 12)] * 3},
                     signals=((0, (A, B)), (1, (A, B))),
                     backtest=BacktestConfig(initial_cash=1000, lot_size=10,
                                            commission_rate=0, minimum_commission=5, slippage_rate=0))
    result = simulate(case)
    assert result.trades.quantity.tolist() == [40, 40]
    assert result.equity_curve.cash.tolist() == [1000, 30, 30]
    assert result.equity_curve.equity.tolist() == [1000, 990, 990]
    assert result.equity_curve.cumulative_fees_taxes.tolist() == [0, 10, 10]


def test_cash_only_missing_axis_day_is_valid():
    case = make_case(signals=(), supplied=True, gaps={A: [1], B: [1], C: [1]})
    result = simulate(case)
    assert result.equity_curve.equity.tolist() == [1000, 1000, 1000]
    assert result.positions.quantity.eq(0).all() and result.positions.weight.eq(0).all()
    assert result.positions.close.isna().sum() == 3


def test_partial_ohlcv_row_rejected(manual_case):
    frame = manual_case[0].copy()
    frame.loc[0, "close"] = np.nan
    with pytest.raises(PortfolioBacktestError) as caught:
        simulate((frame, *manual_case[1:]))
    assert caught.value.reason == "invalid_market_data"


def test_extreme_sell_fee_fails():
    case = make_case(prices={A: [(10, 10), (10, 10), (0.001, 0.001)], B: [(1000, 1000)] * 3},
                     signals=((0, (A,)), (1, (B,))), top_n=1,
                     backtest=BacktestConfig(initial_cash=1000, lot_size=10,
                                            commission_rate=0, minimum_commission=5, slippage_rate=0))
    with pytest.raises(PortfolioBacktestError) as caught:
        simulate(case)
    assert caught.value.reason == "negative_sell_proceeds"
    assert caught.value.date == case[1].dates[2] and caught.value.symbols == (A,)


def test_total_asset_exhaustion_is_explicit_not_fake_weights():
    # 采用二进制可精确表示的价格：买 800 份支出 800+100，卖出收入 100-100。
    case = make_case(prices={A: [(1, 1), (1, 1), (0.125, 0.125)], B: [(1000, 1000)] * 3},
                     signals=((0, (A,)), (1, (B,))), top_n=1,
                     backtest=BacktestConfig(initial_cash=900, lot_size=100,
                                            commission_rate=0, minimum_commission=100, slippage_rate=0))
    with pytest.raises(PortfolioBacktestError) as caught:
        simulate(case)
    assert caught.value.reason == "non_positive_equity"
    assert caught.value.date == case[1].dates[2]


@pytest.mark.parametrize("mode", ["quantity", "position_value", "tax"])
def test_overflow_fails_without_mutation(mode):
    case = make_case(signals=((0, (A,)), (1, (B,))), top_n=1)
    frame, axis, schedule, config = deepcopy(case)
    if mode == "quantity":
        config = replace(config, backtest=replace(config.backtest, initial_cash=1e30))
    elif mode == "position_value":
        frame.loc[frame.date.eq(pd.Timestamp(axis.dates[2])) & frame.symbol.eq(A), ["open", "high", "low", "close"]] = 1e308
    else:
        config = replace(config, backtest=replace(config.backtest, sell_tax_rate=1e308))
    before = frame.copy(deep=True)
    with pytest.raises(PortfolioBacktestError) as caught:
        simulate((frame, axis, schedule, config))
    assert caught.value.reason == "numeric_range"
    assert caught.value.date in axis.dates
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("index_kind", ["duplicate", "multi"])
def test_input_labels_and_return_tables_are_isolated(manual_case, index_kind):
    baseline = simulate(manual_case)
    frame, axis, schedule, config = deepcopy(manual_case)
    frame.index = (pd.Index([0] * len(frame), name="date") if index_kind == "duplicate" else
                   pd.MultiIndex.from_arrays([[0] * len(frame), range(len(frame))], names=["date", "symbol"]))
    schedule.decisions.index = [0] * len(schedule.decisions)
    schedule.targets.index = [0] * len(schedule.targets)
    before = deepcopy((frame, schedule))
    actual = simulate((frame, axis, schedule, config))
    for name in ("decisions", "targets", "orders", "trades", "positions", "equity_curve"):
        pd.testing.assert_frame_equal(getattr(actual, name), getattr(baseline, name))
    actual.targets.loc[0, "target_weight"] = 0
    actual.decisions.loc[0, "status"] = "MODIFIED"
    pd.testing.assert_frame_equal(frame, before[0])
    pd.testing.assert_frame_equal(schedule.targets, before[1].targets)
    pd.testing.assert_frame_equal(schedule.decisions, before[1].decisions)
    assert actual.metadata is None and actual.input_data is None and actual.factor_data is None


def test_daily_accounting_invariants_and_canonical_types(manual_case):
    result = simulate(manual_case)
    assert list(zip(result.positions.date.dt.date, result.positions.symbol)) == list(product(manual_case[1].dates, sorted([A, B, C])))
    for row in result.equity_curve.itertuples():
        positions = result.positions.loc[result.positions.date.eq(row.date)]
        assert row.equity == pytest.approx(row.cash + positions.position_value.sum())
        assert positions.weight.sum() + row.cash / row.equity == pytest.approx(1)
        assert positions.quantity.ge(0).all() and positions.quantity.mod(10).eq(0).all()
    for name in ("orders", "trades", "positions", "equity_curve"):
        table = getattr(result, name)
        assert table.index.equals(pd.RangeIndex(len(table)))
        for column in table:
            if column == "date" or column.endswith("_date"):
                assert table[column].dtype == np.dtype("datetime64[ns]")
            elif column in ("quantity", "requested_quantity", "filled_quantity", "unfilled_quantity", "position_after"):
                assert table[column].dtype == np.dtype("int64")
    assert result.trades.execution_date.gt(result.trades.signal_date).all()


def test_empty_orders_and_trades_still_have_stable_columns_and_types():
    result = simulate(make_case(signals=()))
    assert result.orders.columns.tolist() == ["order_id", "decision_id", "symbol", "signal_date", "execution_date", "side",
                                           "requested_quantity", "filled_quantity", "unfilled_quantity", "status", "reason", "trade_id"]
    assert result.trades.columns.tolist() == ["trade_id", "order_id", "decision_id", "symbol", "signal_date", "execution_date",
                                           "side", "quantity", "open_price", "execution_price", "notional", "commission", "tax",
                                           "slippage_cost", "cash_after", "position_after"]
    assert isinstance(result.orders.trade_id.dtype, pd.StringDtype)
    assert result.trades.quantity.dtype == np.dtype("int64")
    assert result.trades.commission.dtype == np.dtype("float64")


def test_real_factor_schedule_only_trades_after_warmup(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=45)
    factor = run_research_factor(preview, "momentum_20")
    config = PortfolioConfig(top_n=1)
    schedule = build_rebalance_schedule(factor, config=config)
    result = simulate((preview.alignment.data, preview.alignment.calendar, schedule, config))
    dates = pd.to_datetime(factor.calendar.dates)
    assert result.trades.execution_date.isin([dates[21], dates[41]]).all()
    assert result.trades.execution_date.min() == dates[21]
    assert result.equity_curve.equity.iloc[:21].eq(100000).all()
    assert result.positions.loc[result.positions.date.eq(dates[-1]), "quantity"].sum() > 0


@pytest.mark.parametrize("damage", ["duplicate_target", "missing_target", "wrong_weight", "unknown_code", "unknown_decision",
    "target_signal", "nonfinite_factor", "duplicate_decision", "same_day", "skip_day", "outside", "wrong_n", "bad_count",
    "unknown_status", "executed_status", "false_expiry", "skip_with_targets", "bad_ascending", "wrong_schema", "string_date"])
def test_invalid_schedule_never_silently_trades(manual_case, damage):
    frame, axis, schedule, config = deepcopy(manual_case)
    decisions, targets = schedule.decisions, schedule.targets
    if damage == "duplicate_target":
        targets = pd.concat([targets, targets.iloc[[0]]], ignore_index=True)
    elif damage == "missing_target":
        targets = targets.iloc[1:].copy()
    elif damage == "wrong_weight":
        targets.loc[0, "target_weight"] = 0.6
    elif damage == "unknown_code":
        targets.loc[0, "symbol"] = "588000"
    elif damage == "unknown_decision":
        targets.loc[0, "decision_id"] = "PD999999"
    elif damage == "target_signal":
        targets.loc[0, "signal_date"] += pd.Timedelta(days=1)
    elif damage == "nonfinite_factor":
        targets.loc[0, "factor_value"] = np.inf
    elif damage == "duplicate_decision":
        decisions = pd.concat([decisions, decisions.iloc[[0]]], ignore_index=True)
    elif damage == "same_day":
        decisions.loc[0, "execution_date"] = decisions.signal_date.iloc[0]
    elif damage == "skip_day":
        decisions.loc[0, "execution_date"] = pd.Timestamp(axis.dates[2])
    elif damage == "outside":
        decisions.loc[0, "signal_date"] -= pd.Timedelta(days=20)
    elif damage == "wrong_n":
        decisions.loc[0, "top_n"] = 1
    elif damage == "bad_count":
        decisions.loc[0, "valid_count"] = 4
    elif damage in ("unknown_status", "executed_status", "false_expiry", "skip_with_targets"):
        decisions.loc[0, "status"] = {"unknown_status": "UNKNOWN", "executed_status": "EXECUTED",
                                     "false_expiry": "EXPIRED", "skip_with_targets": "SKIPPED_INSUFFICIENT_FACTORS"}[damage]
    elif damage == "bad_ascending":
        schedule = replace(schedule, ascending="False")
    elif damage == "wrong_schema":
        targets = targets.drop(columns="factor_value")
    else:
        decisions["signal_date"] = decisions.signal_date.astype(str)
    schedule = replace(schedule, decisions=decisions, targets=targets)
    with pytest.raises(PortfolioBacktestError) as caught:
        simulate((frame, axis, schedule, config))
    assert caught.value.reason == "invalid_schedule"


@pytest.mark.parametrize("damage", ["drop_row", "duplicate_row", "reverse_rows", "wrong_config", "wrong_schedule", "wrong_axis"])
def test_invalid_inputs_fail_with_portfolio_error(manual_case, damage):
    frame, axis, schedule, config = deepcopy(manual_case)
    if damage == "drop_row":
        frame = frame.iloc[:-1]
    elif damage == "duplicate_row":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif damage == "reverse_rows":
        frame = frame.iloc[::-1]
    elif damage == "wrong_config":
        config = None
    elif damage == "wrong_schedule":
        schedule = None
    else:
        axis = None
    with pytest.raises(PortfolioBacktestError):
        simulate((frame, axis, schedule, config))
