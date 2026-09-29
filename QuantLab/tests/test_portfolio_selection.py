"""调仓日与目标筛选的手算契约；不是成交或组合收益测试。"""

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from quantlab.config.settings import BacktestConfig
from quantlab.factors import run_research_factor
from quantlab.ui.factors import top_n_factor_preview


def config(**kwargs):
    # 延迟导入使缺少功能表现为各项测试失败，而非整个文件无法收集。
    from quantlab.portfolio import PortfolioConfig
    return PortfolioConfig(**kwargs)


def schedule(result, **kwargs):
    from quantlab.portfolio.selection import build_rebalance_schedule
    return build_rebalance_schedule(result, config=config(**kwargs))


@pytest.fixture
def factor_result(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=45)
    return run_research_factor(preview)


def value_grid(base, values):
    """内部筛选单测专用原始值网格，不冒充经公式验证的公开回测输入。"""
    symbols = sorted(values)
    rows = []
    for index, day in enumerate(base.calendar.dates):
        for symbol in symbols:
            value = values[symbol]
            state = "warmup" if index < base.definition.lookback_periods else (
                "missing_window" if value is None else "valid"
            )
            rows.append((pd.Timestamp(day), symbol, value if state == "valid" else np.nan,
                         state == "valid", state))
    frame = pd.DataFrame(rows, columns=["date", "symbol", "factor_value", "is_valid", "status"])
    frame = frame.astype({"date": "datetime64[ns]", "symbol": "string",
                          "factor_value": "float64", "is_valid": "bool", "status": "string"})
    sources = tuple({**deepcopy(base.sources[0]), "symbol": symbol} for symbol in symbols)
    return replace(base, data=frame, sources=sources)


def test_schedule_is_anchored_after_warmup(factor_result):
    result = schedule(factor_result, top_n=1)
    dates = pd.to_datetime(factor_result.calendar.dates)
    assert result.decisions.signal_date.tolist() == [dates[20], dates[40]]
    assert result.decisions.execution_date.tolist() == [dates[21], dates[41]]
    assert result.decisions.status.tolist() == ["PENDING", "PENDING"]
    assert result.targets.symbol.tolist() == ["510300", "510300"]
    assert result.targets.target_weight.tolist() == [1.0, 1.0]
    assert result.decisions.decision_id.tolist() == ["PD000001", "PD000002"]


def test_skip_does_not_move_anchor(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=45, gaps={"159915": [5]})
    factor = run_research_factor(preview)
    result = schedule(factor, top_n=2)
    assert result.decisions.status.tolist() == ["SKIPPED_INSUFFICIENT_FACTORS", "PENDING"]
    assert result.decisions.valid_count.tolist() == [1, 2]
    assert result.decisions.signal_date.tolist() == [pd.Timestamp(factor.calendar.dates[i]) for i in (20, 40)]
    assert result.targets.decision_id.tolist() == ["PD000002", "PD000002"]
    assert result.targets.target_weight.tolist() == [0.5, 0.5]
    assert result.decisions.reason.iloc[0]


@pytest.mark.parametrize("factor_name,lookback", [
    ("momentum_20", 20), ("momentum_60", 60), ("momentum_120", 120),
    ("volatility_20", 20), ("volatility_60", 60),
])
@pytest.mark.parametrize("interval", [1, 20])
def test_each_factor_and_interval_uses_axis_positions(make_portfolio_preview, factor_name, lookback, interval):
    preview, _ = make_portfolio_preview(periods=lookback + 3)
    factor = run_research_factor(preview, factor_name)
    actual = schedule(factor, top_n=1, rebalance_interval=interval)
    positions = [lookback, lookback + 1, lookback + 2] if interval == 1 else [lookback]
    dates = pd.to_datetime(factor.calendar.dates)
    assert actual.decisions.signal_date.tolist() == [dates[i] for i in positions]
    for row, index in zip(actual.decisions.itertuples(), positions):
        if index == lookback + 2:
            assert pd.isna(row.execution_date) and row.status == "EXPIRED"
        else:
            assert row.execution_date == dates[index + 1] and row.status == "PENDING"


@pytest.mark.parametrize("missing,status,count", [(False, "EXPIRED", 2), (True, "SKIPPED_INSUFFICIENT_FACTORS", 0)])
def test_final_day_expiry_has_lower_priority_than_insufficient_factors(make_portfolio_preview, missing, status, count):
    preview, _ = make_portfolio_preview(periods=21, gaps={"159915": [5]} if missing else None)
    factor = run_research_factor(preview)
    actual = schedule(factor, top_n=2)
    assert len(actual.decisions) == 1
    assert actual.decisions.status.iloc[0] == status
    assert pd.isna(actual.decisions.execution_date.iloc[0])
    assert len(actual.targets) == count


@pytest.mark.parametrize("periods", [1, 10, 20])
def test_short_history_keeps_typed_empty_tables(make_portfolio_preview, periods):
    preview, _ = make_portfolio_preview(periods=periods)
    actual = schedule(run_research_factor(preview), top_n=2)
    assert actual.decisions.empty and actual.targets.empty
    assert actual.decisions.columns.tolist() == ["decision_id", "signal_date", "execution_date",
                                               "valid_count", "top_n", "status", "reason", "cutoff_tie"]
    assert actual.targets.columns.tolist() == ["decision_id", "signal_date", "symbol", "factor_value", "target_weight"]
    assert actual.decisions.signal_date.dtype == np.dtype("datetime64[ns]")
    assert actual.decisions.valid_count.dtype == np.dtype("int64")
    assert actual.decisions.cutoff_tie.dtype == np.dtype("bool")
    assert actual.targets.target_weight.dtype == np.dtype("float64")
    assert isinstance(actual.targets.symbol.dtype, pd.StringDtype)


@pytest.mark.parametrize("ascending,n,codes,tied", [
    (False, 1, ["159915"], True),
    (False, 2, ["159915", "510300"], False),
    (False, 3, ["159915", "510300", "512000"], False),
    (True, 1, ["512000"], False),
    (True, 2, ["512000", "159915"], True),
])
def test_exact_seats_ties_zero_negative_and_missing(factor_result, ascending, n, codes, tied):
    factor = value_grid(factor_result, {"510300": 0.0, "159915": 0.0, "512000": -0.1, "588000": None})
    actual = schedule(factor, top_n=n, ascending=ascending)
    assert actual.ascending is ascending
    assert actual.decisions.valid_count.tolist() == [3, 3]
    assert actual.decisions.cutoff_tie.tolist() == [tied, tied]
    for decision in actual.decisions.itertuples():
        selected = actual.targets.loc[actual.targets.decision_id.eq(decision.decision_id)]
        assert selected.symbol.tolist() == codes
        assert selected.target_weight.tolist() == [1.0 / n] * n
        preview = top_n_factor_preview(factor, decision.signal_date.date(), n, ascending=ascending)
        assert selected.symbol.tolist() == preview.table["ETF 代码"].tolist()
        assert selected.target_weight.tolist() == preview.table["目标权重"].tolist()


def test_raw_values_not_display_rounding_determine_seats(factor_result):
    factor = value_grid(factor_result, {"510300": 0.100002, "159915": 0.100001})
    actual = schedule(factor, top_n=1)
    assert actual.targets.symbol.tolist() == ["510300", "510300"]
    assert actual.targets.factor_value.tolist() == [0.100002, 0.100002]
    assert not actual.decisions.cutoff_tie.any()


@pytest.mark.parametrize("direction,ascending,code", [
    ("higher_is_stronger", False, "510300"),
    ("higher_is_more_volatile", True, "159915"),
])
def test_default_direction_is_explicit(factor_result, direction, ascending, code):
    factor = value_grid(factor_result, {"510300": 0.2, "159915": 0.1})
    factor = replace(factor, definition=replace(factor.definition, direction=direction))
    actual = schedule(factor, top_n=1)
    assert actual.ascending is ascending
    assert actual.targets.symbol.tolist() == [code, code]


def test_all_invalid_does_not_generate_a_liquidation_target(factor_result):
    factor = value_grid(factor_result, {"510300": None, "159915": None})
    actual = schedule(factor, top_n=1)
    assert actual.targets.empty
    assert actual.decisions.status.eq("SKIPPED_INSUFFICIENT_FACTORS").all()
    assert actual.decisions.valid_count.eq(0).all()


def test_full_declared_axis_is_not_compressed_to_complete_dates(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=45, supplied=True,
                                        gaps={"510300": [5], "159915": [5]})
    factor = run_research_factor(preview)
    actual = schedule(factor, top_n=1)
    dates = pd.to_datetime(factor.calendar.dates)
    assert actual.decisions.signal_date.tolist() == [dates[20], dates[40]]
    assert actual.decisions.status.tolist() == ["SKIPPED_INSUFFICIENT_FACTORS", "PENDING"]
    assert actual.targets.signal_date.tolist() == [dates[40]]


def test_non_signal_day_does_not_change_scheduled_choices(factor_result):
    original = schedule(factor_result, top_n=1)
    changed = deepcopy(factor_result)
    day = pd.Timestamp(changed.calendar.dates[21])
    changed.data.loc[changed.data.date.eq(day) & changed.data.symbol.eq("159915"), "factor_value"] = 99.0
    actual = schedule(changed, top_n=1)
    pd.testing.assert_frame_equal(actual.decisions, original.decisions)
    pd.testing.assert_frame_equal(actual.targets, original.targets)


@pytest.mark.parametrize("index_kind", ["duplicate", "multi"])
def test_selection_does_not_mutate_or_depend_on_index_labels(factor_result, index_kind):
    original = schedule(factor_result, top_n=1)
    factor = deepcopy(factor_result)
    size = len(factor.data)
    factor.data.index = (pd.Index([0] * size, name="date") if index_kind == "duplicate" else
                         pd.MultiIndex.from_arrays([[0] * size, range(size)], names=["date", "symbol"]))
    before = deepcopy(factor)
    actual = schedule(factor, top_n=1)
    pd.testing.assert_frame_equal(actual.decisions, original.decisions)
    pd.testing.assert_frame_equal(actual.targets, original.targets)
    actual.targets.loc[0, "factor_value"] = 999.0
    pd.testing.assert_frame_equal(factor.data, before.data)
    assert factor.sources == before.sources


def test_declared_range_can_be_wider_than_observed_axis(factor_result):
    factor = replace(factor_result, start=factor_result.start - timedelta(days=10),
                     end=factor_result.end + timedelta(days=10))
    assert len(schedule(factor, top_n=1).decisions) == 2


@pytest.mark.parametrize("field", ["top_n", "rebalance_interval"])
@pytest.mark.parametrize("bad", [0, -1, True, False, 1.5, "2", None])
def test_config_rejects_non_positive_integer_controls(field, bad):
    kwargs = {"top_n": 1, field: bad}
    with pytest.raises(ValueError):
        config(**kwargs)


@pytest.mark.parametrize("bad", [0, 1, "ascending", [], np.bool_(True)])
def test_config_rejects_non_bool_sort_direction(bad):
    with pytest.raises(ValueError):
        config(top_n=1, ascending=bad)


@pytest.mark.parametrize("field", ["initial_cash", "commission_rate", "minimum_commission", "slippage_rate", "sell_tax_rate"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -1.0, True, "0"])
def test_config_revalidates_nested_financial_values(field, bad):
    backtest = BacktestConfig()
    # 模拟程序直接损坏配置，不让旧构造器的校验代替新边界测试。
    object.__setattr__(backtest, field, bad)
    with pytest.raises(ValueError):
        config(top_n=1, backtest=backtest)


@pytest.mark.parametrize("field,bad", [("initial_cash", 0), ("slippage_rate", 1),
                                      ("lot_size", 0), ("lot_size", -1), ("lot_size", True),
                                      ("lot_size", 1.5), ("lot_size", "100")])
def test_config_rejects_other_unsafe_execution_values(field, bad):
    backtest = BacktestConfig()
    object.__setattr__(backtest, field, bad)
    with pytest.raises(ValueError):
        config(top_n=1, backtest=backtest)


@pytest.mark.parametrize("bad", [None, {}, 100])
def test_config_rejects_wrong_backtest_type(bad):
    with pytest.raises(ValueError):
        config(top_n=1, backtest=bad)


def test_n_cannot_exceed_asset_universe_and_error_is_structured(factor_result):
    with pytest.raises(ValueError) as caught:
        schedule(factor_result, top_n=3)
    assert caught.value.reason == "invalid_config"
    assert caught.value.date is None
    assert caught.value.symbols == ()


def test_numpy_integer_controls_are_integers(factor_result):
    actual = schedule(factor_result, top_n=np.int64(1), rebalance_interval=np.int64(20))
    assert actual.targets.symbol.tolist() == ["510300", "510300"]


@pytest.mark.parametrize("field,value", [("direction", "unknown"), ("availability", "before_open")])
@pytest.mark.parametrize("ascending", [None, True])
def test_unknown_direction_or_availability_cannot_silently_trade(factor_result, field, value, ascending):
    factor = replace(factor_result, definition=replace(factor_result.definition, **{field: value}))
    with pytest.raises(ValueError):
        schedule(factor, top_n=1, ascending=ascending)


@pytest.mark.parametrize("damage", [
    "drop_row", "duplicate_row", "reorder", "unknown_symbol", "time", "timezone", "string_date",
    "float32", "valid_nan", "valid_inf", "invalid_number", "flag_disagrees", "flag_string",
    "unknown_status", "missing_status", "extra_column", "missing_column",
])
def test_malformed_factor_grid_is_rejected(factor_result, damage):
    factor = deepcopy(factor_result)
    frame = factor.data
    valid_index = frame.index[frame.is_valid][0]
    if damage == "drop_row":
        frame = frame.iloc[:-1].copy()
    elif damage == "duplicate_row":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif damage == "reorder":
        frame = frame.iloc[::-1].copy()
    elif damage == "unknown_symbol":
        frame.loc[0, "symbol"] = "000001"
    elif damage == "time":
        frame.loc[0, "date"] += pd.Timedelta(hours=1)
    elif damage == "timezone":
        frame["date"] = frame.date.dt.tz_localize("UTC")
    elif damage == "string_date":
        frame["date"] = frame.date.astype(str)
    elif damage == "float32":
        frame["factor_value"] = frame.factor_value.astype("float32")
    elif damage in ("valid_nan", "valid_inf"):
        frame.loc[valid_index, "factor_value"] = np.nan if damage == "valid_nan" else np.inf
    elif damage == "invalid_number":
        frame.loc[0, "factor_value"] = 0.0
    elif damage == "flag_disagrees":
        frame.loc[valid_index, "is_valid"] = False
    elif damage == "flag_string":
        frame["is_valid"] = frame.is_valid.astype(str)
    elif damage == "unknown_status":
        frame.loc[0, "status"] = "other"
    elif damage == "missing_status":
        frame.loc[0, "status"] = pd.NA
    elif damage == "extra_column":
        frame["extra"] = 1
    elif damage == "missing_column":
        frame = frame.drop(columns="factor_value")
    with pytest.raises(ValueError):
        schedule(replace(factor, data=frame), top_n=1)


@pytest.mark.parametrize("damage", ["empty", "duplicate", "reverse", "timestamps", "list", "missing_day"])
def test_invalid_calendar_grid_is_rejected(factor_result, damage):
    dates = factor_result.calendar.dates
    broken = {
        "empty": (), "duplicate": dates + dates[-1:], "reverse": dates[::-1],
        "timestamps": tuple(map(pd.Timestamp, dates)), "list": list(dates), "missing_day": dates[1:],
    }[damage]
    factor = replace(factor_result, calendar=replace(factor_result.calendar, dates=broken))
    with pytest.raises(ValueError):
        schedule(factor, top_n=1)


@pytest.mark.parametrize("damage", ["missing_source", "duplicate_source", "invalid_code", "unsorted_sources"])
def test_invalid_source_symbols_are_rejected(factor_result, damage):
    sources = deepcopy(factor_result.sources)
    if damage == "missing_source":
        sources = sources[:1]
    elif damage == "duplicate_source":
        sources = sources + sources[:1]
    elif damage == "invalid_code":
        sources[0]["symbol"] = "bad"
    else:
        sources = sources[::-1]
    with pytest.raises(ValueError):
        schedule(replace(factor_result, sources=sources), top_n=1)


def test_wrong_boundary_types_fail_clearly(factor_result):
    from quantlab.portfolio.selection import build_rebalance_schedule
    with pytest.raises(ValueError):
        build_rebalance_schedule(None, config=config(top_n=1))
    with pytest.raises(ValueError):
        build_rebalance_schedule(factor_result, config={"top_n": 1})
