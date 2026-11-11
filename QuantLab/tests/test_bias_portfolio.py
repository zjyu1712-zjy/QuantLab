"""固定20观测乖离率在真实研究、筛选、组合和档案链中的集成契约。"""

from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from quantlab.factors import create_default_registry, run_research_factor
from quantlab.portfolio import (
    PortfolioConfig, analyze_portfolio_result, run_equal_weight_benchmark,
    run_factor_portfolio,
)
from quantlab.portfolio.selection import build_rebalance_schedule
from quantlab.ui.factors import ranked_factor_table, top_n_factor_preview


def bias(preview):
    assert "bias_20" in create_default_registry().names()
    return run_research_factor(preview, "bias_20")


def test_research_definition_manual_value_and_existing_default(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=22)
    before = deepcopy(preview)
    result = bias(preview)
    definition = result.definition
    assert definition.name == "bias_20"
    assert definition.version == "1.0"
    assert definition.lookback_periods == 19
    assert definition.window_observations == 20
    assert definition.direction == "higher_is_more_above_ma"
    assert definition.missing_policy == "strict_window"
    assert definition.availability == "after_close"
    first = result.data.loc[result.data.date.eq(pd.Timestamp(result.calendar.dates[19]))]
    expected = {"510300": 109.5 / 104.75 - 1, "159915": 190.5 / 195.25 - 1}
    for row in first.itertuples():
        assert row.factor_value == pytest.approx(expected[row.symbol])
    default = run_research_factor(preview)
    assert default.definition.name == "momentum_20"
    assert default.definition.lookback_periods == 20
    assert default.definition.window_observations == 21
    pd.testing.assert_frame_equal(preview.alignment.data, before.alignment.data)
    assert preview.preview_sha256 == before.preview_sha256


@pytest.mark.parametrize("periods", [19, 20, 21])
def test_exact_warmup_expiry_and_next_open(make_portfolio_preview, periods):
    preview, _ = make_portfolio_preview(periods=periods)
    factor = bias(preview)
    assert factor.data.loc[factor.data.date.lt(pd.Timestamp(factor.calendar.dates[min(19, periods - 1)]))].status.eq("warmup").all()
    result = run_factor_portfolio(preview, factor_name="bias_20", config=PortfolioConfig(top_n=1))
    assert result.metadata["strategy"]["signal_anchor_position"] == 19
    assert result.metadata["strategy"]["resolved_ascending"] is False
    if periods == 19:
        assert factor.data.status.eq("warmup").all()
        assert result.decisions.empty and result.trades.empty
    elif periods == 20:
        assert factor.data.loc[factor.data.date.eq(pd.Timestamp(factor.calendar.dates[19]))].status.eq("valid").all()
        assert result.decisions.status.tolist() == ["EXPIRED"]
        assert result.decisions.execution_date.isna().all()
        assert result.trades.empty
    else:
        assert result.trades.signal_date.eq(pd.Timestamp(factor.calendar.dates[19])).all()
        assert result.trades.execution_date.eq(pd.Timestamp(factor.calendar.dates[20])).all()
        assert result.trades.symbol.tolist() == ["510300"]
    assert result.equity_curve.iloc[:min(20, periods)].equity.eq(100000).all()


@pytest.mark.parametrize("ascending,expected", [(None, "510300"), (False, "510300"), (True, "159915")])
def test_real_portfolio_direction_can_be_reversed(make_portfolio_preview, ascending, expected):
    preview, _ = make_portfolio_preview(periods=21)
    bias(preview)
    result = run_factor_portfolio(preview, factor_name="bias_20", config=PortfolioConfig(top_n=1, ascending=ascending))
    assert result.trades.symbol.tolist() == [expected]
    assert result.metadata["strategy"]["resolved_ascending"] is (ascending is True)


def test_negative_zero_and_exact_ties_share_ui_and_selection(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=21)
    base = bias(preview)
    # 内部筛选契约专用网格；不作为经公式核验的公开回测输入。
    values = {"159915": -.1, "510050": 0.0, "510300": .000041, "512000": .000041, "512100": .000040}
    rows = []
    for position, day in enumerate(base.calendar.dates):
        for symbol, value in values.items():
            valid = position >= 19
            rows.append((pd.Timestamp(day), symbol, value if valid else np.nan, valid, "valid" if valid else "warmup"))
    frame = pd.DataFrame(rows, columns=base.data.columns).astype({
        "date": "datetime64[ns]", "symbol": "string", "factor_value": "float64",
        "is_valid": "bool", "status": "string",
    })
    synthetic = replace(base, data=frame, sources=tuple({**base.sources[0], "symbol": symbol} for symbol in values))
    day = base.calendar.dates[19]
    descending = ranked_factor_table(synthetic, day)
    assert descending["ETF 代码"].tolist() == ["510300", "512000", "512100", "510050", "159915"]
    assert descending["名次"].tolist() == ["1", "1", "3", "4", "5"]
    ascending = ranked_factor_table(synthetic, day, ascending=True)
    assert ascending["ETF 代码"].tolist() == ["159915", "510050", "512100", "510300", "512000"]
    assert ascending["名次"].tolist() == ["1", "2", "3", "4", "4"]
    selected = top_n_factor_preview(synthetic, day, 1)
    assert selected.cutoff_tie is True
    assert selected.table["ETF 代码"].tolist() == ["510300"]
    schedule = build_rebalance_schedule(synthetic, config=PortfolioConfig(top_n=1))
    assert schedule.ascending is False
    assert schedule.targets.symbol.tolist() == ["510300"]
    assert schedule.decisions.cutoff_tie.tolist() == [True]


def test_gap_stays_on_axis_recovers_and_does_not_move_anchor(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=41, gaps={"159915": [5]}, supplied=True)
    result = bias(preview)
    asset = result.data.loc[result.data.symbol.eq("159915")].reset_index(drop=True)
    assert len(asset) == 41
    assert asset.status.iloc[:19].eq("warmup").all()
    assert asset.status.iloc[19:25].eq("missing_window").all()
    assert asset.factor_value.iloc[19:25].isna().all()
    assert asset.status.iloc[25:].eq("valid").all()
    schedule = build_rebalance_schedule(result, config=PortfolioConfig(top_n=2))
    assert schedule.decisions.signal_date.tolist() == [pd.Timestamp(result.calendar.dates[i]) for i in (19, 39)]
    assert schedule.decisions.status.tolist() == ["SKIPPED_INSUFFICIENT_FACTORS", "PENDING"]


def test_future_prices_do_not_change_earlier_factor_rows(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=30)
    changed, _ = make_portfolio_preview(periods=30, bump_after=25)
    first, second = bias(preview), bias(changed)
    cutoff = pd.Timestamp(first.calendar.dates[25])
    pd.testing.assert_frame_equal(first.data.loc[first.data.date.lt(cutoff)], second.data.loc[second.data.date.lt(cutoff)], check_exact=True)
    assert not first.data.loc[first.data.date.eq(cutoff), "factor_value"].equals(second.data.loc[second.data.date.eq(cutoff), "factor_value"])


def test_benchmark_anchor_is_19_even_when_strategy_skips(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=22, gaps={"510300": [2]})
    bias(preview)
    strategy = run_factor_portfolio(preview, factor_name="bias_20", config=PortfolioConfig(top_n=2))
    assert strategy.trades.empty
    benchmark = run_equal_weight_benchmark(preview, factor_name="bias_20").backtest
    assert benchmark.metadata["strategy"]["signal_anchor_position"] == 19
    assert benchmark.metadata["strategy"]["anchor_factor"] == {"name": "bias_20", "version": "1.0"}
    assert benchmark.trades.execution_date.eq(pd.Timestamp(preview.alignment.calendar.dates[20])).all()
    assert set(benchmark.trades.symbol) == {"159915", "510300"}


@pytest.mark.parametrize("name,lookback", [
    ("bias_20", 19), ("momentum_20", 20), ("momentum_60", 60),
    ("momentum_120", 120), ("volatility_20", 20), ("volatility_60", 60),
])
def test_archive_preserves_new_and_old_definitions_without_recomputing(make_portfolio_preview, tmp_path, monkeypatch, name, lookback):
    from quantlab.portfolio.archive import load_portfolio_run, save_portfolio_run

    if name == "bias_20":
        assert name in create_default_registry().names()
    preview, _ = make_portfolio_preview(periods=lookback + 3)
    factor = run_research_factor(preview, name)
    strategy = analyze_portfolio_result(run_factor_portfolio(preview, factor_name=name, config=PortfolioConfig(top_n=1)))
    benchmark = run_equal_weight_benchmark(preview, factor_name=name)
    record = save_portfolio_run(strategy, tmp_path / "runs", name=name, benchmark=benchmark)
    original = record.path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("档案读取不得重新计算因子、组合或绩效")

    monkeypatch.setattr("quantlab.factors.run_research_factor", forbidden)
    monkeypatch.setattr("quantlab.factors.research.run_research_factor", forbidden)
    monkeypatch.setattr("quantlab.portfolio.research.run_research_factor", forbidden)
    monkeypatch.setattr("quantlab.portfolio.research.run_factor_portfolio", forbidden)
    monkeypatch.setattr("quantlab.portfolio.benchmark.run_equal_weight_benchmark", forbidden)
    monkeypatch.setattr("quantlab.portfolio.analysis._performance_from_equity", forbidden)
    loaded = load_portfolio_run(record.path.parent, record.run_id)
    assert loaded.benchmark_status == "complete"
    assert loaded.strategy.backtest.metadata == strategy.backtest.metadata
    assert loaded.strategy.backtest.metadata["factor_definition"] == asdict(factor.definition)
    assert loaded.strategy.backtest.metadata["strategy"]["signal_anchor_position"] == lookback
    for field in ("factor_data", "input_data", "decisions", "targets", "trades", "equity_curve"):
        pd.testing.assert_frame_equal(getattr(loaded.strategy.backtest, field), getattr(strategy.backtest, field), check_exact=True)
    assert loaded.benchmark.backtest.metadata == benchmark.backtest.metadata
    pd.testing.assert_frame_equal(loaded.benchmark.backtest.equity_curve, benchmark.backtest.equity_curve, check_exact=True)
    assert record.path.read_bytes() == original
