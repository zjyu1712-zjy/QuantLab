"""20条价格窗口最大回撤因子的真实研究、组合及档案集成契约。"""

from copy import deepcopy

import pandas as pd
import pytest

from quantlab.factors import create_default_registry, run_research_factor
from quantlab.portfolio import (
    PortfolioConfig, analyze_portfolio_result, run_equal_weight_benchmark,
    run_factor_portfolio,
)
from quantlab.portfolio.selection import build_rebalance_schedule
from quantlab.ui.factors import ranked_factor_table
from test_bias_portfolio import (
    test_archive_preserves_new_and_old_definitions_without_recomputing as check_archive_roundtrip,
)


NAME = "max_drawdown_20"


def factor(preview):
    assert NAME in create_default_registry().names()
    return run_research_factor(preview, NAME)


def test_real_research_manual_values_window_reset_and_input_unchanged(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=22)
    before = deepcopy(preview)
    result = factor(preview)
    definition = result.definition
    assert definition.name == NAME
    assert definition.description == "20 期价格最大回撤"
    assert definition.version == "1.0"
    assert definition.lookback_periods == 19
    assert definition.window_observations == 20
    assert definition.direction == "higher_is_less_drawdown"
    assert definition.availability == "after_close"
    assert definition.missing_policy == "strict_window"
    up = result.data.loc[result.data.symbol.eq("510300")].reset_index(drop=True)
    down = result.data.loc[result.data.symbol.eq("159915")].reset_index(drop=True)
    assert up.factor_value.iloc[19:].eq(0.0).all()
    assert down.factor_value.iloc[19] == pytest.approx(190.5 / 200 - 1)
    assert down.factor_value.iloc[20] == pytest.approx(190 / 199.5 - 1)
    pd.testing.assert_frame_equal(preview.alignment.data, before.alignment.data)
    pd.testing.assert_frame_equal(preview.alignment.coverage, before.alignment.coverage)
    pd.testing.assert_frame_equal(preview.coverage, before.coverage)
    assert preview.sources == before.sources
    assert preview.warnings == before.warnings
    assert preview.preview_sha256 == before.preview_sha256
    assert preview.alignment.content_sha256 == before.alignment.content_sha256
    assert run_research_factor(preview).definition.name == "momentum_20"


def test_recovery_does_not_erase_window_trough_and_old_peak_exits(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=40, bump_after=20)
    result = factor(preview)
    down = result.data.loc[result.data.symbol.eq("159915")].reset_index(drop=True)
    # 当期翻倍并创高，当前回撤为0；窗口仍包含此前峰谷。
    assert down.factor_value.iloc[20] == pytest.approx(190.5 / 199.5 - 1)
    # 最后窗口仅包含翻倍后的20条价格，不能继承窗口外旧回撤。
    assert down.factor_value.iloc[39] == pytest.approx(361 / 380 - 1)


@pytest.mark.parametrize("periods", [19, 20, 21])
def test_exact_warmup_last_day_expiry_and_next_open(make_portfolio_preview, periods):
    preview, _ = make_portfolio_preview(periods=periods)
    values = factor(preview)
    for symbol in ("159915", "510300"):
        asset = values.data.loc[values.data.symbol.eq(symbol)].reset_index(drop=True)
        assert asset.status.iloc[:19].eq("warmup").all()
        assert asset.factor_value.iloc[:19].isna().all()
        if periods >= 20:
            assert asset.status.iloc[19:].eq("valid").all()
    result = run_factor_portfolio(preview, factor_name=NAME, config=PortfolioConfig(top_n=1))
    assert result.metadata["strategy"]["signal_anchor_position"] == 19
    assert result.metadata["strategy"]["lookback_periods"] == 19
    assert result.equity_curve.iloc[:min(20, periods)].equity.eq(100000).all()
    if periods == 19:
        assert result.decisions.empty and result.trades.empty
    elif periods == 20:
        assert result.decisions.status.tolist() == ["EXPIRED"]
        assert result.decisions.execution_date.isna().all()
        assert result.trades.empty
    else:
        assert result.trades.symbol.tolist() == ["510300"]
        assert result.trades.signal_date.eq(pd.Timestamp(values.calendar.dates[19])).all()
        assert result.trades.execution_date.eq(pd.Timestamp(values.calendar.dates[20])).all()


@pytest.mark.parametrize("ascending,expected", [(None, "510300"), (False, "510300"), (True, "159915")])
def test_zero_and_negative_values_default_and_reversed_ranking(make_portfolio_preview, ascending, expected):
    preview, _ = make_portfolio_preview(periods=21)
    values = factor(preview)
    table = ranked_factor_table(values, values.calendar.dates[19], ascending=ascending)
    assert table["ETF 代码"].iloc[0] == expected
    assert table["名次"].tolist() == ["1", "2"]
    result = run_factor_portfolio(preview, factor_name=NAME, config=PortfolioConfig(top_n=1, ascending=ascending))
    assert result.trades.symbol.tolist() == [expected]
    assert result.metadata["strategy"]["resolved_ascending"] is (ascending is True)


def test_missing_window_recovers_without_compressing_axis_or_moving_anchor(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=41, gaps={"159915": [5]}, supplied=True)
    values = factor(preview)
    asset = values.data.loc[values.data.symbol.eq("159915")].reset_index(drop=True)
    assert len(asset) == 41
    assert asset.status.iloc[:19].eq("warmup").all()
    assert asset.status.iloc[19:25].eq("missing_window").all()
    assert asset.factor_value.iloc[19:25].isna().all()
    assert asset.status.iloc[25:].eq("valid").all()
    assert asset.factor_value.iloc[25] == pytest.approx(187.5 / 197 - 1)
    schedule = build_rebalance_schedule(values, config=PortfolioConfig(top_n=2))
    assert schedule.decisions.signal_date.tolist() == [pd.Timestamp(values.calendar.dates[i]) for i in (19, 39)]
    assert schedule.decisions.status.tolist() == ["SKIPPED_INSUFFICIENT_FACTORS", "PENDING"]


def test_benchmark_keeps_anchor_despite_skips_and_rejects_misaligned_pair(make_portfolio_preview, tmp_path):
    from quantlab.portfolio.archive import save_portfolio_run

    preview, _ = make_portfolio_preview(periods=22, gaps={"510300": [2]})
    factor(preview)
    strategy = analyze_portfolio_result(run_factor_portfolio(preview, factor_name=NAME, config=PortfolioConfig(top_n=2)))
    assert strategy.backtest.trades.empty
    benchmark = run_equal_weight_benchmark(preview, factor_name=NAME)
    assert benchmark.backtest.metadata["strategy"]["signal_anchor_position"] == 19
    assert benchmark.backtest.metadata["strategy"]["anchor_factor"] == {"name": NAME, "version": "1.0"}
    assert benchmark.backtest.trades.execution_date.eq(pd.Timestamp(preview.alignment.calendar.dates[20])).all()
    assert set(benchmark.backtest.trades.symbol) == {"159915", "510300"}
    wrong = run_equal_weight_benchmark(preview, factor_name="momentum_20")
    assert wrong.backtest.metadata["strategy"]["signal_anchor_position"] == 20
    root = tmp_path / "mismatched_runs"
    with pytest.raises(ValueError):
        save_portfolio_run(strategy, root, name="不可混配", benchmark=wrong)
    assert not root.exists()


@pytest.mark.parametrize("name,lookback", [
    (NAME, 19), ("bias_20", 19), ("momentum_20", 20), ("momentum_60", 60),
    ("momentum_120", 120), ("volatility_20", 20), ("volatility_60", 60),
])
def test_new_and_six_existing_factor_archives_without_recomputation(make_portfolio_preview, tmp_path, monkeypatch, name, lookback):
    assert name in create_default_registry().names()
    # 复用现有真实档案往返契约，不修改旧文件或重建读取认证逻辑。
    check_archive_roundtrip(make_portfolio_preview, tmp_path, monkeypatch, name, lookback)
