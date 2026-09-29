"""从真实临时 CSV 到绩效的组合验收，不读取个人数据或调用外部行情。"""

import ast
from copy import deepcopy
from dataclasses import asdict
from importlib.util import resolve_name
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.config.settings import BacktestConfig
from quantlab.data.research_storage import load_research_set, save_research_set
from quantlab.portfolio import (
    PortfolioConfig, analyze_portfolio_result, run_factor_portfolio,
)


def _files(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def _assert_preview_unchanged(actual, before):
    pd.testing.assert_frame_equal(actual.alignment.data, before.alignment.data)
    pd.testing.assert_frame_equal(actual.alignment.coverage, before.alignment.coverage)
    pd.testing.assert_frame_equal(actual.coverage, before.coverage)
    assert actual.alignment.calendar == before.alignment.calendar
    assert actual.alignment.content_sha256 == before.alignment.content_sha256
    assert actual.sources == before.sources
    assert actual.warnings == before.warnings
    assert actual.preview_sha256 == before.preview_sha256


def test_saved_snapshot_runs_without_original_files(make_portfolio_preview, tmp_path):
    preview, source_root = make_portfolio_preview(periods=65)
    config = PortfolioConfig(top_n=2)
    original = run_factor_portfolio(preview, factor_name="momentum_20", config=config)
    saved = save_research_set(preview, tmp_path / "sets", name="组合验收合成集合")
    saved_bytes = _files(saved.directory)
    source_root.rename(source_root.with_name(source_root.name + "_moved"))
    before_files = _files(tmp_path)
    before_paths = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))

    loaded = load_research_set(tmp_path / "sets", saved.set_id).preview
    before = deepcopy(loaded)
    result = run_factor_portfolio(loaded, factor_name="momentum_20", config=config)
    analysis = analyze_portfolio_result(result)

    assert loaded.preview_sha256 == result.metadata["data"]["preview_sha256"] == preview.preview_sha256
    assert len(result.equity_curve) == 65
    assert len(result.positions) == 130
    assert not result.trades.empty and result.trades.quantity.gt(0).all()
    assert analysis.performance.cumulative_return is not None
    for name in ("decisions", "targets", "orders", "trades", "positions", "equity_curve",
                 "input_data", "factor_data"):
        pd.testing.assert_frame_equal(getattr(original, name), getattr(result, name))
        pd.testing.assert_frame_equal(getattr(result, name), getattr(analysis.backtest, name))
    left_meta, right_meta = deepcopy(original.metadata), deepcopy(result.metadata)
    for name in ("run_id", "run_at"):
        left_meta.pop(name)
        right_meta.pop(name)
    assert left_meta == right_meta
    assert original.metadata["run_id"] != result.metadata["run_id"]
    assert result.metadata["analytics_config"] is None
    _assert_preview_unchanged(loaded, before)
    assert _files(saved.directory) == saved_bytes
    assert _files(tmp_path) == before_files
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == before_paths


@pytest.mark.parametrize("factor_name", ["momentum_20", "volatility_20"])
@pytest.mark.parametrize("top_n", [1, 2])
def test_future_changes_do_not_change_earlier_execution(make_portfolio_preview, factor_name, top_n):
    preview, _ = make_portfolio_preview(periods=65)
    changed, _ = make_portfolio_preview(periods=65, bump_after=30)
    config = PortfolioConfig(top_n=top_n, rebalance_interval=1)
    first = run_factor_portfolio(preview, factor_name=factor_name, config=config)
    second = run_factor_portfolio(changed, factor_name=factor_name, config=config)
    cutoff = pd.Timestamp(preview.alignment.calendar.dates[30])
    assert first.metadata["data"]["preview_sha256"] != second.metadata["data"]["preview_sha256"]
    assert not first.trades.loc[first.trades.execution_date < cutoff].empty
    for name, date_fields in (
        ("decisions", ("signal_date", "execution_date")),
        ("targets", ("signal_date",)),
        ("orders", ("signal_date", "execution_date")),
        ("trades", ("signal_date", "execution_date")),
        ("positions", ("date",)), ("equity_curve", ("date",)),
        ("input_data", ("date",)), ("factor_data", ("date",)),
    ):
        slices = []
        for result in (first, second):
            table = getattr(result, name)
            mask = pd.Series(True, index=table.index)
            for field in date_fields:
                mask &= table[field] < cutoff
            slices.append(table.loc[mask].reset_index(drop=True))
        pd.testing.assert_frame_equal(*slices)


def test_supplied_axis_keeps_all_missing_cash_day(make_portfolio_preview):
    gaps = {"510300": [3], "159915": [3]}
    supplied, _ = make_portfolio_preview(periods=65, gaps=gaps, supplied=True)
    union, _ = make_portfolio_preview(periods=65, gaps=gaps)
    missing_day = pd.Timestamp(supplied.alignment.calendar.dates[3])
    config = PortfolioConfig(top_n=2)
    kept = run_factor_portfolio(supplied, factor_name="momentum_20", config=config)
    omitted = run_factor_portfolio(union, factor_name="momentum_20", config=config)
    assert len(kept.equity_curve) == 65 and len(omitted.equity_curve) == 64
    assert kept.metadata["data"]["preview_sha256"] != omitted.metadata["data"]["preview_sha256"]
    assert missing_day not in omitted.equity_curve.date.tolist()
    row = kept.equity_curve.set_index("date").loc[missing_day]
    assert row.equity == row.cash == config.backtest.initial_cash
    assert row.position_value == 0
    positions = kept.positions.loc[kept.positions.date == missing_day]
    assert len(positions) == 2 and positions.quantity.eq(0).all()
    assert positions.position_value.eq(0).all() and positions.close.isna().all()
    assert not kept.input_data.loc[kept.input_data.date == missing_day, "has_observation"].any()
    analysis = analyze_portfolio_result(kept)
    assert len(analysis.performance.daily_returns) == 64
    assert analysis.performance.daily_returns.loc[missing_day] == 0


def test_stage1_has_no_portfolio_or_factor_dependency():
    """解析导入语句而非搜索注释；第二阶段对旧模块的单向依赖不在禁用范围。"""
    root = Path(__file__).resolve().parents[1] / "src" / "quantlab"
    paths = [path for area in ("data", "analytics", "strategies", "backtest", "config")
             for path in (root / area).rglob("*.py")]
    paths.append(root / "comparison.py")
    assert all(any(area in path.parts for path in paths)
               for area in ("data", "analytics", "strategies", "backtest"))
    forbidden = ("quantlab.portfolio", "quantlab.factors")
    violations = []
    for path in paths:
        module = "quantlab." + ".".join(path.relative_to(root).with_suffix("").parts)
        package = module.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = resolve_name("." * node.level + base, package)
                names = [base] + [f"{base}.{alias.name}" for alias in node.names]
            if any(name == prefix or name.startswith(prefix + ".")
                   for name in names for prefix in forbidden):
                violations.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not violations, violations


@pytest.mark.parametrize("factor_name", [
    "momentum_20", "momentum_60", "momentum_120", "volatility_20", "volatility_60",
])
@pytest.mark.parametrize("ascending", [None, True, False])
@pytest.mark.parametrize("top_n,interval,lot,rate,minimum,tax,slip", [
    (1, 1, 1, 0.0, 0.0, 0.0, 0.0),
    (2, 7, 10, 0.001, 5.0, 0.002, 0.003),
    (1, 20, 100, 0.0003, 10.0, 0.001, 0.0001),
])
def test_public_roundtrip_accounting_invariants(
    make_portfolio_preview, tmp_path, factor_name, ascending,
    top_n, interval, lot, rate, minimum, tax, slip,
):
    preview, _ = make_portfolio_preview(periods=145 if factor_name == "momentum_120" else 85)
    saved = save_research_set(preview, tmp_path / "sets", name="账本不变量验收")
    loaded = load_research_set(tmp_path / "sets", saved.set_id).preview
    before = deepcopy(loaded)
    config = PortfolioConfig(top_n=top_n, rebalance_interval=interval, ascending=ascending,
                             backtest=BacktestConfig(initial_cash=100000, lot_size=lot,
                             commission_rate=rate, minimum_commission=minimum,
                             sell_tax_rate=tax, slippage_rate=slip))
    result = run_factor_portfolio(loaded, factor_name=factor_name, config=config)
    analysis = analyze_portfolio_result(result)
    assert result.metadata["strategy"]["config"] == asdict(config)
    assert not result.trades.empty
    assert result.trades.execution_date.gt(result.trades.signal_date).all()
    assert result.targets.target_weight.eq(1 / top_n).all()
    assert result.targets.groupby("decision_id").size().eq(top_n).all()
    assert result.orders.requested_quantity.eq(
        result.orders.filled_quantity + result.orders.unfilled_quantity).all()
    _assert_preview_unchanged(loaded, before)
    assert result.metadata["analytics_config"] is None
    assert analysis.backtest.metadata["run_id"] == result.metadata["run_id"]

    # 独立重放公开成交表核对现金和份额，不调用引擎的内部记账函数。
    cash, fees, slippage = 100000.0, 0.0, 0.0
    holdings = dict.fromkeys(result.metadata["data"]["symbols"], 0)
    by_day = {day: frame for day, frame in result.trades.groupby("execution_date")}
    for row in result.equity_curve.itertuples(index=False):
        for trade in by_day.get(row.date, result.trades.iloc[:0]).itertuples(index=False):
            sign = 1 if trade.side == "BUY" else -1
            assert trade.quantity > 0 and trade.quantity % lot == 0
            assert trade.execution_price == pytest.approx(trade.open_price * (1 + sign * slip))
            assert trade.notional == pytest.approx(trade.quantity * trade.execution_price)
            assert trade.commission == pytest.approx(max(trade.notional * rate, minimum))
            assert trade.tax == pytest.approx(trade.notional * tax if sign == -1 else 0)
            assert trade.slippage_cost == pytest.approx(
                trade.quantity * abs(trade.execution_price - trade.open_price))
            cash = fsum((cash, -sign * trade.notional, -trade.commission, -trade.tax))
            holdings[trade.symbol] += sign * trade.quantity
            fees = fsum((fees, trade.commission, trade.tax))
            slippage += trade.slippage_cost
            assert trade.cash_after == pytest.approx(cash, abs=1e-8)
            assert trade.position_after == holdings[trade.symbol]
            assert trade.cash_after >= 0 and trade.position_after >= 0
        positions = result.positions.loc[result.positions.date == row.date]
        assert positions.symbol.tolist() == sorted(holdings)
        assert positions.quantity.tolist() == [holdings[symbol] for symbol in sorted(holdings)]
        assert positions.quantity.ge(0).all() and positions.quantity.mod(lot).eq(0).all()
        np.testing.assert_allclose(positions.position_value, positions.quantity * positions.close)
        assert row.position_value == pytest.approx(fsum(positions.position_value))
        assert row.cash >= 0 and row.cash == pytest.approx(cash, abs=1e-8)
        assert row.equity == pytest.approx(cash + fsum(positions.position_value))
        np.testing.assert_allclose(positions.weight, positions.position_value / row.equity)
        assert fsum(positions.weight) + row.cash / row.equity == pytest.approx(1.0)
        assert row.cumulative_fees_taxes == pytest.approx(fees)
        assert row.cumulative_slippage == pytest.approx(slippage)
        assert row.net_value == pytest.approx(row.equity / 100000)
        assert row.cumulative_return == pytest.approx(row.equity / 100000 - 1)

    equity = result.equity_curve.set_index("date").equity
    assert analysis.performance.cumulative_return == pytest.approx(equity.iloc[-1] / 100000 - 1)
    np.testing.assert_allclose(analysis.performance.daily_returns, equity.pct_change().iloc[1:])
    np.testing.assert_allclose(analysis.performance.net_value, equity / 100000)
    assert len(analysis.performance.net_value) == len(loaded.alignment.calendar.dates)
