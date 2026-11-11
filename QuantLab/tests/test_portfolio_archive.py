"""组合档案使用临时合成数据；历史读取绝不重新运行研究。"""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json

import pandas as pd
import pytest

from quantlab.config.settings import AnalyticsConfig, BacktestConfig
from quantlab.portfolio import (
    PortfolioBacktestError, PortfolioConfig, analyze_portfolio_result,
    run_equal_weight_benchmark, run_factor_portfolio,
)


def api():
    from quantlab.portfolio import archive
    return archive


@pytest.fixture
def experiment(make_portfolio_preview):
    preview, source = make_portfolio_preview(periods=45)
    strategy = analyze_portfolio_result(run_factor_portfolio(
        preview, factor_name="momentum_20", config=PortfolioConfig(top_n=1),
    ))
    benchmark = run_equal_weight_benchmark(preview, factor_name="momentum_20")
    return strategy, benchmark, source


def assert_analysis_equal(actual, expected):
    assert actual.backtest.metadata == expected.backtest.metadata
    for key in ("input_data", "factor_data", "decisions", "targets", "orders", "trades",
                "positions", "equity_curve"):
        left, right = getattr(actual.backtest, key), getattr(expected.backtest, key)
        if right is None:
            assert left is None
        else:
            pd.testing.assert_frame_equal(left, right, check_exact=True)
    for key in ("daily_returns", "net_value", "drawdown"):
        pd.testing.assert_series_equal(getattr(actual.performance, key), getattr(expected.performance, key), check_exact=True)
    for key in ("cumulative_return", "annualized_return", "annualized_volatility", "max_drawdown",
                "sharpe_ratio", "unavailable_reasons"):
        assert getattr(actual.performance, key) == getattr(expected.performance, key)


def rewrite(path, mutate, *, checksum=True):
    payload = json.loads(path.read_text())
    mutate(payload)
    if checksum:
        core = {key: value for key, value in payload.items() if key != "content_sha256"}
        payload["content_sha256"] = sha256(json.dumps(
            core, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False))


def test_roundtrip_independent_of_sources_and_current_formulas(experiment, tmp_path, monkeypatch):
    strategy, benchmark, source = experiment
    archive = api()
    record = archive.save_portfolio_run(strategy, tmp_path / "runs", name="动量实验", benchmark=benchmark)
    original = record.path.read_bytes()
    source.rename(source.with_name("sources_moved"))

    def forbidden(*args, **kwargs):
        pytest.fail("历史读取不能计算因子、重新回测或重算绩效")

    monkeypatch.setattr("quantlab.portfolio.analysis._performance_from_equity", forbidden)
    monkeypatch.setattr("quantlab.portfolio.research.run_factor_portfolio", forbidden)
    monkeypatch.setattr("quantlab.portfolio.benchmark.run_equal_weight_benchmark", forbidden)
    monkeypatch.setattr("quantlab.factors.run_research_factor", forbidden)
    monkeypatch.setattr("quantlab.backtest.provenance.source_tree_sha256", forbidden)
    monkeypatch.setattr("quantlab.backtest.provenance.create_metadata", forbidden)
    loaded = archive.load_portfolio_run(record.path.parent, record.run_id)
    assert loaded.record == record
    assert loaded.benchmark_status == "complete"
    assert loaded.benchmark_failure is None
    assert_analysis_equal(loaded.strategy, strategy)
    assert_analysis_equal(loaded.benchmark, benchmark)
    assert record.path.read_bytes() == original
    assert archive.list_portfolio_runs(record.path.parent).records == (record,)
    loaded.strategy.backtest.metadata["warnings"].append("仅修改内存")
    assert archive.load_portfolio_run(record.path.parent, record.run_id).strategy.backtest.metadata == strategy.backtest.metadata


@pytest.mark.parametrize("periods", [1, 2, 20, 21])
def test_empty_tables_short_cash_series_and_nulls(make_portfolio_preview, tmp_path, periods):
    preview, _ = make_portfolio_preview(periods=periods)
    strategy = analyze_portfolio_result(run_factor_portfolio(
        preview, factor_name="momentum_20", config=PortfolioConfig(top_n=1),
    ))
    benchmark = run_equal_weight_benchmark(preview, factor_name="momentum_20")
    record = api().save_portfolio_run(strategy, tmp_path / "runs", name="现金", benchmark=benchmark)
    loaded = api().load_portfolio_run(record.path.parent, record.run_id)
    assert_analysis_equal(loaded.strategy, strategy)
    assert_analysis_equal(loaded.benchmark, benchmark)
    assert "NaN" not in record.path.read_text()


def test_refuses_overwrite_and_leaves_no_temp_files(experiment, tmp_path):
    strategy, benchmark, _ = experiment
    root = tmp_path / "runs"
    record = api().save_portfolio_run(strategy, root, name="第一次", benchmark=benchmark)
    original = record.path.read_bytes()
    with pytest.raises(FileExistsError):
        api().save_portfolio_run(strategy, root, name="不能覆盖", benchmark=benchmark)
    assert record.path.read_bytes() == original
    assert list(root.iterdir()) == [record.path]


@pytest.mark.parametrize("with_failure", [False, True])
def test_strategy_alone_explicit_benchmark_state(experiment, tmp_path, with_failure):
    strategy, _, _ = experiment
    error = PortfolioBacktestError("secret-token=/private/credential", reason="missing_execution_price",
                                   date=pd.Timestamp("2025-02-03").date(), symbols=("510300",)) if with_failure else None
    record = api().save_portfolio_run(strategy, tmp_path / "runs", name="仅策略", benchmark_error=error)
    loaded = api().load_portfolio_run(record.path.parent, record.run_id)
    assert loaded.benchmark is None
    assert loaded.benchmark_status == ("unavailable" if with_failure else "not_run")
    assert (loaded.benchmark_failure is not None) == with_failure
    assert "secret-token" not in record.path.read_text()
    assert_analysis_equal(loaded.strategy, strategy)


@pytest.mark.parametrize("change", ["market", "axis", "cost", "analytics", "anchor"])
def test_mixed_benchmark_rejected_before_writes(experiment, tmp_path, change):
    strategy, benchmark, _ = experiment
    benchmark = deepcopy(benchmark)
    meta = benchmark.backtest.metadata
    if change == "market":
        meta["data"]["preview_sha256"] = "a" * 64
    elif change == "axis":
        meta["data"]["calendar"]["dates"].pop()
    elif change == "cost":
        meta["strategy"]["config"]["backtest"]["slippage_rate"] = 0.02
    elif change == "analytics":
        meta["analytics_config"]["periods_per_year"] = 365
    else:
        meta["strategy"]["signal_anchor_position"] += 1
    root = tmp_path / "runs"
    with pytest.raises(ValueError):
        api().save_portfolio_run(strategy, root, name="混配", benchmark=benchmark)
    assert not root.exists()


@pytest.mark.parametrize("change", ["snapshot", "performance", "equity", "metadata_extra", "bool", "infinity"])
def test_invalid_strategy_rejected_before_writes(experiment, tmp_path, change):
    strategy, _, _ = experiment
    strategy = deepcopy(strategy)
    if change == "snapshot":
        strategy.backtest.input_data.loc[0, "close"] += 1
    elif change == "performance":
        strategy = replace(strategy, performance=replace(strategy.performance, annualized_return=123.0))
    elif change == "equity":
        strategy.backtest.equity_curve.loc[0, "cash"] -= 1
    elif change == "metadata_extra":
        strategy.backtest.metadata["token"] = "do-not-save"
    elif change == "bool":
        strategy.backtest.metadata["strategy"]["config"]["backtest"]["lot_size"] = True
    else:
        strategy.backtest.trades.loc[0, "commission"] = float("inf")
    root = tmp_path / "runs"
    with pytest.raises(ValueError):
        api().save_portfolio_run(strategy, root, name="坏结果")
    assert not root.exists()


def test_old_code_loads_with_warning_without_rewriting(experiment, tmp_path):
    strategy, _, _ = experiment
    record = api().save_portfolio_run(strategy, tmp_path / "runs", name="旧版本")
    rewrite(record.path, lambda doc: doc["strategy"]["metadata"].update(code_sha256="a" * 64))
    original = record.path.read_bytes()
    loaded = api().load_portfolio_run(record.path.parent, record.run_id)
    assert loaded.warnings
    assert loaded.strategy.backtest.metadata["code_sha256"] == "a" * 64
    assert record.path.read_bytes() == original


@pytest.mark.parametrize("change", ["checksum", "version", "dtype", "grid", "metrics_axis", "null_reason", "id"])
def test_corrupt_records_rejected_even_with_updated_checksum(experiment, tmp_path, change):
    strategy, _, _ = experiment
    record = api().save_portfolio_run(strategy, tmp_path / "runs", name="实验")

    def mutate(doc):
        if change == "checksum":
            doc["name"] = "改动"
        elif change == "version":
            doc["schema_version"] = "future-version"
        elif change == "dtype":
            doc["strategy"]["tables"]["orders"]["dtypes"]["filled_quantity"] = "float64"
        elif change == "grid":
            doc["strategy"]["tables"]["positions"]["rows"].pop()
        elif change == "metrics_axis":
            doc["strategy"]["performance"]["series"]["net_value"]["dates"].pop()
        elif change == "null_reason":
            doc["strategy"]["performance"]["metrics"]["sharpe_ratio"] = None
        else:
            doc["strategy"]["metadata"]["run_id"] = "../outside"

    rewrite(record.path, mutate, checksum=change != "checksum")
    with pytest.raises(ValueError):
        api().load_portfolio_run(record.path.parent, record.run_id)


def test_scan_isolates_bad_file_and_does_not_create_missing_root(experiment, tmp_path):
    archive = api()
    absent = tmp_path / "absent"
    assert archive.list_portfolio_runs(absent).records == ()
    assert not absent.exists()
    strategy, _, _ = experiment
    record = archive.save_portfolio_run(strategy, tmp_path / "runs", name="好记录")
    bad = record.path.parent / "00000000-0000-4000-8000-000000000000.json"
    bad.write_text('{"schema_version": "x", "schema_version": "y"}')
    scan = archive.list_portfolio_runs(record.path.parent)
    assert scan.records == (record,)
    assert len(scan.issues) == 1


@pytest.mark.parametrize("run_id", ["../outside", "", "not-a-uuid", "A" * 32])
def test_load_rejects_noncanonical_id(tmp_path, run_id):
    with pytest.raises(ValueError):
        api().load_portfolio_run(tmp_path, run_id)


def test_load_and_save_refuse_symlink(experiment, tmp_path):
    strategy, _, _ = experiment
    archive = api()
    record = archive.save_portfolio_run(strategy, tmp_path / "real", name="真实")
    other = tmp_path / "other"
    other.mkdir()
    linked = other / record.path.name
    linked.symlink_to(record.path)
    with pytest.raises(ValueError):
        archive.load_portfolio_run(other, record.run_id)
    original = record.path.read_bytes()
    with pytest.raises(FileExistsError):
        archive.save_portfolio_run(strategy, other, name="不覆盖链接")
    assert record.path.read_bytes() == original


@pytest.mark.parametrize("column,value", [("notional", 1.0), ("cash_after", 1.0), ("position_after", 0),
                                         ("commission", -1.0), ("slippage_cost", -1.0)])
def test_rejects_broken_trade_accounting(experiment, tmp_path, column, value):
    strategy, _, _ = experiment
    strategy = deepcopy(strategy)
    strategy.backtest.trades.loc[0, column] = value
    with pytest.raises(ValueError):
        api().save_portfolio_run(strategy, tmp_path / "runs", name="交易对账错误")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("supplied", [False, True])
def test_gaps_and_missing_cash_position_price_roundtrip(make_portfolio_preview, tmp_path, supplied):
    preview, _ = make_portfolio_preview(periods=45, gaps={"510300": [0, 3], "159915": [3]}, supplied=supplied)
    strategy = analyze_portfolio_result(run_factor_portfolio(
        preview, factor_name="momentum_20", config=PortfolioConfig(top_n=1),
    ))
    record = api().save_portfolio_run(strategy, tmp_path / "runs", name="缺失值")
    loaded = api().load_portfolio_run(record.path.parent, record.run_id)
    assert_analysis_equal(loaded.strategy, strategy)


def test_current_save_rejects_old_code_and_changed_running_source(experiment, tmp_path, monkeypatch):
    strategy, _, _ = experiment
    old = deepcopy(strategy)
    old.backtest.metadata["code_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="代码版本"):
        api().save_portfolio_run(old, tmp_path / "runs", name="旧代码")
    monkeypatch.setattr("quantlab.backtest.provenance.source_tree_sha256", lambda *args: "b" * 64)
    with pytest.raises(ValueError, match="源码已更改"):
        api().save_portfolio_run(strategy, tmp_path / "runs", name="运行中修改")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("bad", [True, 1.5, "100", None])
def test_json_quantities_are_not_coerced(experiment, tmp_path, bad):
    strategy, _, _ = experiment
    record = api().save_portfolio_run(strategy, tmp_path / "runs", name="类型")
    rewrite(record.path, lambda doc: doc["strategy"]["tables"]["trades"]["rows"][0].update(quantity=bad))
    with pytest.raises(ValueError):
        api().load_portfolio_run(record.path.parent, record.run_id)


def test_list_newest_first_and_ignores_unpublished_temp(experiment, tmp_path):
    strategy, _, _ = experiment
    archive = api()
    root = tmp_path / "runs"
    first = archive.save_portfolio_run(strategy, root, name="先保存")
    another = deepcopy(strategy)
    another.backtest.metadata["run_id"] = "00000000-0000-4000-8000-000000000001"
    second = archive.save_portfolio_run(another, root, name="后保存")
    (root / ".portfolio-interrupted.tmp").write_text("unfinished")
    scan = archive.list_portfolio_runs(root)
    assert scan.records == (second, first)
    assert scan.issues == ()


def test_atomic_publish_failure_leaves_no_partial_record(experiment, tmp_path, monkeypatch):
    strategy, _, _ = experiment
    archive = api()

    def failed(*args):
        raise OSError("simulated filesystem failure")

    monkeypatch.setattr(archive.os, "link", failed)
    root = tmp_path / "runs"
    with pytest.raises(OSError):
        archive.save_portfolio_run(strategy, root, name="不能发布")
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("payload", ['{"name":1,"name":2}', '{"value":NaN}', '{"value":Infinity}'])
def test_nonstandard_json_rejected(tmp_path, payload):
    run_id = "00000000-0000-4000-8000-000000000001"
    (tmp_path / f"{run_id}.json").write_text(payload)
    with pytest.raises(ValueError):
        api().load_portfolio_run(tmp_path, run_id)


@pytest.mark.parametrize("factor_name", ["momentum_20", "momentum_60", "momentum_120", "volatility_20", "volatility_60"])
@pytest.mark.parametrize("top_n,interval,ascending", [(1, 1, None), (2, 7, True), (1, 20, False)])
def test_all_builtin_factors_with_costs_and_rebalances_roundtrip(
    make_portfolio_preview, tmp_path, factor_name, top_n, interval, ascending,
):
    preview, _ = make_portfolio_preview(periods=125, bump_after=35)
    costs = BacktestConfig(initial_cash=35000, commission_rate=0.002, minimum_commission=5,
                           slippage_rate=0.001, sell_tax_rate=0.001, lot_size=10)
    analytics = AnalyticsConfig(periods_per_year=250, annual_risk_free_rate=0.03)
    strategy = analyze_portfolio_result(run_factor_portfolio(
        preview, factor_name=factor_name,
        config=PortfolioConfig(top_n=top_n, rebalance_interval=interval, ascending=ascending, backtest=costs),
    ), analytics_config=analytics)
    benchmark = run_equal_weight_benchmark(preview, factor_name=factor_name,
                                           backtest_config=costs, analytics_config=analytics)
    record = api().save_portfolio_run(strategy, tmp_path / "runs", name=factor_name, benchmark=benchmark)
    loaded = api().load_portfolio_run(record.path.parent, record.run_id)
    assert_analysis_equal(loaded.strategy, strategy)
    assert_analysis_equal(loaded.benchmark, benchmark)


def test_malformed_failure_context_does_not_block_valid_strategy(experiment, tmp_path):
    strategy, _, _ = experiment
    error = PortfolioBacktestError("secret", reason=["secret"], date="secret", symbols=("secret", "510300"))
    record = api().save_portfolio_run(strategy, tmp_path / "runs", name="安全上下文", benchmark_error=error)
    loaded = api().load_portfolio_run(record.path.parent, record.run_id)
    assert loaded.benchmark_failure == {"reason": "unavailable", "date": None, "symbols": ["510300"]}
    assert "secret" not in record.path.read_text()


@pytest.mark.parametrize("initial_cash", [1e10, 1e11, 1e12])
@pytest.mark.parametrize("top_n", [1, 2])
def test_large_valid_cash_ledger_can_be_archived(make_portfolio_preview, tmp_path, initial_cash, top_n):
    preview, _ = make_portfolio_preview(periods=45)
    costs = BacktestConfig(initial_cash=initial_cash, sell_tax_rate=0.001)
    strategy = analyze_portfolio_result(run_factor_portfolio(
        preview, factor_name="momentum_20",
        config=PortfolioConfig(top_n=top_n, rebalance_interval=1, backtest=costs),
    ))
    assert not strategy.backtest.trades.empty
    if top_n == 2:
        assert strategy.backtest.trades.side.eq("SELL").any()
    benchmark = run_equal_weight_benchmark(preview, factor_name="momentum_20", backtest_config=costs)
    record = api().save_portfolio_run(strategy, tmp_path / "runs", name="大额现金精度", benchmark=benchmark)
    loaded = api().load_portfolio_run(record.path.parent, record.run_id)
    assert_analysis_equal(loaded.strategy, strategy)
    assert_analysis_equal(loaded.benchmark, benchmark)


@pytest.mark.parametrize("date_position", [0, 20, 22])
def test_decision_execution_date_must_match_linked_ledgers(experiment, tmp_path, date_position):
    strategy, _, _ = experiment
    strategy = deepcopy(strategy)
    strategy.backtest.decisions.loc[0, "execution_date"] = strategy.backtest.equity_curve.date.iloc[date_position]
    with pytest.raises(ValueError):
        api().save_portfolio_run(strategy, tmp_path / "runs", name="损坏执行日期")
    assert not (tmp_path / "runs").exists()


def test_damaged_decision_date_rejected_on_load_even_with_new_checksum(experiment, tmp_path):
    strategy, _, _ = experiment
    record = api().save_portfolio_run(strategy, tmp_path / "runs", name="损坏历史日期")
    rewrite(record.path, lambda doc: doc["strategy"]["tables"]["decisions"]["rows"][0].update(
        execution_date="2025-01-02T00:00:00",
    ))
    with pytest.raises(ValueError):
        api().load_portfolio_run(record.path.parent, record.run_id)
