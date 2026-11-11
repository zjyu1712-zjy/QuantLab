import json
from pathlib import Path

import pandas as pd
import pytest

import quantlab.backtest as backtest_api
from quantlab.backtest.provenance import source_tree_sha256
from quantlab.config.settings import AnalyticsConfig, BacktestConfig
from quantlab.data.market_loader import load_market_csv


def run_example():
    return backtest_api.run_dual_moving_average_backtest(
        load_market_csv("tests/fixtures/valid_etf.csv"), 1, 2,
        backtest_config=BacktestConfig(initial_cash=10000, commission_rate=0.001, lot_size=10),
        analytics_config=AnalyticsConfig(periods_per_year=240),
        data_source="tests/fixtures/valid_etf.csv", data_version="fixture-v1",
    )


def test_saved_record_can_reproduce_orders_and_equity_without_original_file(tmp_path: Path) -> None:
    result = run_example()
    save = getattr(backtest_api, "save_backtest_run", None)
    assert callable(save), "回测记录需要可持久保存"
    path = save(result.backtest, tmp_path, performance=result.performance)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metadata"]["data"]["version"] == "fixture-v1"
    assert payload["metadata"]["analytics_config"]["periods_per_year"] == 240
    assert payload["metrics"]["cumulative_return"] == result.performance.cumulative_return
    assert len(payload["orders"]) == len(result.backtest.orders)
    assert len(payload["trades"]) == len(result.backtest.trades)
    replay = backtest_api.run_backtest(
        pd.DataFrame(payload["input_data"]),
        BacktestConfig(**payload["metadata"]["backtest_config"]),
    )
    pd.testing.assert_frame_equal(replay.equity_curve, result.backtest.equity_curve)
    pd.testing.assert_frame_equal(replay.trades, result.backtest.trades)


def test_archive_does_not_overwrite_existing_run(tmp_path: Path) -> None:
    save = getattr(backtest_api, "save_backtest_run", None)
    assert callable(save)
    result = run_example()
    path = save(result.backtest, tmp_path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save(result.backtest, tmp_path)
    assert path.read_bytes() == original


def test_export_never_includes_environment_secrets(monkeypatch) -> None:
    export = getattr(backtest_api, "serialize_backtest_run", None)
    assert callable(export)
    monkeypatch.setenv("IFIND_ACCESS_TOKEN", "private-value-must-not-be-recorded")
    result = run_example()
    serialized = export(result.backtest, performance=result.performance)
    assert "private-value-must-not-be-recorded" not in serialized
    payload = json.loads(serialized)
    assert set(payload["metadata"]["environment"]) == {"python", "pandas", "numpy", "pyarrow"}


def test_modified_input_snapshot_cannot_be_exported_as_original_run(tmp_path: Path) -> None:
    save = getattr(backtest_api, "save_backtest_run", None)
    assert callable(save)
    result = run_example()
    result.backtest.input_data.loc[0, "close"] += 0.001
    with pytest.raises(ValueError, match="指纹"):
        save(result.backtest, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_code_fingerprint_changes_with_uncommitted_source_but_ignores_credentials(tmp_path: Path) -> None:
    source = tmp_path / "engine.py"
    source.write_text("version = 1\n", encoding="utf-8")
    original = source_tree_sha256(tmp_path)
    (tmp_path / ".env").write_text("SECRET=private\n", encoding="utf-8")
    assert source_tree_sha256(tmp_path) == original
    source.write_text("version = 2\n", encoding="utf-8")
    assert source_tree_sha256(tmp_path) != original


def test_exported_metrics_require_their_calculation_configuration() -> None:
    raw = backtest_api.run_backtest(
        load_market_csv("tests/fixtures/valid_etf.csv").assign(
            signal=["BUY", None, None, "SELL", None],
        ),
    )
    analysis = backtest_api.analyze_backtest_result(
        raw, analytics_config=AnalyticsConfig(periods_per_year=240, annual_risk_free_rate=0.03),
    )
    with pytest.raises(ValueError, match="analysis.backtest"):
        backtest_api.serialize_backtest_run(raw, performance=analysis.performance)
    payload = json.loads(backtest_api.serialize_backtest_run(
        analysis.backtest, performance=analysis.performance,
    ))
    assert payload["metadata"]["analytics_config"]["annual_risk_free_rate"] == 0.03


def test_export_rejects_metrics_from_a_different_analysis() -> None:
    result = run_example()
    other_analysis = backtest_api.analyze_backtest_result(
        result.backtest,
        analytics_config=AnalyticsConfig(periods_per_year=12, annual_risk_free_rate=0.03),
    )
    with pytest.raises(ValueError, match="绩效指标.*不一致"):
        backtest_api.serialize_backtest_run(
            result.backtest, performance=other_analysis.performance,
        )
