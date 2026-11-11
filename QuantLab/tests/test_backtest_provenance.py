from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd

from quantlab.backtest import run_backtest, run_buy_and_hold, run_dual_moving_average_backtest
from quantlab.comparison import compare_three_approaches
from quantlab.config.settings import AnalyticsConfig, BacktestConfig
from quantlab.data import load_trade_csv
from quantlab.data.market_loader import load_market_csv


def market_data() -> pd.DataFrame:
    return load_market_csv("tests/fixtures/valid_etf.csv")


def test_dual_moving_average_records_inputs_parameters_and_execution_identity() -> None:
    before = datetime.now(timezone.utc)
    result = run_dual_moving_average_backtest(
        market_data(), 1, 2,
        backtest_config=BacktestConfig(initial_cash=12345, commission_rate=0.001),
        analytics_config=AnalyticsConfig(periods_per_year=240, annual_risk_free_rate=0.02),
    )
    metadata = getattr(result.backtest, "metadata", None)
    assert metadata is not None, "每次回测必须携带可复现的运行记录"
    assert metadata.strategy == {
        "name": "dual_moving_average", "version": "1", "parameters": {"short_window": 1, "long_window": 2},
    }
    assert metadata.backtest_config["initial_cash"] == 12345
    assert metadata.backtest_config["commission_rate"] == 0.001
    assert metadata.analytics_config == {"periods_per_year": 240, "annual_risk_free_rate": 0.02}
    assert before <= datetime.fromisoformat(metadata.run_at) <= datetime.now(timezone.utc)
    assert metadata.data["source"] is None  # 不把内存数据伪装成已归档文件。
    assert metadata.data["version"] == "sha256:" + metadata.data["market_sha256"]
    assert metadata.data["symbols"] == ["510300"]
    assert metadata.data["row_count"] == 5
    assert len(metadata.code_sha256) == 64
    assert metadata.environment["python"]
    assert metadata.environment["pandas"] == pd.__version__


def test_declared_source_and_version_are_preserved_without_losing_content_hash() -> None:
    result = run_dual_moving_average_backtest(
        market_data(), 1, 2,
        data_source="data/processed/market/510300/example.parquet",
        data_version="import-20260922",
    )
    assert result.backtest.metadata.data["source"] == "data/processed/market/510300/example.parquet"
    assert result.backtest.metadata.data["version"] == "import-20260922"
    assert len(result.backtest.metadata.data["market_sha256"]) == 64


def test_raw_signal_input_is_snapshotted_and_signal_change_changes_input_hash() -> None:
    data = market_data().assign(signal=["BUY", None, None, "SELL", None])
    original = run_backtest(data)
    metadata = getattr(original, "metadata", None)
    assert metadata is not None
    first_open = float(data.iloc[0]["open"])
    data.loc[0, "signal"] = None
    changed_signal = run_backtest(data)
    assert original.input_data.loc[0, "signal"] == "BUY"
    assert metadata.data["market_sha256"] == changed_signal.metadata.data["market_sha256"]
    assert metadata.data["input_sha256"] != changed_signal.metadata.data["input_sha256"]
    data.loc[0, "open"] = first_open + 0.001
    changed_price = run_backtest(data)
    assert original.input_data.iloc[0]["open"] == first_open
    assert metadata.data["market_sha256"] != changed_price.metadata.data["market_sha256"]


def test_fingerprint_is_independent_of_input_order_but_runs_have_unique_ids() -> None:
    data = market_data().assign(signal=None)
    first = run_backtest(data)
    second = run_backtest(data.iloc[::-1])
    metadata = getattr(first, "metadata", None)
    assert metadata is not None
    assert metadata.data["input_sha256"] == second.metadata.data["input_sha256"]
    assert metadata.run_id != second.metadata.run_id
    assert metadata.code_sha256 == second.metadata.code_sha256


def test_buy_and_hold_and_comparison_keep_both_simulations_auditable() -> None:
    data = market_data()
    benchmark = run_buy_and_hold(data)
    metadata = getattr(benchmark.backtest, "metadata", None)
    assert metadata is not None
    assert metadata.strategy["name"] == "buy_and_hold"
    assert metadata.execution_rule == "first_day_open"
    comparison = compare_three_approaches(
        data, load_trade_csv("tests/fixtures/valid_trades.csv"), 1, 2,
        data_source="example.parquet", data_version="v1",
    )
    hold = comparison.buy_and_hold.backtest.metadata
    strategy = comparison.strategy.backtest.metadata
    assert hold.run_id != strategy.run_id
    assert hold.data["market_sha256"] == strategy.data["market_sha256"]
    assert hold.data["version"] == strategy.data["version"] == "v1"
    assert strategy.execution_rule == "next_day_open"


def test_loaded_code_cannot_claim_a_changed_disk_version(tmp_path: Path) -> None:
    # 在隔离的短进程中模拟 Notebook/服务一直运行、源码却已经修改的情况。
    shutil.copytree(
        Path("src/quantlab"), tmp_path / "quantlab",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    script = '''
import importlib
from pathlib import Path
import pandas as pd
from quantlab.backtest import run_backtest
from quantlab.backtest import provenance

data = pd.DataFrame({
    "date": ["2024-01-02", "2024-01-03"], "symbol": ["510300"] * 2,
    "open": [3.0, 3.1], "high": [3.0, 3.1], "low": [3.0, 3.1],
    "close": [3.0, 3.1], "volume": [1000, 1000], "signal": [None, None],
})
run_backtest(data)
source = Path(provenance.__file__).with_name("engine.py")
source.write_text(source.read_text() + "\\n# changed after import\\n")
# 只重载记录模块也不能把已经加载的旧引擎冒充为新版本。
importlib.reload(provenance)
try:
    run_backtest(data)
except ValueError as error:
    assert "重新启动" in str(error)
else:
    raise AssertionError("源码改变后仍记录了新指纹，但实际执行的是旧代码")
'''
    process = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True, text=True, timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr
