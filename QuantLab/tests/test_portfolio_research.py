"""真实合成研究快照进入组合回测；输入身份不能被修补或与返回值共享。"""

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from uuid import UUID

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.provenance import dataframe_sha256
from quantlab.config.settings import BacktestConfig
from quantlab.data.research_sets import SOURCE_FIELDS
from quantlab.factors import run_research_factor
from quantlab.portfolio import PortfolioBacktestError, PortfolioConfig
from quantlab.portfolio.types import (
    DECISION_DTYPES, TARGET_DTYPES, ORDER_DTYPES, TRADE_DTYPES,
)


def run(preview, *, factor_name="momentum_20", config=None, **kwargs):
    from quantlab.portfolio import run_factor_portfolio

    return run_factor_portfolio(
        preview, factor_name=factor_name,
        config=PortfolioConfig(top_n=1) if config is None else config, **kwargs,
    )


def assert_preview_unchanged(actual, before):
    pd.testing.assert_frame_equal(actual.alignment.data, before.alignment.data)
    pd.testing.assert_frame_equal(actual.alignment.coverage, before.alignment.coverage)
    pd.testing.assert_frame_equal(actual.coverage, before.coverage)
    assert actual.sources == before.sources
    assert actual.alignment.calendar == before.alignment.calendar
    assert actual.warnings == before.warnings
    assert actual.preview_sha256 == before.preview_sha256
    assert actual.alignment.content_sha256 == before.alignment.content_sha256


def assert_same_business(left, right):
    for field in ("decisions", "targets", "orders", "trades", "positions", "equity_curve",
                  "input_data", "factor_data"):
        pd.testing.assert_frame_equal(getattr(left, field), getattr(right, field))
    left_meta, right_meta = deepcopy(left.metadata), deepcopy(right.metadata)
    for key in ("run_id", "run_at"):
        left_meta.pop(key)
        right_meta.pop(key)
    assert left_meta == right_meta


def test_real_factor_runs_only_at_next_open(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=45)
    result = run(preview)
    from quantlab.portfolio import PortfolioBacktestResult

    assert isinstance(result, PortfolioBacktestResult)
    dates = pd.to_datetime(preview.alignment.calendar.dates)
    assert result.trades.iloc[0].signal_date == dates[20]
    assert result.trades.iloc[0].execution_date == dates[21]
    assert result.trades.iloc[0].symbol == "510300"
    assert result.equity_curve.iloc[:21].equity.eq(100000).all()
    assert result.metadata["schema_version"] == "portfolio-run-v1"
    assert result.metadata["status"] == "complete"
    assert result.metadata["data"]["preview_sha256"] == preview.preview_sha256
    assert not any("processed_path" in item for item in result.metadata["data"]["sources"])


@pytest.mark.parametrize("factor_name,lookback,ascending,symbol", [
    ("momentum_20", 20, False, "510300"),
    ("momentum_60", 60, False, "510300"),
    ("momentum_120", 120, False, "510300"),
    ("volatility_20", 20, True, "159915"),
    ("volatility_60", 60, True, "159915"),
])
def test_all_builtins_keep_definition_and_warmup(make_portfolio_preview, factor_name, lookback, ascending, symbol):
    preview, _ = make_portfolio_preview(periods=lookback + 2)
    result = run(preview, factor_name=factor_name)
    factor = run_research_factor(preview, factor_name)
    pd.testing.assert_frame_equal(result.factor_data, factor.data)
    assert result.metadata["factor_definition"] == asdict(factor.definition)
    strategy = result.metadata["strategy"]
    assert strategy["resolved_ascending"] is ascending
    assert strategy["lookback_periods"] == lookback
    assert strategy["signal_anchor_position"] == lookback
    assert strategy["signal_rule"] == "L+k*K"
    assert result.trades.iloc[0].execution_date.date() == preview.alignment.calendar.dates[lookback + 1]
    assert result.trades.iloc[0].symbol == symbol


@pytest.mark.parametrize("supplied", [False, True])
def test_complete_identity_and_path_free_sources(make_portfolio_preview, supplied):
    preview, root = make_portfolio_preview(periods=25, supplied=supplied)
    config = PortfolioConfig(top_n=1, rebalance_interval=3, ascending=True,
                             backtest=BacktestConfig(initial_cash=50000, lot_size=10,
                             minimum_commission=5, sell_tax_rate=0.001, slippage_rate=0.002))
    result = run(preview, config=config)
    meta = result.metadata
    assert str(UUID(meta["run_id"])) == meta["run_id"]
    assert datetime.fromisoformat(meta["run_at"]).utcoffset() is not None
    assert len(meta["code_sha256"]) == 64
    assert all(meta["environment"][name] for name in ("python", "pandas", "numpy", "pyarrow"))
    assert meta["analytics_config"] is None
    assert meta["strategy"]["name"] == "single_factor_top_n"
    assert meta["strategy"]["version"] == "1"
    assert meta["strategy"]["config"] == asdict(config)
    assert meta["strategy"]["resolved_ascending"] is True
    assert meta["strategy"]["execution_rule"] == "next_research_date_open"
    assert result.trades.iloc[0].symbol == "159915"  # 显式反向选负动量。
    data = meta["data"]
    assert data["alignment_content_sha256"] == preview.alignment.content_sha256
    assert data["market_sha256"] == dataframe_sha256(result.input_data)
    assert data["factor_sha256"] == dataframe_sha256(result.factor_data)
    assert "has_observation" in result.input_data
    assert data["symbols"] == ["159915", "510300"]
    assert data["row_count"] == 50
    assert (data["start"], data["end"]) == (preview.start.isoformat(), preview.end.isoformat())
    expected_calendar = asdict(preview.alignment.calendar)
    expected_calendar["dates"] = [day.isoformat() for day in preview.alignment.calendar.dates]
    for field in ("coverage_start", "coverage_end"):
        expected_calendar[field] = expected_calendar[field].isoformat()
    assert data["calendar"] == expected_calendar
    assert data["sources"] == [
        {key: source[key] for key in SOURCE_FIELDS if key != "processed_path"}
        for source in preview.sources
    ]
    assert str(root) not in json.dumps(meta, ensure_ascii=False, allow_nan=False)


@pytest.mark.parametrize("adjustment", ["none", "qfq", "hfq"])
def test_limitations_travel_with_result(make_portfolio_preview, adjustment):
    preview, _ = make_portfolio_preview(adjustment=adjustment)
    result = run(preview)
    warnings = result.metadata["warnings"]
    assert all(item in warnings for item in run_research_factor(preview).warnings)
    for text in ("静态", "幸存者", "252", "下一研究日期", "成交量", "分红", "实际权重"):
        assert any(text in item for item in warnings)
    assert any("总回报" in item for item in warnings) if adjustment == "none" else any(
        "模型单位" in item and "真实" in item for item in warnings
    )
    assert result.metadata["data"]["adjustment"] == adjustment


@pytest.mark.parametrize("labels", ["duplicate", "noncontiguous", "multi"])
@pytest.mark.parametrize("supplied", [False, True])
def test_row_labels_do_not_change_identity_or_mutate_input(make_portfolio_preview, labels, supplied):
    preview, _ = make_portfolio_preview(periods=25, supplied=supplied)
    expected = run(preview)
    size = len(preview.alignment.data)
    index = pd.Index([0] * size if labels == "duplicate" else np.arange(size) * 3, name="date")
    if labels == "multi":
        index = pd.MultiIndex.from_arrays([[0] * size, np.arange(size)], names=["date", "symbol"])
    preview.alignment.data.index = index
    before = deepcopy(preview)
    actual = run(preview)
    assert_same_business(actual, expected)
    assert_preview_unchanged(preview, before)


@pytest.mark.parametrize("mutation", [
    "price", "volume", "source", "extra_source", "calendar", "calendar_hash", "hash",
    "alignment_hash", "coverage", "warnings", "observation", "grid_order",
])
def test_mutated_preview_is_rejected_not_repaired(make_portfolio_preview, mutation):
    preview, _ = make_portfolio_preview(periods=25, supplied=True)
    if mutation == "price":
        preview.alignment.data.loc[0, ["open", "high", "low", "close"]] *= 1.01
    elif mutation == "volume":
        preview.alignment.data.loc[0, "volume"] += 1
    elif mutation == "source":
        preview.sources[0]["version_id"] = "changed-version"
    elif mutation == "extra_source":
        preview.sources[0]["secret"] = "synthetic-only"
    elif mutation in ("calendar", "calendar_hash"):
        axis = preview.alignment.calendar
        axis = replace(axis, **({"source": "changed"} if mutation == "calendar" else {"file_sha256": "0" * 64}))
        preview = replace(preview, alignment=replace(preview.alignment, calendar=axis))
    elif mutation == "hash":
        preview = replace(preview, preview_sha256="0" * 64)
    elif mutation == "alignment_hash":
        preview = replace(preview, alignment=replace(preview.alignment, content_sha256="0" * 64))
    elif mutation == "coverage":
        preview.coverage.loc[0, "missing_count"] = 999
    elif mutation == "warnings":
        preview = replace(preview, warnings=())
    elif mutation == "observation":
        preview.alignment.data.loc[0, "has_observation"] = False
    else:
        preview = replace(preview, alignment=replace(preview.alignment, data=preview.alignment.data.iloc[::-1]))
    before = deepcopy(preview)
    with pytest.raises(PortfolioBacktestError, match="研究") as caught:
        run(preview)
    assert caught.value.reason == "invalid_research_input"
    assert_preview_unchanged(preview, before)


@pytest.mark.parametrize("name", ["unknown", " momentum_20", "", None, []])
def test_invalid_factor_name_has_context(make_portfolio_preview, name):
    preview, _ = make_portfolio_preview(periods=3)
    with pytest.raises(PortfolioBacktestError, match="因子"):
        run(preview, factor_name=name)


@pytest.mark.parametrize("which", ["preview", "config", "too_many"])
def test_invalid_entry_types_or_asset_count(make_portfolio_preview, which):
    preview, _ = make_portfolio_preview(periods=3)
    with pytest.raises(PortfolioBacktestError):
        run(None if which == "preview" else preview,
            config={} if which == "config" else PortfolioConfig(top_n=3 if which == "too_many" else 1))


def test_external_factor_injection_is_not_public(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=3)
    with pytest.raises(TypeError, match="registry"):
        run(preview, registry={})
    with pytest.raises(TypeError, match="factor_data"):
        run(preview, factor_data=pd.DataFrame())


def test_allowed_numeric_config_is_json_serializable(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=25)
    config = PortfolioConfig(top_n=np.int64(1), rebalance_interval=np.int64(20),
                             backtest=BacktestConfig(initial_cash=np.float64(100000),
                             lot_size=np.int64(100), commission_rate=np.float64(0.0003)))
    result = run(preview, config=config)
    restored = json.loads(json.dumps(result.metadata, allow_nan=False))
    assert restored["strategy"]["config"]["backtest"]["lot_size"] == 100
    assert restored["strategy"]["config"]["backtest"]["initial_cash"] == 100000
    assert result.trades.iloc[0].quantity == 900


def test_mutated_frozen_config_is_revalidated(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=25)
    config = PortfolioConfig(top_n=1, backtest=BacktestConfig())
    object.__setattr__(config.backtest, "slippage_rate", float("nan"))
    with pytest.raises(PortfolioBacktestError) as caught:
        run(preview, config=config)
    assert caught.value.reason == "invalid_config"


def test_snapshots_and_outputs_do_not_share_input_or_other_runs(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=25)
    before = deepcopy(preview)
    first, second = run(preview), run(preview)
    assert first.metadata["run_id"] != second.metadata["run_id"]
    assert_same_business(first, second)
    for field in ("decisions", "targets", "orders", "trades", "positions", "equity_curve",
                  "input_data", "factor_data"):
        getattr(first, field).drop(index=getattr(first, field).index, inplace=True)
    first.metadata["data"]["sources"][0]["provider"] = "output-only"
    first.metadata["data"]["calendar"]["dates"].clear()
    first.metadata["strategy"]["config"]["backtest"]["initial_cash"] = 1
    first.metadata["warnings"].append("output-only")
    assert_preview_unchanged(preview, before)
    assert_same_business(second, run(preview))


def test_loaded_snapshot_runs_without_source_files_or_writes(make_portfolio_preview, tmp_path):
    preview, root = make_portfolio_preview(periods=25)
    root.rename(root.with_name(root.name + "_moved"))
    before = {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")}
    result = run(preview)
    assert not result.trades.empty
    assert before == {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")}


def test_all_cash_result_keeps_identity_and_empty_schemas(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=10)
    result = run(preview)
    assert result.metadata["status"] == "complete"
    assert result.equity_curve.equity.eq(100000).all()
    assert len(result.input_data) == len(result.factor_data) == 20
    for name, schema in (("decisions", DECISION_DTYPES), ("targets", TARGET_DTYPES),
                         ("orders", ORDER_DTYPES), ("trades", TRADE_DTYPES)):
        frame = getattr(result, name)
        assert frame.empty
        assert {key: str(value) for key, value in frame.dtypes.items()} == schema
    assert any("热身" in item for item in result.metadata["warnings"])


def test_execution_failure_keeps_structured_context_and_input(make_portfolio_preview):
    preview, _ = make_portfolio_preview(periods=25, zero_volume={"510300": [21]})
    before = deepcopy(preview)
    with pytest.raises(PortfolioBacktestError) as caught:
        run(preview)
    assert caught.value.reason == "zero_volume"
    assert caught.value.date == preview.alignment.calendar.dates[21]
    assert caught.value.symbols == ("510300",)
    assert_preview_unchanged(preview, before)


def test_portfolio_rejects_disk_code_changed_after_load(make_portfolio_preview, tmp_path):
    preview, _ = make_portfolio_preview(periods=25)
    sandbox = tmp_path / "code_copy"
    sandbox.mkdir()
    shutil.copytree(Path("src/quantlab"), sandbox / "quantlab", ignore=shutil.ignore_patterns("__pycache__"))
    (sandbox / "preview.pkl").write_bytes(pickle.dumps(preview))
    script = '''
import importlib
from pathlib import Path
import pickle
from quantlab.portfolio import run_factor_portfolio, PortfolioConfig, PortfolioBacktestError
from quantlab.backtest import provenance
preview = pickle.loads(Path("preview.pkl").read_bytes())
run_factor_portfolio(preview, factor_name="momentum_20", config=PortfolioConfig(top_n=1))
source = Path("quantlab/portfolio/engine.py")
source.write_text(source.read_text() + "\\n# changed after import\\n")
importlib.reload(provenance)
try:
    run_factor_portfolio(preview, factor_name="momentum_20", config=PortfolioConfig(top_n=1))
except PortfolioBacktestError as error:
    assert "重新启动" in str(error)
else:
    raise AssertionError("Changed disk code was incorrectly claimed as the loaded execution version")
'''
    process = subprocess.run([sys.executable, "-c", script], cwd=sandbox,
                             env={**os.environ, "PYTHONPATH": str(sandbox)},
                             capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr
