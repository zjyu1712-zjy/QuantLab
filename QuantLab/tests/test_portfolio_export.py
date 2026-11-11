"""实验包由真实存档生成；检查备份保真、精度、安全与不重算边界。"""

import csv
from dataclasses import replace
from hashlib import sha256
from io import BytesIO, StringIO
import json
from zipfile import ZipFile

import pandas as pd
import pytest

import quantlab.portfolio as portfolio
from quantlab.config.settings import AnalyticsConfig, BacktestConfig


@pytest.fixture
def saved_experiment(tmp_path, make_portfolio_preview):
    def create(*, periods=45, baseline=True, failure=False, gaps=None):
        preview, source = make_portfolio_preview(periods=periods, gaps=gaps)
        costs = BacktestConfig(initial_cash=123456.78901234567, commission_rate=.00031)
        analytics = AnalyticsConfig(periods_per_year=365, annual_risk_free_rate=.023456789012345)
        strategy = portfolio.analyze_portfolio_result(portfolio.run_factor_portfolio(
            preview, factor_name="momentum_20", config=portfolio.PortfolioConfig(top_n=1, backtest=costs),
        ), analytics_config=analytics)
        benchmark = portfolio.run_equal_weight_benchmark(
            preview, factor_name="momentum_20", backtest_config=costs, analytics_config=analytics,
        ) if baseline else None
        error = portfolio.PortfolioBacktestError("secret-token", reason="missing_close", symbols=("510300",)) if failure else None
        record = portfolio.save_portfolio_run(strategy, tmp_path / "runs", name="中文实验", benchmark=benchmark,
                                              benchmark_error=error)
        return record, strategy, benchmark, source
    return create


def export(loaded):
    build = getattr(portfolio, "build_portfolio_export", None)
    assert callable(build), "缺少只读的组合实验导出入口"
    return build(loaded)


def read_csv(bundle, name):
    return list(csv.DictReader(StringIO(bundle.read(name).decode("utf-8-sig"))))


def rewrite(path, mutate):
    payload = json.loads(path.read_bytes())
    mutate(payload)
    core = {key: value for key, value in payload.items() if key != "content_sha256"}
    payload["content_sha256"] = sha256(json.dumps(
        core, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    path.write_bytes(("\n  " + json.dumps(payload, ensure_ascii=False, indent=4, allow_nan=False) + "\n\n").encode())


def test_zip_preserves_original_json_and_stored_numeric_values(saved_experiment):
    record, strategy, benchmark, _ = saved_experiment()
    rewrite(record.path, lambda doc: doc.update(name="备份，中文\n多行名称"))
    before = record.path.read_bytes()
    loaded = portfolio.load_portfolio_run(record.path.parent, record.run_id)
    result = export(loaded)
    assert result.file_name == f"quantlab-portfolio-{record.run_id}.zip"
    with ZipFile(BytesIO(result.content)) as bundle:
        assert bundle.testzip() is None
        assert bundle.read("archive.json") == before
        assert not any(name.startswith("/") or ".." in name.split("/") for name in bundle.namelist())
        for prefix, analysis in (("strategy", strategy), ("benchmark", benchmark)):
            rows = read_csv(bundle, f"{prefix}/equity_curve.csv")
            assert len(rows) == len(analysis.backtest.equity_curve)
            for field in ("cash", "equity", "position_value", "net_value", "cumulative_return"):
                assert [float(row[field]) for row in rows] == analysis.backtest.equity_curve[field].tolist()
            metrics = {row["metric"]: row for row in read_csv(bundle, f"{prefix}/metrics.csv")}
            for key in ("cumulative_return", "annualized_return", "annualized_volatility", "max_drawdown", "sharpe_ratio"):
                assert float(metrics[key]["value"]) == getattr(analysis.performance, key)
            assert metrics["cumulative_return"]["unit"] == "ratio"
            performance = read_csv(bundle, f"{prefix}/performance.csv")
            assert [float(row["drawdown"]) for row in performance] == analysis.performance.drawdown.tolist()
            assert performance[0]["daily_return"] == "", "首日没有前一期，不能补零收益"
            assert [float(row["daily_return"]) for row in performance[1:]] == analysis.performance.daily_returns.tolist()
            assert [row["date"] for row in rows] == [day.date().isoformat() for day in analysis.backtest.equity_curve.date]
        assert "strategy/factor_data.csv" in bundle.namelist()
        assert "benchmark/factor_data.csv" not in bundle.namelist()
        assert "Excel" in bundle.read("README.txt").decode("utf-8")
        meta = json.loads(bundle.read("metadata.json"))
        assert meta["strategy"] == json.loads(before)["strategy"]["metadata"]
        assert meta["benchmark_status"] == "complete"
    assert record.path.read_bytes() == before
    assert export(loaded).content == result.content, "同一快照再次导出不应混入当前时间"


def test_export_never_reopens_sources_or_recomputes_and_ignores_mutable_views(saved_experiment, monkeypatch):
    record, strategy, _, source = saved_experiment()
    loaded = portfolio.load_portfolio_run(record.path.parent, record.run_id)
    before = record.path.read_bytes()
    source.rename(source.with_name("moved_sources"))
    record.path.write_text("file replaced after loading")
    loaded.strategy.backtest.equity_curve.loc[:, "equity"] = 1.0
    loaded.strategy.backtest.metadata["run_id"] = "not-an-id"
    loaded.benchmark.backtest.metadata["warnings"].append("mutated-display-only")
    def forbidden(*args, **kwargs):
        pytest.fail("导出不得重新打开文件、运行回测或计算指标")
    monkeypatch.setattr("quantlab.portfolio.archive.os.open", forbidden)
    monkeypatch.setattr("quantlab.portfolio.analysis._performance_from_equity", forbidden)
    monkeypatch.setattr("quantlab.portfolio.research.run_factor_portfolio", forbidden)
    monkeypatch.setattr("quantlab.portfolio.benchmark.run_equal_weight_benchmark", forbidden)
    monkeypatch.setattr("quantlab.backtest.provenance.source_tree_sha256", forbidden)
    result = export(loaded)
    with ZipFile(BytesIO(result.content)) as bundle:
        assert bundle.read("archive.json") == before
        assert float(read_csv(bundle, "strategy/equity_curve.csv")[-1]["equity"]) == strategy.backtest.equity_curve.equity.iloc[-1]
        assert "mutated-display-only" not in bundle.read("metadata.json").decode()
    assert record.path.read_text() == "file replaced after loading"


@pytest.mark.parametrize("baseline,failure,status", [(True, False, "complete"), (False, False, "not_run"), (False, True, "unavailable")])
def test_benchmark_states_are_explicit_not_fake_empty_success(saved_experiment, baseline, failure, status):
    record, _, _, _ = saved_experiment(baseline=baseline, failure=failure)
    result = export(portfolio.load_portfolio_run(record.path.parent, record.run_id))
    with ZipFile(BytesIO(result.content)) as bundle:
        assert any(name.startswith("benchmark/") for name in bundle.namelist()) == baseline
        meta = json.loads(bundle.read("metadata.json"))
        assert meta["benchmark_status"] == status
        assert (meta["benchmark_failure"] is not None) == failure
        assert status in bundle.read("README.txt").decode()
        assert b"secret-token" not in bundle.read("archive.json")


@pytest.mark.parametrize("periods", [1, 20, 21])
def test_empty_tables_keep_headers_and_undefined_metrics_keep_reasons(saved_experiment, periods):
    record, strategy, _, _ = saved_experiment(periods=periods)
    result = export(portfolio.load_portfolio_run(record.path.parent, record.run_id))
    with ZipFile(BytesIO(result.content)) as bundle:
        for prefix in ("strategy", "benchmark"):
            assert not read_csv(bundle, f"{prefix}/trades.csv")
            header = next(csv.reader(StringIO(bundle.read(f"{prefix}/trades.csv").decode("utf-8-sig"))))
            assert header == strategy.backtest.trades.columns.tolist()
            metrics = {row["metric"]: row for row in read_csv(bundle, f"{prefix}/metrics.csv")}
            assert metrics["sharpe_ratio"]["value"] == ""
            assert metrics["sharpe_ratio"]["unavailable_reason"]
            assert metrics["max_drawdown"]["value"] in ("0.0", "0")
        if periods == 21:
            assert read_csv(bundle, "strategy/decisions.csv")[0]["execution_date"] == ""


def test_missing_prices_and_warmup_values_are_empty_not_zero(saved_experiment):
    record, _, _, _ = saved_experiment(gaps={"510300": [0, 3]}, baseline=False)
    result = export(portfolio.load_portfolio_run(record.path.parent, record.run_id))
    with ZipFile(BytesIO(result.content)) as bundle:
        market = read_csv(bundle, "strategy/input_data.csv")
        assert any(row["symbol"] == "510300" and row["close"] == "" for row in market)
        factor = read_csv(bundle, "strategy/factor_data.csv")
        assert factor[0]["factor_value"] == ""
        for name in bundle.namelist():
            if name.endswith(".csv"):
                assert not any(cell in ("nan", "NaT", "None") for row in read_csv(bundle, name) for cell in row.values())


def test_old_versions_and_pathlike_names_are_preserved_without_unsafe_zip_paths(saved_experiment):
    record, _, _, _ = saved_experiment(baseline=False)
    def old(doc):
        doc["name"] = "../../private/测试"
        doc["strategy"]["metadata"]["code_sha256"] = "a" * 64
    rewrite(record.path, old)
    before = record.path.read_bytes()
    loaded = portfolio.load_portfolio_run(record.path.parent, record.run_id)
    assert loaded.warnings
    result = export(loaded)
    with ZipFile(BytesIO(result.content)) as bundle:
        assert bundle.read("archive.json") == before
        assert json.loads(bundle.read("metadata.json"))["strategy"]["code_sha256"] == "a" * 64
        assert not any("private" in name or ".." in name or name.startswith("/") for name in bundle.namelist())
        assert str(record.path.parent) not in bundle.read("metadata.json").decode()


@pytest.mark.parametrize("dangerous", ["=1+2", "+1+2", "-1+2", "@SUM(A1)", " \t=1+2", "\tplain", "\nplain"])
def test_csv_formula_text_is_escaped_but_json_and_negative_numbers_are_not(saved_experiment, dangerous):
    record, _, _, _ = saved_experiment(periods=1, baseline=False)
    def mutate(doc):
        doc["strategy"]["performance"]["unavailable_reasons"]["sharpe_ratio"] = dangerous
        doc["strategy"]["performance"]["metrics"]["cumulative_return"] = -0.12345678901234567
        doc["strategy"]["performance"]["unavailable_reasons"].pop("cumulative_return", None)
    rewrite(record.path, mutate)
    # 历史指标不按当前公式重算；本例用于验证已存数值及文本的导出，不证明旧公式正确。
    result = export(portfolio.load_portfolio_run(record.path.parent, record.run_id))
    with ZipFile(BytesIO(result.content)) as bundle:
        metrics = {row["metric"]: row for row in read_csv(bundle, "strategy/metrics.csv")}
        assert metrics["sharpe_ratio"]["unavailable_reason"] == "'" + dangerous
        assert float(metrics["cumulative_return"]["value"]) == -0.12345678901234567
        assert json.loads(bundle.read("archive.json"))["strategy"]["performance"]["unavailable_reasons"]["sharpe_ratio"] == dangerous


@pytest.mark.parametrize("content", [b"", b"{}", b"not-json"])
def test_invalid_raw_snapshot_is_not_exported(saved_experiment, content):
    record, _, _, _ = saved_experiment()
    loaded = portfolio.load_portfolio_run(record.path.parent, record.run_id)
    assert hasattr(loaded, "archive_bytes"), "读取结果必须保留同次校验的原始字节"
    with pytest.raises(ValueError):
        export(replace(loaded, archive_bytes=content))


def test_raw_snapshot_tampering_is_rejected_even_if_display_is_valid(saved_experiment):
    record, _, _, _ = saved_experiment()
    loaded = portfolio.load_portfolio_run(record.path.parent, record.run_id)
    assert hasattr(loaded, "archive_bytes")
    doc = json.loads(loaded.archive_bytes)
    doc["name"] = "篡改后没有更新指纹"
    with pytest.raises(ValueError):
        export(replace(loaded, archive_bytes=json.dumps(doc).encode()))


def test_loader_retains_the_exact_bytes_it_validated(saved_experiment, monkeypatch):
    from quantlab.portfolio import archive
    record, _, _, _ = saved_experiment()
    before = record.path.read_bytes()
    original_decode = archive.decode_analysis
    def replace_file_after_read(*args, **kwargs):
        record.path.write_text("changed during validation")
        return original_decode(*args, **kwargs)
    monkeypatch.setattr(archive, "decode_analysis", replace_file_after_read)
    loaded = archive.load_portfolio_run(record.path.parent, record.run_id)
    assert getattr(loaded, "archive_bytes", None) == before
    assert record.path.read_text() == "changed during validation"


def test_csv_conversion_preserves_text_codes_integer_shares_and_small_floats():
    from quantlab.portfolio.export import _csv_bytes
    # 测转换边界，防止统一格式化把代码、整型份额或很小的浮点值改掉。
    frame = pd.DataFrame({"symbol": ["000001", "510050"], "quantity": [9007199254740993, 0],
                          "value": [1.2345678901234567, 1e-300]})
    rows = list(csv.DictReader(StringIO(_csv_bytes(frame).decode("utf-8-sig"))))
    assert [row["symbol"] for row in rows] == ["000001", "510050"]
    assert [row["quantity"] for row in rows] == ["9007199254740993", "0"]
    assert [float(row["value"]) for row in rows] == [1.2345678901234567, 1e-300]


def test_export_rejects_snapshot_with_another_records_identity(saved_experiment):
    first, _, _, _ = saved_experiment()
    second, _, _, _ = saved_experiment()
    loaded = portfolio.load_portfolio_run(first.path.parent, first.run_id)
    with pytest.raises(ValueError):
        export(replace(loaded, record=second))
