"""组合存档界面的真实往返、历史独立性及会话边界。"""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from quantlab.config.settings import AnalyticsConfig, BacktestConfig
from quantlab.data.research_storage import save_research_set
from quantlab.portfolio import (
    PortfolioBacktestError, PortfolioConfig, analyze_portfolio_result, run_factor_portfolio,
    run_equal_weight_benchmark, save_portfolio_run, load_portfolio_run,
)
from quantlab.ui import runtime


PAGE = "pages/7_因子研究.py"


@pytest.fixture
def workspace(tmp_path, monkeypatch, make_portfolio_preview):
    sets = tmp_path / "research_sets"
    archives = tmp_path / "portfolio_runs"
    monkeypatch.delenv("QUANTLAB_PUBLIC_MODE", raising=False)
    monkeypatch.delenv("QUANTLAB_ACCESS_PASSWORD", raising=False)
    monkeypatch.setattr(runtime, "get_research_storage_dir", lambda project: sets)

    def create(*, with_set=True, periods=45, cash=100000.0, analytics=None, benchmark=True):
        preview, source = make_portfolio_preview(periods=periods)
        saved = save_research_set(preview, sets, name="合成研究") if with_set else None
        costs = BacktestConfig(initial_cash=cash)
        analysis = analyze_portfolio_result(run_factor_portfolio(
            preview, factor_name="momentum_20", config=PortfolioConfig(top_n=1, backtest=costs),
        ), analytics_config=analytics or AnalyticsConfig())
        baseline = run_equal_weight_benchmark(preview, factor_name="momentum_20", backtest_config=costs,
                                              analytics_config=analytics or AnalyticsConfig()) if benchmark else None
        return saved, analysis, baseline, source

    return sets, archives, create


def page():
    result = AppTest.from_file(PAGE, default_timeout=25).run()
    assert not result.exception
    return result


def run_live(app):
    app.button(key="portfolio_run").click().run()
    assert not app.exception and not app.error
    return app.session_state["portfolio_analysis"]


def open_history(app, run_id):
    assert any(widget.key == "portfolio_history_id" for widget in app.selectbox), "缺少独立历史入口"
    app.selectbox(key="portfolio_history_id").set_value(run_id).run()
    assert not app.exception


def test_explicit_save_roundtrip_and_duplicate_do_not_overwrite(workspace):
    _, archives, create = workspace
    create()
    app = page()
    assert not archives.exists()
    assert not any(button.key == "portfolio_archive_save" for button in app.button)
    live = run_live(app)
    assert not archives.exists(), "运行不能自动保存"
    assert any(button.key == "portfolio_archive_save" for button in app.button), "成功结果应提供保存按钮"
    app.text_input(key="portfolio_archive_name").set_value("我的动量实验")
    app.button(key="portfolio_archive_save").click().run()
    assert not app.exception and not app.error
    run_id = live.backtest.metadata["run_id"]
    record = load_portfolio_run(archives, run_id)
    assert record.record.name == "我的动量实验"
    assert record.benchmark_status == "complete"
    pd.testing.assert_frame_equal(record.strategy.backtest.equity_curve, live.backtest.equity_curve, check_exact=True)
    original = record.record.path.read_bytes()
    app.text_input(key="portfolio_archive_name").set_value("不能改名覆盖")
    app.button(key="portfolio_archive_save").click().run()
    assert not app.exception
    assert record.record.path.read_bytes() == original
    assert len(list(archives.glob("*.json"))) == 1
    assert any("未覆盖" in item.value for item in app.info)
    assert app.session_state["portfolio_analysis"].backtest.metadata["run_id"] == run_id
    open_history(app, run_id)
    assert len(app.get("vega_lite_chart")) == 6
    assert not app.error, "同一结果在当前和历史同时展示也不能出现重复组件"


@pytest.mark.parametrize("problem", ["no_sets", "scan_error", "corrupt_set", "load_error"])
def test_history_survives_missing_or_failed_research(workspace, monkeypatch, problem):
    import quantlab.data.research_storage as storage

    sets, archives, create = workspace
    saved, analysis, baseline, source = create(with_set=problem != "no_sets")
    record = save_portfolio_run(analysis, archives, name="独立历史", benchmark=baseline)
    source.rename(source.with_name(source.name + "_moved"))
    if problem == "corrupt_set":
        (saved.directory / "manifest.json").write_text("{}")
    elif problem in ("scan_error", "load_error"):
        def failed(*args, **kwargs):
            raise OSError("token=DO_NOT_DISPLAY /Users/private")
        monkeypatch.setattr(storage, "list_research_sets" if problem == "scan_error" else "load_research_set", failed)
    app = page()
    open_history(app, record.run_id)
    assert "portfolio_analysis" not in app.session_state
    assert "portfolio_benchmark" not in app.session_state
    assert any("历史实验结果" in item.value for item in app.subheader)
    assert any(metric.label == "累计收益率" and metric.value == f"{analysis.performance.cumulative_return:.2%}" for metric in app.metric)
    assert not any(button.key == "portfolio_archive_save" for button in app.button)
    assert "DO_NOT_DISPLAY" not in str(app.error)


def test_history_uses_own_benchmark_and_cannot_replace_live_result(workspace, monkeypatch):
    from quantlab.ui import portfolio

    _, archives, create = workspace
    _, historical, baseline, _ = create(cash=200000)
    record = save_portfolio_run(historical, archives, name="旧实验", benchmark=baseline)
    app = page()
    live = run_live(app)
    live_id = live.backtest.metadata["run_id"]
    live_baseline_id = app.session_state["portfolio_benchmark"].backtest.metadata["run_id"]

    def forbidden(*args, **kwargs):
        pytest.fail("查看历史或保存不应重新计算")

    monkeypatch.setattr(portfolio, "run_factor_portfolio", forbidden)
    monkeypatch.setattr(portfolio, "run_equal_weight_benchmark", forbidden)
    monkeypatch.setattr(portfolio, "analyze_portfolio_result", forbidden)
    open_history(app, record.run_id)
    assert not app.error
    assert app.session_state["portfolio_analysis"].backtest.metadata["run_id"] == live_id
    assert app.session_state["portfolio_benchmark"].backtest.metadata["run_id"] == live_baseline_id
    # 两份结果同时存在，图表须有独立身份，不触发 Streamlit 重复元素错误。
    assert len(app.get("vega_lite_chart")) == 6
    values = [item.value for item in app.metric if item.label == "期末总资产（元）"]
    assert values == [f"{live.backtest.equity_curve.equity.iloc[-1]:,.2f}",
                      f"{historical.backtest.equity_curve.equity.iloc[-1]:,.2f}"]
    app.button(key="portfolio_archive_save").click().run()
    assert load_portfolio_run(archives, live_id).strategy.backtest.metadata["run_id"] == live_id
    app.number_input(key="portfolio_initial_cash").set_value(300000.0).run()
    assert "portfolio_analysis" not in app.session_state
    assert app.selectbox(key="portfolio_history_id").value == record.run_id
    assert len(app.metric) == 7


def test_failed_benchmark_saved_as_unavailable_without_exception_text(workspace, monkeypatch):
    from quantlab.ui import portfolio

    _, archives, create = workspace
    create()
    def failed(*args, **kwargs):
        raise PortfolioBacktestError("token=DO_NOT_DISPLAY /Users/private", reason="missing_close",
                                     date=pd.Timestamp("2025-02-03").date(), symbols=("159915",))
    monkeypatch.setattr(portfolio, "run_equal_weight_benchmark", failed)
    app = page()
    live = run_live(app)
    app.button(key="portfolio_archive_save").click().run()
    loaded = load_portfolio_run(archives, live.backtest.metadata["run_id"])
    assert loaded.benchmark_status == "unavailable"
    assert loaded.benchmark_failure["reason"] == "missing_close"
    assert "DO_NOT_DISPLAY" not in loaded.record.path.read_text()
    assert "DO_NOT_DISPLAY" not in str(app.warning)
    open_history(app, loaded.record.run_id)
    assert not any("累计收益差" in metric.label for metric in app.metric)


@pytest.mark.parametrize("failure", ["save", "list", "load"])
def test_storage_failure_is_safe_and_does_not_erase_live_result(workspace, monkeypatch, failure):
    from quantlab.ui import portfolio_records

    _, archives, create = workspace
    _, analysis, baseline, _ = create()
    record = save_portfolio_run(analysis, archives, name="保存前", benchmark=baseline)
    app = page()
    live = run_live(app)
    def failed(*args, **kwargs):
        raise OSError("token=DO_NOT_DISPLAY /Users/private")
    monkeypatch.setattr(portfolio_records, f"{failure}_portfolio_run" if failure != "list" else "list_portfolio_runs", failed)
    if failure == "save":
        app.button(key="portfolio_archive_save").click().run()
    else:
        open_history(app, record.run_id)
    assert not app.exception and app.error
    assert "DO_NOT_DISPLAY" not in str(app.error)
    assert app.session_state["portfolio_analysis"].backtest.metadata["run_id"] == live.backtest.metadata["run_id"]


def test_corrupt_archive_removes_stale_history_and_hides_filename(workspace):
    _, archives, create = workspace
    _, analysis, baseline, _ = create(with_set=False)
    record = save_portfolio_run(analysis, archives, name="原来完好", benchmark=baseline)
    app = page()
    open_history(app, record.run_id)
    assert app.metric
    record.path.write_text("{}")
    (archives / "token=DO_NOT_DISPLAY.json").write_text("{}")
    app.run()
    assert not app.exception and not app.metric
    assert any("2" in item.value and "校验" in item.value for item in app.warning)
    assert "DO_NOT_DISPLAY" not in str(app.warning)


def test_history_displays_saved_metric_assumptions_not_current_defaults(workspace):
    _, archives, create = workspace
    _, analysis, _, _ = create(with_set=False, analytics=AnalyticsConfig(periods_per_year=365, annual_risk_free_rate=.05), benchmark=False)
    record = save_portfolio_run(analysis, archives, name="特殊指标参数")
    app = page()
    open_history(app, record.run_id)
    assert any("365" in item.value and "5" in item.value for item in app.caption)
    assert any("未运行基准" in item.value for item in app.info)
    assert not any("累计收益差" in item.label for item in app.metric)


def test_empty_history_does_not_create_directory(workspace):
    _, archives, _ = workspace
    app = page()
    assert any("历史实验" in tab.label for tab in app.tabs)
    assert not archives.exists()
    assert not app.metric


def test_blank_name_keeps_live_result_without_writing(workspace):
    _, archives, create = workspace
    create()
    app = page()
    live = run_live(app)
    app.text_input(key="portfolio_archive_name").set_value("   ")
    app.button(key="portfolio_archive_save").click().run()
    assert not app.exception and app.error
    assert not archives.exists()
    assert app.session_state["portfolio_analysis"].backtest.metadata["run_id"] == live.backtest.metadata["run_id"]


def test_archive_directory_file_conflict_does_not_claim_run_already_saved(workspace):
    _, archives, create = workspace
    create()
    app = page()
    live = run_live(app)
    archives.write_text("existing file: preserve")
    app.button(key="portfolio_archive_save").click().run()
    assert not app.exception
    assert archives.read_text() == "existing file: preserve"
    notices = " ".join(item.value for item in list(app.info) + list(app.error))
    assert "路径冲突" in notices
    assert "该运行编号的档案已存在" not in notices
    assert app.session_state["portfolio_analysis"].backtest.metadata["run_id"] == live.backtest.metadata["run_id"]


def test_old_code_warning_is_visible_without_rewriting_archive(workspace):
    _, archives, create = workspace
    _, analysis, _, _ = create(with_set=False, benchmark=False)
    record = save_portfolio_run(analysis, archives, name="旧版实验")
    payload = json.loads(record.path.read_text())
    payload["strategy"]["metadata"]["code_sha256"] = "a" * 64
    core = {key: value for key, value in payload.items() if key != "content_sha256"}
    payload["content_sha256"] = sha256(json.dumps(
        core, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    record.path.write_text(json.dumps(payload, ensure_ascii=False))
    before = record.path.read_bytes()
    app = page()
    open_history(app, record.run_id)
    assert app.metric and any("代码" in item.value for item in app.warning)
    assert record.path.read_bytes() == before


def test_access_check_precedes_both_research_and_archive_reads(workspace, monkeypatch):
    import quantlab.data.research_storage as storage
    from quantlab.ui import portfolio_records

    monkeypatch.setenv("QUANTLAB_ACCESS_PASSWORD", "test-only-password")
    def forbidden(*args, **kwargs):
        pytest.fail("访问验证前不得读取研究或档案目录")
    monkeypatch.setattr(storage, "list_research_sets", forbidden)
    monkeypatch.setattr(portfolio_records, "list_portfolio_runs", forbidden)
    app = page()
    assert not app.selectbox and not app.tabs and not app.metric
    assert app.text_input(key="_quantlab_access_password")


def test_archive_scope_change_clears_history_without_replacing_live(workspace, monkeypatch, tmp_path):
    _, archives, create = workspace
    _, analysis, baseline, _ = create()
    record = save_portfolio_run(analysis, archives, name="原目录实验", benchmark=baseline)
    app = page()
    live = run_live(app)
    open_history(app, record.run_id)
    monkeypatch.setattr(runtime, "get_portfolio_storage_dir", lambda project: tmp_path / "other_runs")
    app.run()
    assert not app.exception
    assert not any(item.key == "portfolio_history_id" for item in app.selectbox)
    assert len(app.metric) == 7
    assert app.session_state["portfolio_analysis"].backtest.metadata["run_id"] == live.backtest.metadata["run_id"]


def test_public_save_and_history_are_not_shared_with_another_session(monkeypatch, make_portfolio_preview):
    monkeypatch.setenv("QUANTLAB_PUBLIC_MODE", "1")
    monkeypatch.delenv("QUANTLAB_ACCESS_PASSWORD", raising=False)
    app = page()
    root = Path(app.session_state["_quantlab_public_temporary_directory"].name)
    preview, _ = make_portfolio_preview(periods=45)
    save_research_set(preview, root / "research_sets", name="会话合成集合")
    app.run()
    live = run_live(app)
    app.button(key="portfolio_archive_save").click().run()
    run_id = live.backtest.metadata["run_id"]
    loaded = load_portfolio_run(root / "portfolio_runs", run_id)
    open_history(app, loaded.record.run_id)
    assert not app.exception and not app.error
    assert len(app.get("download_button")) == 1
    assert any("临时" in item.value for item in app.warning)
    other = page()
    assert not other.metric
    assert not other.get("download_button")
    assert not any(item.key == "portfolio_history_id" for item in other.selectbox)
    other_root = Path(other.session_state["_quantlab_public_temporary_directory"].name)
    assert other_root != root and not (other_root / "portfolio_runs").exists()


def test_public_archive_roots_are_session_scoped(monkeypatch):
    monkeypatch.setenv("QUANTLAB_PUBLIC_MODE", "1")
    script = """
from pathlib import Path
import streamlit as st
from quantlab.ui.runtime import get_research_storage_dir, get_portfolio_storage_dir
st.session_state['archive_root'] = get_portfolio_storage_dir(Path('/never-local-project'))
st.session_state['research_root'] = get_research_storage_dir(Path('/never-local-project'))
"""
    first = AppTest.from_string(script).run()
    second = AppTest.from_string(script).run()
    assert not first.exception and not second.exception
    root = first.session_state["archive_root"]
    assert root.parent == first.session_state["research_root"].parent
    assert root != second.session_state["archive_root"]
    assert "never-local-project" not in str(root)
    first.run()
    assert first.session_state["archive_root"] == root
    assert not root.exists()


@pytest.fixture
def downloaded_payloads(monkeypatch):
    """记录真实下载组件收到的字节，继续调用原组件，不模拟导出内容。"""
    import streamlit as st
    original = st.download_button
    payloads = []
    def capture(*args, **kwargs):
        payloads.append(kwargs.copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(st, "download_button", capture)
    return payloads


def test_download_is_only_for_selected_history_and_contains_that_snapshot(workspace, downloaded_payloads):
    _, archives, create = workspace
    _, historical, baseline, _ = create(cash=200000)
    record = save_portfolio_run(historical, archives, name="待导出历史", benchmark=baseline)
    app = page()
    live = run_live(app)
    assert not app.get("download_button"), "未选择历史，不能导出未保存的当前结果"
    open_history(app, record.run_id)
    assert len(app.get("download_button")) == 1, "历史页应提供实验包下载"
    payload = downloaded_payloads[-1]
    assert payload["mime"] == "application/zip" and payload["on_click"] == "ignore"
    assert record.run_id in payload["file_name"]
    with ZipFile(BytesIO(payload["data"])) as bundle:
        assert bundle.read("archive.json") == record.path.read_bytes()
    assert app.session_state["portfolio_analysis"].backtest.metadata["run_id"] == live.backtest.metadata["run_id"]
    assert any("分享" in item.value and "数据" in item.value for item in app.warning)


def test_switching_history_replaces_download_and_invalidated_record_removes_it(workspace, downloaded_payloads):
    _, archives, create = workspace
    records = []
    for cash in (100000, 200000):
        _, analysis, baseline, _ = create(with_set=False, cash=cash)
        records.append(save_portfolio_run(analysis, archives, name=str(cash), benchmark=baseline))
    app = page()
    for record in records:
        open_history(app, record.run_id)
        assert len(app.get("download_button")) == 1
        with ZipFile(BytesIO(downloaded_payloads[-1]["data"])) as bundle:
            assert bundle.read("archive.json") == record.path.read_bytes()
    records[-1].path.write_text("corrupt")
    app.run()
    assert not app.exception and not app.get("download_button") and not app.metric


def test_export_failure_keeps_history_and_live_result_without_private_error(workspace, monkeypatch):
    from quantlab.ui import portfolio_records
    _, archives, create = workspace
    _, analysis, baseline, _ = create()
    record = save_portfolio_run(analysis, archives, name="导出失败", benchmark=baseline)
    app = page()
    live = run_live(app)
    def failed(*args, **kwargs):
        raise OSError("secret-token /private/user")
    monkeypatch.setattr(portfolio_records, "build_portfolio_export", failed, raising=False)
    open_history(app, record.run_id)
    assert not app.exception and not app.get("download_button")
    assert app.error and "secret-token" not in str(app.error)
    assert len(app.metric) == 14
    assert app.session_state["portfolio_analysis"].backtest.metadata["run_id"] == live.backtest.metadata["run_id"]


def test_old_code_history_can_be_downloaded_without_original_research(workspace, downloaded_payloads):
    _, archives, create = workspace
    _, analysis, _, source = create(with_set=False, benchmark=False)
    record = save_portfolio_run(analysis, archives, name="独立备份")
    doc = json.loads(record.path.read_bytes())
    doc["strategy"]["metadata"]["code_sha256"] = "b" * 64
    doc["content_sha256"] = sha256(json.dumps(
        {key: value for key, value in doc.items() if key != "content_sha256"},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    record.path.write_text(json.dumps(doc, ensure_ascii=False))
    source.rename(source.with_name(source.name + "_moved"))
    app = page()
    open_history(app, record.run_id)
    assert not app.exception and len(app.get("download_button")) == 1
    with ZipFile(BytesIO(downloaded_payloads[-1]["data"])) as bundle:
        assert bundle.read("archive.json") == record.path.read_bytes()
        assert json.loads(bundle.read("metadata.json"))["strategy"]["code_sha256"] == "b" * 64


def test_yearly_history_is_derived_without_rewriting_archive_or_export(workspace, monkeypatch):
    from quantlab.portfolio import build_portfolio_export
    from quantlab.ui import portfolio

    _, archives, create = workspace
    _, strategy, benchmark, source = create(periods=300)
    record = save_portfolio_run(strategy, archives, name="跨年历史", benchmark=benchmark)
    original = record.path.read_bytes()
    zip_before = build_portfolio_export(load_portfolio_run(archives, record.run_id)).content
    app = page()
    live = run_live(app)
    live_id = live.backtest.metadata["run_id"]
    source.rename(source.with_name(source.name + "_moved"))
    def forbidden(*args, **kwargs):
        pytest.fail("查看历史年度表不得重跑回测或原全区间绩效")
    monkeypatch.setattr(portfolio, "run_factor_portfolio", forbidden)
    monkeypatch.setattr(portfolio, "run_equal_weight_benchmark", forbidden)
    monkeypatch.setattr(portfolio, "analyze_portfolio_result", forbidden)
    open_history(app, record.run_id)
    tables = [item.value for item in app.dataframe if "策略区间收益" in item.value.columns]
    assert len(tables) == 2, "当前与历史各自有独立年度表"
    assert tables[0]["年份"].tolist() == tables[1]["年份"].tolist() == [2025, 2026]
    # 当前默认选两只，历史选一只，不能混用策略或基准结果。
    assert tables[0]["策略区间收益"].tolist() != tables[1]["策略区间收益"].tolist()
    curve = strategy.backtest.equity_curve
    year_end = curve.loc[curve.date.dt.year.eq(2025), "equity"].iloc[-1]
    assert tables[1].iloc[1]["策略区间收益"] == pytest.approx(curve.equity.iloc[-1] / year_end - 1)
    assert app.session_state["portfolio_analysis"].backtest.metadata["run_id"] == live_id
    assert len(app.metric) == 14 and not app.exception and not app.error
    assert record.path.read_bytes() == original
    assert build_portfolio_export(load_portfolio_run(archives, record.run_id)).content == zip_before
    # 取消历史选择仅清除历史年度表，不残留上一份结果。
    app.selectbox(key="portfolio_history_id").set_value(None).run()
    assert len([item for item in app.dataframe if "策略区间收益" in item.value.columns]) == 1
