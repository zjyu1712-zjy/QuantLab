import json
from io import BytesIO
from pathlib import Path

from streamlit.testing.v1 import AppTest

from quantlab.data.market_loader import load_market_csv
from quantlab.ui import MarketDatasetRecord


def test_backtest_page_saves_each_explicit_run_with_selected_data_version(tmp_path, monkeypatch):
    import quantlab.ui as ui

    data = load_market_csv("tests/fixtures/valid_etf.csv")
    path = tmp_path / "fixture.parquet"
    data.to_parquet(path)
    record = MarketDatasetRecord("510300", "fixture-v1", 5, "2026-09-22", str(path))
    monkeypatch.setattr(ui, "list_market_datasets", lambda *args: [record])
    monkeypatch.setattr(ui, "get_market_storage_dirs", lambda root: (
        tmp_path / "raw" / "market", tmp_path / "processed" / "market",
    ))
    page = AppTest.from_file("pages/3_双均线回测.py").run()
    assert not page.exception
    page.button[0].click().run()
    assert not page.exception
    files = list((tmp_path / "backtests").glob("*.json"))
    assert len(files) == 1, "点击运行后必须自动保存回测记录"
    payload = json.loads(files[0].read_text())
    assert payload["metadata"]["data"]["version"] == "fixture-v1"
    assert payload["metadata"]["data"]["source"] == str(path)
    assert payload["metadata"]["strategy"]["parameters"] == {"short_window": 1, "long_window": 2}
    assert payload["metrics"]["max_drawdown"] <= 0
    assert len(page.get("download_button")) == 1
    page.button[0].click().run()
    assert not page.exception
    assert len(list((tmp_path / "backtests").glob("*.json"))) == 2


def test_comparison_page_archives_both_simulations_but_not_private_trade_upload(tmp_path, monkeypatch):
    import quantlab.ui as ui
    import streamlit as st

    path = tmp_path / "fixture.parquet"
    load_market_csv("tests/fixtures/valid_etf.csv").to_parquet(path)
    record = MarketDatasetRecord("510300", "comparison-v1", 5, "2026-09-22", str(path))
    monkeypatch.setattr(ui, "list_market_datasets", lambda *args: [record])
    monkeypatch.setattr(ui, "get_market_storage_dirs", lambda root: (
        tmp_path / "raw" / "market", tmp_path / "processed" / "market",
    ))
    # AppTest 没有上传操作接口，只替代文件上传边界，实际 CSV 加载与比较照常执行。
    upload = BytesIO(Path("tests/fixtures/valid_trades.csv").read_bytes())
    upload.name = "private-trades.csv"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    page = AppTest.from_file("pages/4_三方对比.py").run()
    page.button[0].click().run()
    assert not page.exception
    assert not page.error
    files = list((tmp_path / "backtests").glob("*.json"))
    assert len(files) == 2
    payloads = [json.loads(path.read_text()) for path in files]
    assert {p["metadata"]["strategy"]["name"] for p in payloads} == {"buy_and_hold", "dual_moving_average"}
    assert all(p["metadata"]["data"]["version"] == "comparison-v1" for p in payloads)
    assert len(page.get("download_button")) == 2
    assert all("private-trades.csv" not in path.read_text() for path in files)


def test_disk_save_failure_still_leaves_download_available(tmp_path, monkeypatch):
    import quantlab.ui as ui

    path = tmp_path / "fixture.parquet"
    load_market_csv("tests/fixtures/valid_etf.csv").to_parquet(path)
    record = MarketDatasetRecord("510300", "fixture-v1", 5, "2026-09-22", str(path))
    monkeypatch.setattr(ui, "list_market_datasets", lambda *args: [record])
    monkeypatch.setattr(ui, "get_market_storage_dirs", lambda root: (
        tmp_path / "raw" / "market", tmp_path / "processed" / "market",
    ))
    (tmp_path / "backtests").write_text("a file prevents creating the archive directory")
    page = AppTest.from_file("pages/3_双均线回测.py").run()
    page.button[0].click().run()
    assert not page.exception
    assert len(page.warning) == 1
    assert len(page.get("download_button")) == 1
    assert len(page.metric) >= 6
