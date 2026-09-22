from pathlib import Path

import pandas as pd
import pytest

from quantlab.data.providers import (
    MarketDataProviderNoDataError,
    MarketDataProviderResponseError,
)
from quantlab.ui.market_data import (
    get_cached_market_data_provider,
    friendly_market_error,
    import_uploaded_market_history,
    import_uploaded_market_csv,
    inspect_uploaded_market_history,
    list_market_datasets,
    load_market_dataset,
    market_source_label,
)


def test_cached_market_provider_is_reused_across_page_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quantlab.ui.market_data as market_data_module

    sentinel = object()
    calls = 0

    def build_provider() -> object:
        nonlocal calls
        calls += 1
        return sentinel

    get_cached_market_data_provider.clear()
    monkeypatch.setattr(
        market_data_module,
        "get_market_data_provider",
        build_provider,
        raising=False,
    )
    try:
        first = get_cached_market_data_provider()
        second = get_cached_market_data_provider()
    finally:
        get_cached_market_data_provider.clear()

    assert first is sentinel
    assert second is sentinel
    assert calls == 1


def test_market_source_label_for_ifind_fetch() -> None:
    assert market_source_label("ifind", "miss") == "iFinD"


def test_market_source_label_for_cached_ifind_data() -> None:
    assert market_source_label("ifind", "hit") == "本地数据（iFinD）"


def test_market_source_label_for_akshare_fetch() -> None:
    assert market_source_label("akshare", "miss") == "AkShare"


def test_market_source_label_for_extended_ifind_data() -> None:
    assert market_source_label("ifind", "extended") == "iFinD"


def test_market_source_label_hides_unknown_provider_key() -> None:
    assert market_source_label("internal-provider-key", "miss") == "其他行情服务"


def test_market_source_label_handles_empty_cached_provider() -> None:
    assert market_source_label("", "hit") == "本地数据（其他行情服务）"


def test_friendly_market_error_uses_no_data_error_type_not_message() -> None:
    message, detail = friendly_market_error(
        MarketDataProviderNoDataError("该时间段没有任何记录")
    )

    assert message == "未找到该 ETF 的行情数据，请检查代码或日期范围。"
    assert detail == "该时间段没有任何记录"


def test_friendly_market_error_keeps_response_error_as_service_failure() -> None:
    message, detail = friendly_market_error(
        MarketDataProviderResponseError("未找到必需的响应字段")
    )

    assert message == (
        "行情服务暂时无法连接，请稍后重试，或者使用文件上传方式导入数据。"
    )
    assert detail == "未找到必需的响应字段"


def test_friendly_market_error_explains_invalid_etf_symbol() -> None:
    message, detail = friendly_market_error(
        ValueError(
            "ETF 代码应为六位数字，例如 159516、159516.SZ 或 sh510300。"
        )
    )

    assert message == "请输入正确的 ETF 代码，例如 159516 或 510300。"
    assert detail == "ETF 代码应为六位数字，例如 159516、159516.SZ 或 sh510300。"


def test_uploaded_market_csv_runs_existing_import_pipeline(
    tmp_path: Path,
) -> None:
    source = Path("tests/fixtures/valid_etf.csv")

    result = import_uploaded_market_csv(
        source.name,
        source.read_bytes(),
        tmp_path / "raw",
        tmp_path / "processed",
    )

    assert result.symbol == "510300"
    assert result.row_count == 5
    assert Path(result.version.archived_path).name == source.name
    assert Path(result.processed_path).is_file()


def test_uploaded_market_csv_rejects_empty_content(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="不能为空"):
        import_uploaded_market_csv(
            "empty.csv",
            b"",
            tmp_path / "raw",
            tmp_path / "processed",
        )


def test_market_dataset_catalog_and_load(
    tmp_path: Path,
) -> None:
    source = Path("tests/fixtures/valid_etf.csv")
    result = import_uploaded_market_csv(
        source.name,
        source.read_bytes(),
        tmp_path / "raw",
        tmp_path / "processed",
    )

    records = list_market_datasets(
        tmp_path / "processed",
        tmp_path / "raw",
    )
    restored = load_market_dataset(records[0].processed_path)

    assert len(records) == 1
    assert records[0].symbol == "510300"
    assert records[0].row_count == 5
    assert records[0].version_id == result.version.version_id
    assert isinstance(restored, pd.DataFrame)
    assert len(restored) == 5


def test_market_dataset_catalog_ignores_uncommitted_version(
    tmp_path: Path,
) -> None:
    processed_path = tmp_path / "processed" / "159516" / "incomplete.parquet"
    processed_path.parent.mkdir(parents=True)
    pd.DataFrame({"date": ["2025-01-02"]}).to_parquet(processed_path)
    metadata_path = (
        tmp_path / "raw" / "ifind" / "incomplete" / "provider_metadata.json"
    )
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text("{}", encoding="utf-8")

    assert list_market_datasets(tmp_path / "processed", tmp_path / "raw") == []


def test_friendly_market_import_preserves_original_and_normalizes(
    tmp_path: Path,
) -> None:
    original = (
        "日期,开盘,最高,最低,收盘,成交量\n"
        "2025-01-02,3.95,4.02,3.93,4.00,1200000\n"
    ).encode("gb18030")

    inspection = inspect_uploaded_market_history(
        "东方财富_510300.csv",
        original,
        symbol="510300",
    )
    result = import_uploaded_market_history(
        "东方财富_510300.csv",
        original,
        tmp_path / "raw",
        tmp_path / "processed",
        symbol="510300",
    )
    restored = load_market_dataset(result.import_result.processed_path)

    assert inspection.source_key == "eastmoney"
    assert Path(result.original_path).read_bytes() == original
    assert result.symbol == "510300"
    assert result.row_count == 1
    assert list(restored.columns) == [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


def test_market_catalog_page_keeps_healthy_version_when_another_is_corrupt(imported_pair, caplog):
    raw, processed, imports = imported_pair
    bad_version_dir = Path(imports[0].version.archived_path).parent
    (bad_version_dir / "import_result.json").write_text("{unfinished", encoding="utf-8")
    records = list_market_datasets(processed, raw)
    assert len(records) == 1
    assert records[0].version_id == imports[1].version.version_id
    assert isinstance(records[0].processed_path, str)
    assert imports[0].version.version_id in caplog.text
