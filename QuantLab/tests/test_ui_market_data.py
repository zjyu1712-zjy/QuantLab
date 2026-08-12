from pathlib import Path

import pandas as pd
import pytest

from quantlab.ui.market_data import (
    import_uploaded_market_history,
    import_uploaded_market_csv,
    inspect_uploaded_market_history,
    list_market_datasets,
    load_market_dataset,
    market_source_label,
)


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
