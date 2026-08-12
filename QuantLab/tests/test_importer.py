from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest

from quantlab.data.importer import import_market_csv
from quantlab.data.market_loader import load_market_csv


def test_import_market_csv_runs_complete_pipeline(
    tmp_path: Path,
) -> None:
    source = Path("tests/fixtures/valid_etf.csv")

    result = import_market_csv(
        source,
        tmp_path / "raw",
        tmp_path / "processed",
        imported_at=datetime(
            2025,
            1,
            10,
            8,
            30,
            tzinfo=timezone.utc,
        ),
    )

    archived_path = Path(result.version.archived_path)
    processed_path = Path(result.processed_path)
    result_path = archived_path.parent / "import_result.json"

    assert archived_path.read_bytes() == source.read_bytes()
    assert processed_path.exists()
    assert result.symbol == "510300"
    assert result.row_count == 5
    assert processed_path.parent.name == "510300"
    assert processed_path.stem == result.version.version_id

    expected_data = load_market_csv(source)
    restored_data = pd.read_parquet(processed_path)
    pd.testing.assert_frame_equal(restored_data, expected_data)

    saved_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert saved_result["symbol"] == "510300"
    assert saved_result["row_count"] == 5
    assert saved_result["processed_path"] == str(processed_path)


def test_import_market_csv_rejects_multiple_symbols(
    tmp_path: Path,
) -> None:
    source = tmp_path / "multiple_etfs.csv"
    source.write_text(
        "date,symbol,open,high,low,close,volume\n"
        "2025-01-02,510300,3.95,4.02,3.93,4.00,1200000\n"
        "2025-01-02,159915,2.00,2.05,1.98,2.03,900000\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="只能包含一只 ETF"):
        import_market_csv(
            source,
            tmp_path / "raw",
            tmp_path / "processed",
        )

    assert list((tmp_path / "raw").glob("*/multiple_etfs.csv"))
    assert not (tmp_path / "processed").exists()
