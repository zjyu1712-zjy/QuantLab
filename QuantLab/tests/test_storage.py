from pathlib import Path

import pandas as pd

from quantlab.data.market_loader import load_market_csv
from quantlab.data.storage import save_market_parquet


def test_save_market_parquet_round_trip(tmp_path: Path) -> None:
    original_data = load_market_csv("tests/fixtures/valid_etf.csv")
    output_path = tmp_path / "nested" / "market.parquet"

    saved_path = save_market_parquet(
        original_data,
        output_path,
    )

    assert saved_path == output_path
    assert output_path.exists()

    restored_data = pd.read_parquet(output_path)

    pd.testing.assert_frame_equal(
        restored_data,
        original_data,
    )
