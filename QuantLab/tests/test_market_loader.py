import pandas as pd
import pytest

from quantlab.data.market_loader import load_market_csv


def test_load_market_csv_converts_data_types() -> None:
    market_data = load_market_csv("tests/fixtures/valid_etf.csv")

    assert len(market_data) == 5
    assert isinstance(market_data.loc[0, "date"], pd.Timestamp)
    assert str(market_data["symbol"].dtype) == "string"


def test_load_market_csv_runs_validation() -> None:
    with pytest.raises(ValueError, match="volume"):
        load_market_csv("tests/fixtures/missing_volume.csv")

def test_load_market_csv_sorts_dates() -> None:
    market_data = load_market_csv("tests/fixtures/unsorted_etf.csv")

    assert market_data["date"].is_monotonic_increasing
    assert market_data.index.tolist() == [0, 1, 2]
