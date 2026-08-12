from pathlib import Path

import pytest

from quantlab.ui.trade_data import load_uploaded_trade_csv


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_uploaded_trade_csv_reuses_standard_validation() -> None:
    content = (FIXTURES / "valid_trades.csv").read_bytes()

    trades = load_uploaded_trade_csv("valid_trades.csv", content)

    assert trades["trade_id"].tolist() == ["T001", "T002"]
    assert trades["side"].tolist() == ["BUY", "SELL"]
    assert str(trades["symbol"].dtype) == "string"


def test_load_uploaded_trade_csv_rejects_wrong_extension() -> None:
    with pytest.raises(ValueError, match="CSV"):
        load_uploaded_trade_csv("trades.txt", b"trade_id")


def test_load_uploaded_trade_csv_rejects_empty_file() -> None:
    with pytest.raises(ValueError, match="不能为空"):
        load_uploaded_trade_csv("trades.csv", b"")
