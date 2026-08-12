import pandas as pd
import pytest

from quantlab.data.trade_loader import load_trade_csv


def test_load_trade_csv_converts_valid_data() -> None:
    trade_data = load_trade_csv("tests/fixtures/valid_trades.csv")

    assert trade_data["trade_id"].tolist() == ["T001", "T002"]
    assert trade_data["side"].tolist() == ["BUY", "SELL"]
    assert trade_data["quantity"].dtype == "int64"
    assert trade_data["price"].dtype == "float64"
    assert trade_data["fee"].dtype == "float64"
    assert str(trade_data["symbol"].dtype) == "string"
    assert str(trade_data["trade_datetime"].dt.tz) == "Asia/Shanghai"
    assert trade_data["trade_datetime"].is_monotonic_increasing


@pytest.mark.parametrize(
    ("column", "invalid_value", "message"),
    [
        ("side", "HOLD", "只允许 BUY 或 SELL"),
        ("quantity", 0, "必须是正整数"),
        ("quantity", 1.5, "必须是正整数"),
        ("price", 0, "必须大于 0"),
        ("fee", -0.01, "不能为负数"),
    ],
)
def test_load_trade_csv_rejects_invalid_values(
    tmp_path,
    column: str,
    invalid_value: object,
    message: str,
) -> None:
    trade = {
        "trade_id": "T001",
        "trade_datetime": "2025-01-03 10:00:00",
        "symbol": "510300",
        "side": "BUY",
        "quantity": 100,
        "price": 4.06,
        "fee": 0.50,
    }
    trade[column] = invalid_value
    trade_data = pd.DataFrame([trade])
    path = tmp_path / "invalid_trade.csv"
    trade_data.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_trade_csv(path)


def test_load_trade_csv_rejects_duplicate_trade_id(tmp_path) -> None:
    trade_data = pd.read_csv("tests/fixtures/valid_trades.csv")
    trade_data.loc[1, "trade_id"] = trade_data.loc[0, "trade_id"]
    path = tmp_path / "duplicate_trade.csv"
    trade_data.to_csv(path, index=False)

    with pytest.raises(ValueError, match="交易编号不能重复"):
        load_trade_csv(path)


def test_load_trade_csv_rejects_missing_column(tmp_path) -> None:
    trade_data = pd.read_csv("tests/fixtures/valid_trades.csv")
    path = tmp_path / "missing_fee.csv"
    trade_data.drop(columns="fee").to_csv(path, index=False)

    with pytest.raises(ValueError, match="缺少必要列：fee"):
        load_trade_csv(path)
