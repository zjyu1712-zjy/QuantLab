import pandas as pd
import pytest

from quantlab.data.validator import validate_market_data


def make_valid_market_data() -> pd.DataFrame:
    """创建一份供测试使用的正确行情数据。"""
    return pd.DataFrame(
        {
            "date": ["2025-01-02"],
            "symbol": ["510300"],
            "open": [3.95],
            "high": [4.02],
            "low": [3.93],
            "close": [4.00],
            "volume": [1200000],
        }
    )


def test_valid_market_data_passes() -> None:
    market_data = make_valid_market_data()
    validate_market_data(market_data)
def test_zero_price_raises_error() -> None:
    market_data = make_valid_market_data()
    market_data.loc[0, "close"] = 0

    with pytest.raises(ValueError, match="价格必须大于0"):
        validate_market_data(market_data)


def test_negative_volume_raises_error() -> None:
    market_data = make_valid_market_data()
    market_data.loc[0, "volume"] = -1

    with pytest.raises(ValueError, match="成交量不能为负数"):
        validate_market_data(market_data)

def test_missing_column_raises_error() -> None:
    market_data = make_valid_market_data().drop(columns=["volume"])

    with pytest.raises(ValueError, match="volume"):
        validate_market_data(market_data)


def test_empty_data_raises_error() -> None:
    market_data = make_valid_market_data().iloc[0:0]

    with pytest.raises(ValueError, match="行情数据为空"):
        validate_market_data(market_data)


def test_missing_value_raises_error() -> None:
    market_data = make_valid_market_data()
    market_data.loc[0, "close"] = float("nan")

    with pytest.raises(ValueError, match="close"):
        validate_market_data(market_data)


def test_duplicate_data_raises_error() -> None:
    market_data = make_valid_market_data()
    market_data = pd.concat([market_data, market_data], ignore_index=True)

    with pytest.raises(ValueError, match="重复记录"):
        validate_market_data(market_data)
def test_invalid_high_price_raises_error() -> None:
    market_data = make_valid_market_data()
    market_data.loc[0, "high"] = 3.99

    with pytest.raises(ValueError, match="最高价不能低于"):
        validate_market_data(market_data)


def test_invalid_low_price_raises_error() -> None:
    market_data = make_valid_market_data()
    market_data.loc[0, "low"] = 3.96

    with pytest.raises(ValueError, match="最低价不能高于"):
        validate_market_data(market_data)
