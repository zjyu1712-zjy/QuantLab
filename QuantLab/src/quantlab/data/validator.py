import pandas as pd


REQUIRED_MARKET_COLUMNS = (
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

PRICE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
)


def validate_market_columns(market_data: pd.DataFrame) -> None:
    """检查ETF行情是否包含全部必要列。"""
    missing_columns = set(REQUIRED_MARKET_COLUMNS) - set(market_data.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"行情数据缺少必要列：{missing_text}")


def validate_market_data(market_data: pd.DataFrame) -> None:
    """检查ETF行情的完整性和数值合理性。"""
    validate_market_columns(market_data)

    if market_data.empty:
        raise ValueError("行情数据为空")

    columns_with_missing_values = [
        column
        for column in REQUIRED_MARKET_COLUMNS
        if market_data[column].isna().any()
    ]

    if columns_with_missing_values:
        missing_text = ", ".join(columns_with_missing_values)
        raise ValueError(f"行情数据存在空值：{missing_text}")

    invalid_price_columns = [
        column
        for column in PRICE_COLUMNS
        if (market_data[column] <= 0).any()
    ]

    if invalid_price_columns:
        invalid_text = ", ".join(invalid_price_columns)
        raise ValueError(f"行情价格必须大于0：{invalid_text}")

    if (market_data["volume"] < 0).any():
        raise ValueError("成交量不能为负数")

    invalid_high = (
        market_data["high"]
        < market_data[["open", "low", "close"]].max(axis=1)
    )

    if invalid_high.any():
        raise ValueError("最高价不能低于开盘价、最低价或收盘价")

    invalid_low = (
        market_data["low"]
        > market_data[["open", "high", "close"]].min(axis=1)
    )

    if invalid_low.any():
        raise ValueError("最低价不能高于开盘价、最高价或收盘价")
    duplicate_count = int(
        market_data.duplicated(subset=["date", "symbol"]).sum()
    )

    if duplicate_count > 0:
        raise ValueError(f"行情数据包含 {duplicate_count} 条重复记录")
