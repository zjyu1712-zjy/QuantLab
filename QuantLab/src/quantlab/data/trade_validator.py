import pandas as pd


REQUIRED_TRADE_COLUMNS = [
    "trade_id",
    "trade_datetime",
    "symbol",
    "side",
    "quantity",
    "price",
    "fee",
]


def validate_trade_columns(trade_data: pd.DataFrame) -> None:
    """在类型转换前检查交易表是否具备必要字段和值。"""
    missing_columns = [
        column
        for column in REQUIRED_TRADE_COLUMNS
        if column not in trade_data.columns
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"交易记录缺少必要列：{missing_text}")

    if trade_data.empty:
        raise ValueError("交易记录不能为空")

    if trade_data[REQUIRED_TRADE_COLUMNS].isna().any().any():
        raise ValueError("交易记录包含缺失值")


def validate_trade_data(trade_data: pd.DataFrame) -> None:
    """校验已完成类型转换的个人交易记录。"""
    validate_trade_columns(trade_data)

    if trade_data["trade_id"].str.strip().eq("").any():
        raise ValueError("交易编号不能为空")

    if trade_data["symbol"].str.strip().eq("").any():
        raise ValueError("证券代码不能为空")

    if trade_data["trade_id"].duplicated().any():
        raise ValueError("交易编号不能重复")

    valid_sides = {"BUY", "SELL"}
    if not trade_data["side"].isin(valid_sides).all():
        raise ValueError("买卖方向只允许 BUY 或 SELL")

    quantity = trade_data["quantity"]
    if (quantity <= 0).any() or (quantity % 1 != 0).any():
        raise ValueError("成交数量必须是正整数")

    if (trade_data["price"] <= 0).any():
        raise ValueError("成交价格必须大于 0")

    if (trade_data["fee"] < 0).any():
        raise ValueError("手续费不能为负数")
