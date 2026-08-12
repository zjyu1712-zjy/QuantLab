from pathlib import Path

import pandas as pd

from quantlab.data.trade_validator import (
    validate_trade_columns,
    validate_trade_data,
)


TRADE_TIMEZONE = "Asia/Shanghai"


def load_trade_csv(file_path: str | Path) -> pd.DataFrame:
    """读取个人交易 CSV，转换类型、校验并按成交时间排序。"""
    path = Path(file_path)
    trade_data = pd.read_csv(
        path,
        dtype={
            "trade_id": "string",
            "symbol": "string",
            "side": "string",
        },
    )

    validate_trade_columns(trade_data)

    for column in ["trade_id", "symbol", "side"]:
        trade_data[column] = trade_data[column].str.strip()

    trade_data["side"] = trade_data["side"].str.upper()
    trade_data["trade_datetime"] = pd.to_datetime(
        trade_data["trade_datetime"],
        errors="raise",
    ).dt.tz_localize(TRADE_TIMEZONE)
    trade_data["quantity"] = pd.to_numeric(
        trade_data["quantity"],
        errors="raise",
    )
    trade_data["price"] = pd.to_numeric(
        trade_data["price"],
        errors="raise",
    ).astype("float64")
    trade_data["fee"] = pd.to_numeric(
        trade_data["fee"],
        errors="raise",
    ).astype("float64")

    validate_trade_data(trade_data)
    trade_data["quantity"] = trade_data["quantity"].astype("int64")

    return trade_data.sort_values(
        by=["trade_datetime", "trade_id"],
    ).reset_index(drop=True)
