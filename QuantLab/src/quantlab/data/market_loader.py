from pathlib import Path

import pandas as pd

from quantlab.data.validator import validate_market_data

def load_market_csv(file_path: str | Path) -> pd.DataFrame:
    """读取ETF行情CSV，校验列名并转换关键字段类型。"""
    path = Path(file_path)

    market_data = pd.read_csv(
        path,
        dtype={"symbol": "string"},
    )

    validate_market_data(market_data)

    market_data["date"] = pd.to_datetime(
        market_data["date"],
        errors="raise",
    )

    market_data = market_data.sort_values(
        by=["symbol", "date"],
    ).reset_index(drop=True)

    return market_data
