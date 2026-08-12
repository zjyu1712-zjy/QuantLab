from pathlib import Path

import pandas as pd


def save_market_parquet(
    market_data: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """把标准化行情保存为Parquet文件。"""
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    market_data.to_parquet(
        path,
        index=False,
    )

    return path
