from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from quantlab.data.trade_loader import load_trade_csv


def load_uploaded_trade_csv(
    filename: str,
    content: bytes,
) -> pd.DataFrame:
    """读取界面上传的交易 CSV，并复用统一的类型转换和校验。"""
    safe_name = Path(filename).name
    if Path(safe_name).suffix.lower() != ".csv":
        raise ValueError("交易记录必须是 CSV 文件")

    if not content:
        raise ValueError("交易 CSV 不能为空")

    with TemporaryDirectory(prefix="quantlab_trade_") as temp_dir:
        upload_path = Path(temp_dir) / safe_name
        upload_path.write_bytes(content)
        return load_trade_csv(upload_path)
