"""第二阶段测试使用的合成行情，不接触用户的数据目录。"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from quantlab.data.importer import import_market_csv


@pytest.fixture
def sample_assets():
    def frame(symbol, dates):
        return pd.DataFrame({
            "date": pd.to_datetime(dates), "symbol": symbol,
            "open": 3.0, "high": 3.2, "low": 2.9,
            "close": 3.1, "volume": 1000.0,
        })

    return {
        "510300": frame("510300", ["2025-01-02", "2025-01-03", "2025-01-07"]),
        "159915": frame("159915", ["2025-01-03", "2025-01-06", "2025-01-07"]),
    }


@pytest.fixture
def imported_pair(tmp_path, sample_assets):
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    results = []
    for index, (symbol, data) in enumerate(sample_assets.items()):
        source = tmp_path / f"{symbol}.csv"
        data.to_csv(source, index=False)
        results.append(import_market_csv(
            source, raw, processed,
            imported_at=datetime(2025, 1, 10 + index, tzinfo=timezone.utc),
        ))
    return raw, processed, tuple(results)
