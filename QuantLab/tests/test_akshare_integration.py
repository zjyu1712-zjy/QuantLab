"""真实网络集成测试；默认跳过，显式设置环境变量后运行。"""

from datetime import date
import os

import pytest

from quantlab.data.providers import AkShareETFProvider


@pytest.mark.skipif(
    os.environ.get("QUANTLAB_RUN_AKSHARE_INTEGRATION") != "1",
    reason="默认不访问外部行情接口",
)
def test_real_akshare_etf_history() -> None:
    provider = AkShareETFProvider(timeout_seconds=30, max_attempts=2)
    data = provider.fetch_etf_history(
        "510300", date(2025, 1, 2), date(2025, 1, 10)
    )

    assert not data.empty
    assert list(data.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
