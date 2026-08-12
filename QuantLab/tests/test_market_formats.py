import pandas as pd
import pytest

from quantlab.data import (
    AkShareETFProvider,
    create_market_template_bytes,
    get_market_data_provider,
    inspect_market_file,
)


def test_inspect_quantlab_file_uses_embedded_symbol() -> None:
    content = (
        b"date,symbol,open,high,low,close,volume\n"
        b"2025-01-02,510300,3.95,4.02,3.93,4.00,1200000\n"
    )

    inspection = inspect_market_file("market.csv", content)

    assert inspection.source_key == "quantlab"
    assert inspection.symbol == "510300"
    assert list(inspection.standardized_data.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert pd.api.types.is_datetime64_any_dtype(
        inspection.standardized_data["date"]
    )


def test_inspect_chinese_market_file_maps_common_headers() -> None:
    content = (
        "日期,开盘价,最高价,最低价,收盘价,成交量\n"
        "2025/01/03,4.01,4.08,3.99,4.06,1350000\n"
    ).encode("gb18030")

    inspection = inspect_market_file(
        "东方财富_历史行情.csv",
        content,
        symbol="510300",
    )

    assert inspection.source_key == "eastmoney"
    assert inspection.encoding == "gb18030"
    assert inspection.field_mapping["日期"] == "date"
    assert inspection.field_mapping["收盘价"] == "close"
    assert inspection.standardized_data.loc[0, "close"] == pytest.approx(4.06)


def test_inspect_yahoo_file_ignores_adjusted_close() -> None:
    content = (
        b"Date,Open,High,Low,Close,Adj Close,Volume\n"
        b"2025-01-02,3.95,4.02,3.93,4.00,3.98,1200000\n"
    )

    inspection = inspect_market_file(
        "yahoo_history.csv",
        content,
        symbol="510300",
    )

    assert inspection.source_key == "yahoo"
    assert "Adj Close" not in inspection.field_mapping
    assert inspection.standardized_data.loc[0, "close"] == pytest.approx(4.00)


def test_inspect_market_file_explains_unrecognized_format() -> None:
    with pytest.raises(ValueError, match="暂未识别"):
        inspect_market_file(
            "unknown.csv",
            b"day,last\n2025-01-02,4.0\n",
            symbol="510300",
        )


def test_market_template_has_only_public_six_columns() -> None:
    assert create_market_template_bytes() == (
        b"date,open,high,low,close,volume\n"
    )


def test_default_provider_is_akshare_without_making_request() -> None:
    provider = get_market_data_provider(env={})

    assert isinstance(provider, AkShareETFProvider)
    assert provider.key == "akshare"
