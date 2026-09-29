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


@pytest.fixture
def research_preview(imported_pair):
    """后续保存/页面测试共用的真实版本预览，所有文件均位于临时目录。"""
    from datetime import date
    from quantlab.data.catalog import scan_market_catalog
    from quantlab.data.research_sets import DatasetSelection, build_research_set

    raw, processed, _ = imported_pair
    selected = tuple(DatasetSelection(record, "none") for record in scan_market_catalog(processed, raw).records)
    return build_research_set(selected, start=date(2025, 1, 2), end=date(2025, 1, 7))


@pytest.fixture
def make_portfolio_preview(tmp_path):
    """组合研究用真实导入链路；所有文件与每次调用均隔离，正常成交量为正。"""
    from itertools import count

    import numpy as np

    from quantlab.data.calendar import parse_calendar_csv
    from quantlab.data.catalog import scan_market_catalog
    from quantlab.data.research_sets import DatasetSelection, build_research_set

    sequence = count()

    def build(*, periods=65, gaps=None, supplied=False, adjustment="none",
              bump_after=None, zero_volume=None):
        source_root = tmp_path / f"portfolio_sources_{next(sequence)}"
        source_root.mkdir()
        raw, processed = source_root / "raw", source_root / "processed"
        dates = pd.date_range("2025-01-02", periods=periods, freq="B").astype("datetime64[ns]")
        paths = (("510300", 100.0 + 0.5 * np.arange(periods)),
                 ("159915", 200.0 - 0.5 * np.arange(periods)))
        for symbol, values in paths:
            if bump_after is not None:
                values[bump_after:] *= 2
            frame = pd.DataFrame({
                "date": dates, "symbol": symbol, "open": values,
                "high": values, "low": values, "close": values, "volume": 1000.0,
            })
            frame.loc[(zero_volume or {}).get(symbol, []), "volume"] = 0.0
            frame = frame.drop(index=(gaps or {}).get(symbol, []))
            path = source_root / f"{symbol}.csv"
            frame.to_csv(path, index=False)
            import_market_csv(path, raw, processed,
                              imported_at=datetime(2025, 9, 1, tzinfo=timezone.utc))
        calendar = None
        if supplied:
            content = ("date\n" + "\n".join(day.date().isoformat() for day in dates) + "\n").encode()
            calendar = parse_calendar_csv(
                content, source="组合测试合成日期轴",
                coverage_start=dates[0].date(), coverage_end=dates[-1].date(),
            )
        scan = scan_market_catalog(processed, raw)
        assert not scan.issues and len(scan.records) == 2
        preview = build_research_set(
            tuple(DatasetSelection(record, adjustment) for record in scan.records),
            start=dates[0].date(), end=dates[-1].date(), calendar=calendar,
        )
        return preview, source_root

    return build
