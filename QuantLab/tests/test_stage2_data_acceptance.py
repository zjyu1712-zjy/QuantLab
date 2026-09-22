"""合成示例端到端：真实导入→目录→显式日历→预览→保存→移走源目录→重读。"""

from datetime import date

import pandas as pd

from quantlab.data.calendar import parse_calendar_csv
from quantlab.data.catalog import scan_market_catalog
from quantlab.data.research_sets import DatasetSelection, build_research_set
from quantlab.data.research_storage import load_research_set, save_research_set


def test_synthetic_research_workflow_is_reproducible_without_source_directories(imported_pair, tmp_path):
    raw, processed, _ = imported_pair
    selections = tuple(DatasetSelection(record, "none") for record in scan_market_catalog(processed, raw).records)
    calendar = parse_calendar_csv(
        b"date\n2025-01-02\n2025-01-03\n2025-01-06\n2025-01-07\n2025-01-08\n",
        source="合成验收日历，非官方交易日历", coverage_start=date(2025, 1, 1), coverage_end=date(2025, 1, 10),
    )
    original = build_research_set(selections, start=date(2025, 1, 2), end=date(2025, 1, 8), calendar=calendar)
    saved = save_research_set(original, tmp_path / "sets", name="合成 ETF 数据验收，不用于投资决策")
    raw.rename(tmp_path / "moved-raw")
    processed.rename(tmp_path / "moved-processed")
    restored = load_research_set(tmp_path / "sets", saved.set_id).preview
    assert len(restored.alignment.data) == 2 * 5
    assert restored.coverage.observation_count.tolist() == [3, 3]
    assert restored.coverage.missing_count.tolist() == [2, 2]
    assert restored.alignment.common_dates == (date(2025, 1, 3), date(2025, 1, 7))
    assert not restored.alignment.data.loc[lambda d: d.date.eq("2025-01-08"), "has_observation"].any()
    pd.testing.assert_frame_equal(restored.alignment.data, original.alignment.data)
    pd.testing.assert_frame_equal(restored.coverage, original.coverage)
    assert restored.sources == original.sources
    assert restored.preview_sha256 == original.preview_sha256
