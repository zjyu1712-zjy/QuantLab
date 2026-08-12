from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path

from quantlab.data.market_service import MarketDataService
from quantlab.data.providers.ifind import IFindETFProvider


IFIND_HISTORY_RESPONSE = {
    "errorcode": 0,
    "errmsg": "Success",
    "tables": [
        {
            "thscode": "510300.SH",
            "time": ["2025-01-02", "2025-01-03"],
            "table": {
                "open": [3.95, 4.01],
                "high": [4.02, 4.08],
                "low": [3.93, 3.99],
                "close": [4.00, 4.06],
                "volume": [1_200_000, 1_350_000],
            },
        }
    ],
}


class FakeIFindClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    def post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append((path, payload))
        return self.response


def test_ifind_provider_flows_through_service_and_cache(tmp_path: Path) -> None:
    client = FakeIFindClient(IFIND_HISTORY_RESPONSE)
    service = MarketDataService(
        IFindETFProvider(client),
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
    )

    first = service.fetch_etf_history(
        "510300", date(2025, 1, 1), date(2025, 1, 4)
    )
    second = service.fetch_etf_history(
        "510300", date(2025, 1, 1), date(2025, 1, 4)
    )

    assert first.cache_status == "miss"
    assert first.fetched_ranges == (("2025-01-01", "2025-01-04"),)
    assert second.cache_status == "hit"
    assert second.fetched_ranges == ()
    assert client.calls == [
        (
            "/api/v1/cmd_history_quotation",
            {
                "codes": "510300.SH",
                "indicators": "open,high,low,close,volume",
                "startdate": "2025-01-01",
                "enddate": "2025-01-04",
                "functionpara": {
                    "Interval": "D",
                    "Fill": "Omit",
                    "CPS": "1",
                },
            },
        )
    ]
    assert second.data.equals(first.data)

    metadata = first.metadata
    assert metadata.symbol == "510300"
    assert metadata.provider == "ifind"
    assert metadata.period == "daily"
    assert metadata.adjustment == "none"
    assert metadata.start_date == "2025-01-02"
    assert metadata.end_date == "2025-01-03"
    assert metadata.requested_start_date == "2025-01-01"
    assert metadata.requested_end_date == "2025-01-04"
    assert metadata.requested_ranges == (("2025-01-01", "2025-01-04"),)
    assert datetime.fromisoformat(metadata.fetched_at).tzinfo is not None
    assert metadata.data_version
    assert metadata.row_count == 2
    assert Path(metadata.processed_path).is_file()
    assert second.metadata == metadata

    metadata_paths = list(
        (tmp_path / "raw" / "ifind").glob("*/provider_metadata.json")
    )
    assert len(metadata_paths) == 1
    assert json.loads(metadata_paths[0].read_text(encoding="utf-8")) == {
        "symbol": "510300",
        "provider": "ifind",
        "period": "daily",
        "adjustment": "none",
        "start_date": "2025-01-02",
        "end_date": "2025-01-03",
        "fetched_at": metadata.fetched_at,
        "data_version": metadata.data_version,
        "row_count": 2,
        "processed_path": metadata.processed_path,
        "requested_start_date": "2025-01-01",
        "requested_end_date": "2025-01-04",
        "requested_ranges": [["2025-01-01", "2025-01-04"]],
    }
