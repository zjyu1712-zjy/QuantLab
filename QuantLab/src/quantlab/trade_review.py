"""个人真实交易与双均线模型信号的逐笔复盘。"""

from dataclasses import dataclass

import pandas as pd

from quantlab.strategies import generate_dual_moving_average_signals
from quantlab.trades import TradeAnalysis, analyze_trades


SIGNAL_EVENT_COLUMNS = [
    "signal_id",
    "signal_date",
    "expected_execution_date",
    "side",
    "status",
]

ALIGNMENT_COLUMNS = [
    "trade_id",
    "trade_datetime",
    "side",
    "quantity",
    "price",
    "fee",
    "signal_id",
    "signal_date",
    "expected_execution_date",
    "offset_trading_days",
    "timing_status",
]


@dataclass(frozen=True)
class TradeSignalReview:
    """交易分析、信号匹配、行为统计和费用摘要。"""

    analysis: TradeAnalysis
    signal_events: pd.DataFrame
    trade_alignment: pd.DataFrame
    missed_signals: pd.DataFrame
    timing_counts: dict[str, int]
    total_fees: float
    turnover: float
    fee_to_turnover_rate: float
    fee_to_initial_cash_rate: float


def _build_signal_events(
    market_data: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    signals = generate_dual_moving_average_signals(
        market_data,
        short_window,
        long_window,
    )
    next_market_date = signals["date"].shift(-1)
    events = signals.loc[
        signals["signal"].notna(),
        ["date", "signal"],
    ].copy()
    events["expected_execution_date"] = next_market_date.loc[
        events.index
    ]
    events = events.rename(
        columns={"date": "signal_date", "signal": "side"}
    ).reset_index(drop=True)
    events.insert(
        0,
        "signal_id",
        [f"S{index + 1:06d}" for index in range(len(events))],
    )
    events["status"] = events["expected_execution_date"].apply(
        lambda value: "EXPIRED" if pd.isna(value) else "EXECUTABLE"
    )
    return events.loc[:, SIGNAL_EVENT_COLUMNS]


def review_trades_against_signals(
    market_data: pd.DataFrame,
    trade_data: pd.DataFrame,
    short_window: int,
    long_window: int,
    *,
    initial_cash: float,
    tolerance_trading_days: int = 3,
) -> TradeSignalReview:
    """将实际成交与同方向模型信号按最近预期成交日进行匹配。"""
    if (
        not isinstance(tolerance_trading_days, int)
        or tolerance_trading_days < 0
    ):
        raise ValueError("信号匹配容差必须是非负整数")

    analysis = analyze_trades(
        market_data,
        trade_data,
        initial_cash=initial_cash,
    )
    events = _build_signal_events(
        market_data,
        short_window,
        long_window,
    )

    market_dates = (
        pd.to_datetime(market_data["date"], errors="raise")
        .dt.tz_localize(None)
        .dt.normalize()
        .sort_values()
        .drop_duplicates()
        .tolist()
    )
    market_index = {
        pd.Timestamp(date): index
        for index, date in enumerate(market_dates)
    }
    executable = events.loc[
        events["status"] == "EXECUTABLE"
    ].copy()
    matched_signal_ids: set[str] = set()
    alignment_records: list[dict[str, object]] = []

    for trade in analysis.trades.to_dict("records"):
        trade_date = (
            pd.Timestamp(trade["trade_datetime"])
            .tz_localize(None)
            .normalize()
        )
        trade_index = market_index[trade_date]
        same_side = executable.loc[
            executable["side"] == trade["side"]
        ].copy()

        selected: dict[str, object] | None = None
        selected_offset: int | None = None
        if not same_side.empty:
            same_side["_offset"] = same_side[
                "expected_execution_date"
            ].map(
                lambda date: (
                    trade_index
                    - market_index[pd.Timestamp(date).normalize()]
                )
            )
            same_side["_distance"] = same_side["_offset"].abs()
            nearest = same_side.sort_values(
                by=["_distance", "expected_execution_date"],
            ).iloc[0]
            if int(nearest["_distance"]) <= tolerance_trading_days:
                selected = nearest.to_dict()
                selected_offset = int(nearest["_offset"])

        if selected is None:
            timing_status = "NO_MATCH"
            signal_id = pd.NA
            signal_date = pd.NaT
            expected_date = pd.NaT
            offset = pd.NA
        else:
            signal_id = str(selected["signal_id"])
            matched_signal_ids.add(signal_id)
            signal_date = selected["signal_date"]
            expected_date = selected["expected_execution_date"]
            offset = selected_offset
            if selected_offset is not None and selected_offset < 0:
                timing_status = "EARLY"
            elif selected_offset == 0:
                timing_status = "ON_TIME"
            else:
                timing_status = "LATE"

        record = {
            "trade_id": str(trade["trade_id"]),
            "trade_datetime": trade["trade_datetime"],
            "side": str(trade["side"]),
            "quantity": int(trade["quantity"]),
            "price": float(trade["price"]),
            "fee": float(trade["fee"]),
            "signal_id": signal_id,
            "signal_date": signal_date,
            "expected_execution_date": expected_date,
            "offset_trading_days": offset,
            "timing_status": timing_status,
        }
        for optional_column in ["note", "strategy_tag", "account"]:
            if optional_column in trade:
                record[optional_column] = trade[optional_column]
        alignment_records.append(record)

    alignment = pd.DataFrame(alignment_records)
    if alignment.empty:
        alignment = pd.DataFrame(columns=ALIGNMENT_COLUMNS)
    else:
        alignment["offset_trading_days"] = alignment[
            "offset_trading_days"
        ].astype("Int64")

    missed_signals = executable.loc[
        ~executable["signal_id"].isin(matched_signal_ids)
    ].reset_index(drop=True)
    timing_counts = {
        status: int((alignment["timing_status"] == status).sum())
        for status in ["ON_TIME", "EARLY", "LATE", "NO_MATCH"]
    }
    timing_counts["MISSED_SIGNAL"] = len(missed_signals)

    total_fees = float(analysis.trades["fee"].sum())
    turnover = float(
        (
            analysis.trades["quantity"] * analysis.trades["price"]
        ).sum()
    )

    return TradeSignalReview(
        analysis=analysis,
        signal_events=events,
        trade_alignment=alignment,
        missed_signals=missed_signals,
        timing_counts=timing_counts,
        total_fees=total_fees,
        turnover=turnover,
        fee_to_turnover_rate=(
            total_fees / turnover if turnover > 0 else 0.0
        ),
        fee_to_initial_cash_rate=total_fees / float(initial_cash),
    )
