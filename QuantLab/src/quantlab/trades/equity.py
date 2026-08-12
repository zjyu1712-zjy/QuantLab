from math import isfinite

import pandas as pd

from quantlab.data.trade_loader import TRADE_TIMEZONE
from quantlab.data.trade_validator import validate_trade_data
from quantlab.data.validator import validate_market_data


EQUITY_CURVE_COLUMNS = [
    "date",
    "symbol",
    "close",
    "cash",
    "quantity",
    "position_value",
    "equity",
    "net_value",
    "cumulative_return",
    "cumulative_fees",
]


def build_trade_equity_curve(
    market_data: pd.DataFrame,
    trade_data: pd.DataFrame,
    initial_cash: float,
) -> pd.DataFrame:
    """按真实成交和每日收盘价生成个人交易资金曲线。"""
    initial_cash = float(initial_cash)
    if not isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("初始资金必须大于 0")

    validate_market_data(market_data)
    validate_trade_data(trade_data)

    market = market_data.copy()
    market["date"] = pd.to_datetime(
        market["date"],
        errors="raise",
    ).dt.normalize()
    market = market.sort_values("date").reset_index(drop=True)

    market_symbols = market["symbol"].astype(str).unique()
    trade_symbols = trade_data["symbol"].astype(str).unique()

    if len(market_symbols) != 1 or len(trade_symbols) != 1:
        raise ValueError("第一阶段资金曲线只支持单只 ETF")

    symbol = str(market_symbols[0])
    if symbol != str(trade_symbols[0]):
        raise ValueError("交易记录与行情数据的证券代码不一致")

    trades = trade_data.copy().sort_values(
        by=["trade_datetime", "trade_id"],
    )
    if trades["trade_datetime"].dt.tz is None:
        raise ValueError("交易时间必须包含时区")

    trades["_trade_date"] = (
        trades["trade_datetime"]
        .dt.tz_convert(TRADE_TIMEZONE)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    market_dates = set(market["date"])
    invalid_dates = sorted(
        set(trades["_trade_date"]) - market_dates
    )
    if invalid_dates:
        invalid_text = ", ".join(
            date.strftime("%Y-%m-%d")
            for date in invalid_dates
        )
        raise ValueError(f"交易日期缺少对应行情：{invalid_text}")

    trades_by_date = {
        date: group.to_dict("records")
        for date, group in trades.groupby("_trade_date", sort=True)
    }

    cash = initial_cash
    quantity = 0
    cumulative_fees = 0.0
    curve_records: list[dict[str, object]] = []

    for market_row in market.to_dict("records"):
        date = market_row["date"]

        for trade in trades_by_date.get(date, []):
            trade_quantity = int(trade["quantity"])
            trade_value = float(trade["price"]) * trade_quantity
            fee = float(trade["fee"])

            if trade["side"] == "BUY":
                required_cash = trade_value + fee
                if required_cash > cash + 1e-12:
                    raise ValueError(
                        f"{trade['trade_id']} 买入所需资金 "
                        f"{required_cash:.2f} 超过可用现金 "
                        f"{cash:.2f}"
                    )
                cash -= required_cash
                quantity += trade_quantity
            else:
                if trade_quantity > quantity:
                    raise ValueError(
                        f"{trade['trade_id']} 卖出数量超过可用持仓"
                    )
                cash += trade_value - fee
                quantity -= trade_quantity

            cumulative_fees += fee

        close = float(market_row["close"])
        position_value = quantity * close
        equity = cash + position_value

        curve_records.append(
            {
                "date": date,
                "symbol": symbol,
                "close": close,
                "cash": cash,
                "quantity": quantity,
                "position_value": position_value,
                "equity": equity,
                "net_value": equity / initial_cash,
                "cumulative_return": equity / initial_cash - 1,
                "cumulative_fees": cumulative_fees,
            }
        )

    return pd.DataFrame(
        curve_records,
        columns=EQUITY_CURVE_COLUMNS,
    )
