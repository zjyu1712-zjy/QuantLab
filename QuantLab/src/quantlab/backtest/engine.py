from dataclasses import dataclass

import pandas as pd

from quantlab.config.settings import (
    DEFAULT_BACKTEST_CONFIG,
    BacktestConfig,
)
from quantlab.data.validator import validate_market_data


BACKTEST_TRADE_COLUMNS = [
    "trade_id",
    "signal_date",
    "execution_date",
    "side",
    "quantity",
    "open_price",
    "execution_price",
    "notional",
    "commission",
    "tax",
    "cash_after",
    "position_after",
]

BACKTEST_ORDER_COLUMNS = [
    "order_id",
    "signal_date",
    "execution_date",
    "side",
    "status",
    "reason",
    "quantity",
    "trade_id",
]

BACKTEST_EQUITY_COLUMNS = [
    "date",
    "symbol",
    "close",
    "cash",
    "quantity",
    "position_value",
    "equity",
    "net_value",
    "cumulative_return",
    "cumulative_costs",
]


@dataclass(frozen=True)
class BacktestResult:
    """基础回测产生的订单、成交和每日资金曲线。"""

    orders: pd.DataFrame
    trades: pd.DataFrame
    equity_curve: pd.DataFrame


def _calculate_commission(
    notional: float,
    config: BacktestConfig,
) -> float:
    return max(
        notional * config.commission_rate,
        config.minimum_commission,
    )


def _calculate_buy_quantity(
    cash: float,
    execution_price: float,
    config: BacktestConfig,
) -> int:
    lots = int(cash // (execution_price * config.lot_size))
    quantity = lots * config.lot_size

    while quantity > 0:
        notional = execution_price * quantity
        commission = _calculate_commission(notional, config)
        if notional + commission <= cash + 1e-12:
            return quantity
        quantity -= config.lot_size

    return 0


def run_backtest(
    signal_data: pd.DataFrame,
    config: BacktestConfig = DEFAULT_BACKTEST_CONFIG,
) -> BacktestResult:
    """按下一交易日开盘价执行单 ETF、只做多的全仓回测。"""
    validate_market_data(signal_data)

    if "signal" not in signal_data.columns:
        raise ValueError("回测数据缺少 signal 列")

    invalid_signals = set(signal_data["signal"].dropna()) - {
        "BUY",
        "SELL",
    }
    if invalid_signals:
        invalid_text = ", ".join(sorted(map(str, invalid_signals)))
        raise ValueError(f"未知交易信号：{invalid_text}")

    data = signal_data.copy()
    if data["symbol"].nunique() != 1:
        raise ValueError("第一阶段回测只支持单只 ETF")

    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values("date").reset_index(drop=True)
    symbol = str(data.iloc[0]["symbol"])

    cash = float(config.initial_cash)
    quantity = 0
    cumulative_costs = 0.0
    order_records: list[dict[str, object]] = []
    trade_records: list[dict[str, object]] = []
    equity_records: list[dict[str, object]] = []

    for index, row in data.iterrows():
        if index > 0:
            previous_row = data.iloc[index - 1]
            raw_signal = previous_row["signal"]
            signal = None if pd.isna(raw_signal) else str(raw_signal)

            if signal is not None:
                order_id = f"BO{len(order_records) + 1:06d}"
                order = {
                    "order_id": order_id,
                    "signal_date": previous_row["date"],
                    "execution_date": row["date"],
                    "side": signal,
                    "status": "REJECTED",
                    "reason": "",
                    "quantity": 0,
                    "trade_id": pd.NA,
                }

                if signal == "BUY" and quantity > 0:
                    order["reason"] = "已有持仓，忽略重复买入"

                elif signal == "BUY":
                    execution_price = float(row["open"]) * (
                        1 + config.slippage_rate
                    )
                    buy_quantity = _calculate_buy_quantity(
                        cash,
                        execution_price,
                        config,
                    )

                    if buy_quantity == 0:
                        order["reason"] = "资金不足，无法买入一手"
                    else:
                        notional = execution_price * buy_quantity
                        commission = _calculate_commission(
                            notional,
                            config,
                        )
                        cash -= notional + commission
                        quantity = buy_quantity
                        cumulative_costs += commission
                        trade_id = f"BT{len(trade_records) + 1:06d}"
                        trade_records.append(
                            {
                                "trade_id": trade_id,
                                "signal_date": previous_row["date"],
                                "execution_date": row["date"],
                                "side": "BUY",
                                "quantity": buy_quantity,
                                "open_price": float(row["open"]),
                                "execution_price": execution_price,
                                "notional": notional,
                                "commission": commission,
                                "tax": 0.0,
                                "cash_after": cash,
                                "position_after": quantity,
                            }
                        )
                        order.update(
                            {
                                "status": "FILLED",
                                "reason": "已成交",
                                "quantity": buy_quantity,
                                "trade_id": trade_id,
                            }
                        )

                elif signal == "SELL" and quantity == 0:
                    order["reason"] = "当前空仓，无法卖出"

                elif signal == "SELL":
                    execution_price = float(row["open"]) * (
                        1 - config.slippage_rate
                    )
                    notional = execution_price * quantity
                    commission = _calculate_commission(
                        notional,
                        config,
                    )
                    tax = notional * config.sell_tax_rate
                    sell_quantity = quantity
                    cash += notional - commission - tax
                    quantity = 0
                    cumulative_costs += commission + tax
                    trade_id = f"BT{len(trade_records) + 1:06d}"
                    trade_records.append(
                        {
                            "trade_id": trade_id,
                            "signal_date": previous_row["date"],
                            "execution_date": row["date"],
                            "side": "SELL",
                            "quantity": sell_quantity,
                            "open_price": float(row["open"]),
                            "execution_price": execution_price,
                            "notional": notional,
                            "commission": commission,
                            "tax": tax,
                            "cash_after": cash,
                            "position_after": quantity,
                        }
                    )
                    order.update(
                        {
                            "status": "FILLED",
                            "reason": "已成交",
                            "quantity": sell_quantity,
                            "trade_id": trade_id,
                        }
                    )

                order_records.append(order)

        close = float(row["close"])
        position_value = quantity * close
        equity = cash + position_value

        equity_records.append(
            {
                "date": row["date"],
                "symbol": symbol,
                "close": close,
                "cash": cash,
                "quantity": quantity,
                "position_value": position_value,
                "equity": equity,
                "net_value": equity / config.initial_cash,
                "cumulative_return": (
                    equity / config.initial_cash - 1
                ),
                "cumulative_costs": cumulative_costs,
            }
        )

    last_row = data.iloc[-1]
    raw_last_signal = last_row["signal"]
    if not pd.isna(raw_last_signal):
        order_records.append(
            {
                "order_id": f"BO{len(order_records) + 1:06d}",
                "signal_date": last_row["date"],
                "execution_date": pd.NaT,
                "side": str(raw_last_signal),
                "status": "EXPIRED",
                "reason": "没有下一交易日",
                "quantity": 0,
                "trade_id": pd.NA,
            }
        )

    return BacktestResult(
        orders=pd.DataFrame(
            order_records,
            columns=BACKTEST_ORDER_COLUMNS,
        ),
        trades=pd.DataFrame(
            trade_records,
            columns=BACKTEST_TRADE_COLUMNS,
        ),
        equity_curve=pd.DataFrame(
            equity_records,
            columns=BACKTEST_EQUITY_COLUMNS,
        ),
    )
