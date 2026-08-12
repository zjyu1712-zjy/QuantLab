"""第一阶段的买入并持有基准。"""

import pandas as pd

from quantlab.backtest.analysis import (
    BacktestAnalysis,
    analyze_backtest_result,
)
from quantlab.backtest.engine import (
    BACKTEST_EQUITY_COLUMNS,
    BACKTEST_ORDER_COLUMNS,
    BACKTEST_TRADE_COLUMNS,
    BacktestResult,
    _calculate_buy_quantity,
    _calculate_commission,
)
from quantlab.config.settings import (
    DEFAULT_ANALYTICS_CONFIG,
    DEFAULT_BACKTEST_CONFIG,
    AnalyticsConfig,
    BacktestConfig,
)
from quantlab.data.validator import validate_market_data


def run_buy_and_hold(
    market_data: pd.DataFrame,
    *,
    backtest_config: BacktestConfig = DEFAULT_BACKTEST_CONFIG,
    analytics_config: AnalyticsConfig = DEFAULT_ANALYTICS_CONFIG,
) -> BacktestAnalysis:
    """首日开盘全仓买入，并持有至区间结束。"""
    validate_market_data(market_data)
    data = market_data.copy()

    if data["symbol"].nunique() != 1:
        raise ValueError("第一阶段买入持有基准只支持单只 ETF")

    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values("date").reset_index(drop=True)
    first_row = data.iloc[0]
    symbol = str(first_row["symbol"])

    execution_price = float(first_row["open"]) * (
        1 + backtest_config.slippage_rate
    )
    quantity = _calculate_buy_quantity(
        backtest_config.initial_cash,
        execution_price,
        backtest_config,
    )

    order_status = "FILLED" if quantity > 0 else "REJECTED"
    order_reason = (
        "基准首日建仓"
        if quantity > 0
        else "资金不足，无法买入一手"
    )
    trade_id: object = (
        "BH000001" if quantity > 0 else pd.NA
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": "BHO000001",
                "signal_date": first_row["date"],
                "execution_date": first_row["date"],
                "side": "BUY",
                "status": order_status,
                "reason": order_reason,
                "quantity": quantity,
                "trade_id": trade_id,
            }
        ],
        columns=BACKTEST_ORDER_COLUMNS,
    )

    cash = float(backtest_config.initial_cash)
    commission = 0.0
    trades = pd.DataFrame(columns=BACKTEST_TRADE_COLUMNS)
    if quantity > 0:
        notional = execution_price * quantity
        commission = _calculate_commission(
            notional,
            backtest_config,
        )
        cash -= notional + commission
        trades = pd.DataFrame(
            [
                {
                    "trade_id": trade_id,
                    "signal_date": first_row["date"],
                    "execution_date": first_row["date"],
                    "side": "BUY",
                    "quantity": quantity,
                    "open_price": float(first_row["open"]),
                    "execution_price": execution_price,
                    "notional": notional,
                    "commission": commission,
                    "tax": 0.0,
                    "cash_after": cash,
                    "position_after": quantity,
                }
            ],
            columns=BACKTEST_TRADE_COLUMNS,
        )

    equity_records: list[dict[str, object]] = []
    for row in data.itertuples(index=False):
        position_value = quantity * float(row.close)
        equity = cash + position_value
        equity_records.append(
            {
                "date": row.date,
                "symbol": symbol,
                "close": float(row.close),
                "cash": cash,
                "quantity": quantity,
                "position_value": position_value,
                "equity": equity,
                "net_value": equity / backtest_config.initial_cash,
                "cumulative_return": (
                    equity / backtest_config.initial_cash - 1
                ),
                "cumulative_costs": commission,
            }
        )

    backtest = BacktestResult(
        orders=orders,
        trades=trades,
        equity_curve=pd.DataFrame(
            equity_records,
            columns=BACKTEST_EQUITY_COLUMNS,
        ),
    )
    return analyze_backtest_result(
        backtest,
        analytics_config=analytics_config,
    )
