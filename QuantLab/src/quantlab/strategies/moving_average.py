import pandas as pd

from quantlab.data.validator import validate_market_data


def generate_dual_moving_average_signals(
    market_data: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    """计算双均线，并在真正交叉发生的当天生成买卖信号。"""
    if (
        not isinstance(short_window, int)
        or not isinstance(long_window, int)
        or short_window <= 0
        or long_window <= 0
    ):
        raise ValueError("均线周期必须是正整数")

    if short_window >= long_window:
        raise ValueError("短期均线周期必须小于长期均线周期")

    validate_market_data(market_data)
    result = market_data.copy()

    if result["symbol"].nunique() != 1:
        raise ValueError("第一阶段双均线策略只支持单只 ETF")

    result["date"] = pd.to_datetime(
        result["date"],
        errors="raise",
    )
    result = result.sort_values("date").reset_index(drop=True)

    short_column = f"sma_{short_window}"
    long_column = f"sma_{long_window}"
    result[short_column] = result["close"].rolling(
        window=short_window,
        min_periods=short_window,
    ).mean()
    result[long_column] = result["close"].rolling(
        window=long_window,
        min_periods=long_window,
    ).mean()

    previous_short = result[short_column].shift(1)
    previous_long = result[long_column].shift(1)
    current_valid = (
        result[short_column].notna()
        & result[long_column].notna()
    )
    previous_valid = (
        previous_short.notna()
        & previous_long.notna()
    )

    buy_signal = (
        current_valid
        & previous_valid
        & (result[short_column] > result[long_column])
        & (previous_short <= previous_long)
    )
    sell_signal = (
        current_valid
        & previous_valid
        & (result[short_column] < result[long_column])
        & (previous_short >= previous_long)
    )

    result["signal"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string",
    )
    result.loc[buy_signal, "signal"] = "BUY"
    result.loc[sell_signal, "signal"] = "SELL"

    return result
