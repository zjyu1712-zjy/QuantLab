import logging

import pytest

from quantlab.config.logging import configure_logging
from quantlab.config.settings import (
    DEFAULT_ANALYTICS_CONFIG,
    DEFAULT_BACKTEST_CONFIG,
    AnalyticsConfig,
    BacktestConfig,
)


def test_default_analytics_config() -> None:
    assert DEFAULT_ANALYTICS_CONFIG.periods_per_year == 252
    assert DEFAULT_ANALYTICS_CONFIG.annual_risk_free_rate == 0.0


def test_default_backtest_config() -> None:
    assert DEFAULT_BACKTEST_CONFIG.initial_cash == 100_000.0
    assert DEFAULT_BACKTEST_CONFIG.commission_rate == 0.0003
    assert DEFAULT_BACKTEST_CONFIG.slippage_rate == 0.0001
    assert DEFAULT_BACKTEST_CONFIG.lot_size == 100


def test_analytics_config_rejects_invalid_periods() -> None:
    with pytest.raises(ValueError, match="每年期数必须大于0"):
        AnalyticsConfig(periods_per_year=0)


def test_backtest_config_rejects_negative_cost() -> None:
    with pytest.raises(ValueError, match="手续费率不能为负数"):
        BacktestConfig(commission_rate=-0.01)


def test_configure_logging_creates_log_file(tmp_path) -> None:
    logger = configure_logging(tmp_path)
    logger.info("QuantLab logging test")

    for handler in logger.handlers:
        handler.flush()

    log_file = tmp_path / "quantlab.log"

    assert log_file.exists()
    assert "QuantLab logging test" in log_file.read_text(encoding="utf-8")
    assert logger.level == logging.INFO
