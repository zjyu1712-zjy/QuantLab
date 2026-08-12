"""QuantLab 的本地日志初始化。"""

import logging
from pathlib import Path


LOGGER_NAME = "quantlab"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(
    log_directory: str | Path = "logs",
    *,
    level: int = logging.INFO,
) -> logging.Logger:
    """配置控制台和本地文件日志，并避免重复添加处理器。"""
    directory = Path(log_directory)
    directory.mkdir(parents=True, exist_ok=True)
    log_file = (directory / "quantlab.log").resolve()

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)

    has_file_handler = any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == log_file
        for handler in logger.handlers
    )

    if not has_file_handler:
        file_handler = logging.FileHandler(
            log_file,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    has_console_handler = any(
        getattr(handler, "_quantlab_console", False)
        for handler in logger.handlers
    )

    if not has_console_handler:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler._quantlab_console = True
        logger.addHandler(console_handler)

    return logger
