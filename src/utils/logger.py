from __future__ import annotations

import logging


def build_logger(logger_name: str) -> logging.Logger:
    """Create a console logger for scripts and pipeline entrypoints."""
    logger: logging.Logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if len(logger.handlers) == 0:
        handler: logging.Handler = logging.StreamHandler()
        formatter: logging.Formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

