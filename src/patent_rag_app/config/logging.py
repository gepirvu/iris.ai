"""Application logging utilities."""

import logging
import warnings
from logging.config import dictConfig

warnings.filterwarnings(
    "ignore",
    message="pythonjsonlogger.jsonlogger has been moved",
    category=DeprecationWarning,
)

DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.json.JsonFormatter",
            "fmt": "%(asctime)s %(levelname)s %(name)s %(message)s",
        },
        "console": {
            "format": "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "console",
        },
    },
    "root": {
        "handlers": ["default"],
        "level": "INFO",
    },
}


def configure_logging(config: dict | None = None) -> None:
    """Configure logging for the application."""
    dictConfig(config or DEFAULT_LOGGING_CONFIG)


def get_logger(name: str) -> logging.Logger:
    """Retrieve a logger with the given name."""
    return logging.getLogger(name)
