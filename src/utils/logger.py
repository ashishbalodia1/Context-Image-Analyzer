"""
Centralized logger for City Guardian system.
Configures structured logging with file + console output.
"""
import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger instance.

    Args:
        name: Logger name (usually __name__)
        level: Logging level

    Returns:
        Configured Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Avoid duplicate handlers

    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(LOG_DIR / "city_guardian.log", encoding="utf-8")
    fh.setFormatter(_formatter)
    logger.addHandler(fh)

    return logger
