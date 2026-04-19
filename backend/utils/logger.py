"""
backend/utils/logger.py
────────────────────────
Structured logger with file + console handlers.
"""
import logging
import sys
from pathlib import Path

from config.settings import settings


def get_logger(name: str = "surveillance") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — force UTF-8 so Windows cp1252 never chokes on Unicode
    import io
    try:
        utf8_stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except AttributeError:
        utf8_stream = sys.stdout   # already wrapped (pytest, etc.)
    ch = logging.StreamHandler(utf8_stream)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = get_logger()
