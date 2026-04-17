"""Structured logging for the package.

Use :func:`get_logger` from any module — the root ``crypto_prediction`` logger
is configured the first time :func:`configure_logging` is called (or lazily on
first ``get_logger`` call with default settings). A single config call at
process start is enough; later calls reconfigure handlers.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_ROOT = 'crypto_prediction'
_CONFIGURED = False

_DEFAULT_FMT = '%(asctime)s %(levelname)-5s %(name)s: %(message)s'
_DEFAULT_DATEFMT = '%Y-%m-%d %H:%M:%S'


def configure_logging(level: str | int = 'INFO',
                      log_file: Optional[str] = None,
                      fmt: str = _DEFAULT_FMT) -> None:
    """Set up the package root logger. Idempotent — safe to call multiple times."""
    global _CONFIGURED
    logger = logging.getLogger(_ROOT)
    logger.setLevel(level)
    # Clear any pre-existing handlers so reconfiguration replaces cleanly.
    for h in list(logger.handlers):
        logger.removeHandler(h)

    formatter = logging.Formatter(fmt, _DEFAULT_DATEFMT)
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger. Auto-configures with defaults on first use."""
    if not _CONFIGURED:
        configure_logging(os.environ.get('CRYPTO_PREDICTION_LOG_LEVEL', 'INFO'))
    if not name.startswith(_ROOT):
        name = f'{_ROOT}.{name}'
    return logging.getLogger(name)
