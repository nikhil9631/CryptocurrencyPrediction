"""Process-level runtime helpers: GPU setup, directory creation, timing."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager

from .logging import get_logger

log = get_logger('runtime')


def configure_gpu() -> None:
    """Enable memory growth on each visible GPU. Safe on CPU-only machines."""
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        log.info('No GPU detected; running on CPU.')
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs are initialized.
            log.warning('Could not enable memory growth on %s: %s', gpu, e)
    log.info('Configured %d GPU(s) with memory growth.', len(gpus))


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


@contextmanager
def timed(label: str):
    """Context manager that logs elapsed wall time at INFO."""
    start = time.perf_counter()
    try:
        yield
    finally:
        log.info('%s completed in %.2fs', label, time.perf_counter() - start)
