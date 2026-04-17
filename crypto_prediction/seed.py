"""Reproducibility helpers."""
import os
import random
import numpy as np


def seed_everything(seed: int, deterministic_tf: bool = True) -> None:
    """Seed python, numpy and tensorflow RNGs.

    When ``deterministic_tf`` is True, also enables TF's deterministic op
    implementations so repeated runs on the same hardware produce identical
    results (at some performance cost).
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Import lazily so data-only utilities don't pull in TF.
    import tensorflow as tf
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        pass
    if deterministic_tf:
        os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError:
            pass
