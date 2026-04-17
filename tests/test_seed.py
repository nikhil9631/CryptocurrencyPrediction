import numpy as np

from crypto_prediction.config import ModelConfig
from crypto_prediction.seed import seed_everything
from crypto_prediction.models import build_model


def _init_weights(seed):
    seed_everything(seed, deterministic_tf=True)
    m = build_model(64, 1, ModelConfig(name='lstm', units=8, output_size=8))
    return [w.copy() for w in m.get_weights()]


def test_seed_everything_makes_weight_init_deterministic():
    a = _init_weights(7)
    b = _init_weights(7)
    for wa, wb in zip(a, b):
        np.testing.assert_array_equal(wa, wb)


def test_different_seeds_differ():
    a = _init_weights(1)
    b = _init_weights(2)
    assert any(not np.array_equal(wa, wb) for wa, wb in zip(a, b))
