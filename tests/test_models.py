import numpy as np
import pytest

from crypto_prediction.config import ModelConfig
from crypto_prediction.models import build_model, MODEL_BUILDERS


@pytest.mark.parametrize('name,step', [('cnn', 256), ('gru', 64), ('lstm', 64)])
def test_forward_pass_shape(name, step):
    feat = 1
    mc = ModelConfig(name=name, units=8, output_size=8, dropout=0.1)
    model = build_model(step, feat, mc)
    x = np.random.rand(2, step, feat).astype('float32')
    y = model.predict(x, verbose=0)
    assert y.shape[0] == 2
    if name == 'cnn':
        assert y.ndim == 3 and y.shape[-1] == feat
    else:
        assert y.shape == (2, mc.output_size)


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        build_model(64, 1, ModelConfig(name='transformer'))


def test_registry_complete():
    assert set(MODEL_BUILDERS) == {'cnn', 'gru', 'lstm', 'gru_wf'}
