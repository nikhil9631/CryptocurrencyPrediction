import glob
import os
import json

import numpy as np

from crypto_prediction import train as train_mod
from crypto_prediction.inference import predict


def test_train_run_produces_artifacts(tiny_config):
    history = train_mod.run(tiny_config, verbose=0)
    assert 'loss' in history.history
    assert 'val_loss' in history.history

    # config snapshot written for provenance
    cfg_json = os.path.join(tiny_config.train.result_dir,
                            tiny_config.train.output_name + '.config.json')
    assert os.path.exists(cfg_json)
    saved = json.load(open(cfg_json))
    assert saved['model']['name'] == 'cnn'

    # csv log + at least one checkpoint
    assert os.path.exists(os.path.join(tiny_config.train.result_dir,
                                       tiny_config.train.output_name + '.csv'))
    ckpts = glob.glob(os.path.join(tiny_config.train.weights_dir, '*.weights.h5'))
    assert ckpts, 'expected a checkpoint file'

    # inference round-trip on the same config + checkpoint
    times, truth, pred = predict(tiny_config, ckpts[0])
    assert times.shape == truth.shape == pred.shape
    assert truth.ndim == 1


def test_train_is_reproducible(tiny_config):
    h1 = train_mod.run(tiny_config, verbose=0).history['loss']
    h2 = train_mod.run(tiny_config, verbose=0).history['loss']
    np.testing.assert_allclose(h1, h2, rtol=1e-5, atol=1e-6)
