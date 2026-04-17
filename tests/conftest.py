import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from crypto_prediction.config import Config, DataConfig
from crypto_prediction.data import build_dataset, save_dataset


@pytest.fixture(scope='session')
def tiny_data_cfg():
    """Small synthetic dataset config — fast enough for unit tests.

    input_steps stays at 256 because the CNN's fixed kernel/stride geometry
    requires it; tail is kept small so the suite stays fast.
    """
    return DataConfig(synthetic=True, tail=12000, input_steps=256,
                      output_steps=16, sliding=False)


@pytest.fixture(scope='session')
def tiny_dataset(tiny_data_cfg):
    return build_dataset(tiny_data_cfg, seed=0)


@pytest.fixture(scope='session')
def tiny_h5(tmp_path_factory, tiny_dataset):
    path = tmp_path_factory.mktemp('data') / 'tiny.h5'
    save_dataset(tiny_dataset, str(path))
    return str(path)


@pytest.fixture
def tiny_config(tiny_h5, tmp_path):
    cfg = Config()
    cfg.seed = 0
    cfg.data.path = tiny_h5
    cfg.data.input_steps = 256
    cfg.data.output_steps = 16
    cfg.data.label_feature = None
    cfg.model.name = 'cnn'
    cfg.model.output_size = 16
    cfg.train.epochs = 1
    cfg.train.batch_size = 4
    cfg.train.output_name = 'test_run'
    cfg.train.weights_dir = str(tmp_path / 'weights')
    cfg.train.result_dir = str(tmp_path / 'result')
    return cfg
