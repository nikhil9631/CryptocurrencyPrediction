import pytest
import yaml

from crypto_prediction.config import Config, load_config


def test_defaults():
    cfg = Config()
    assert cfg.seed == 42
    assert cfg.data.train_ratio == 0.8
    assert cfg.model.name == 'cnn'
    assert cfg.train.batch_size == 8


def test_yaml_roundtrip(tmp_path):
    p = tmp_path / 'cfg.yaml'
    p.write_text(yaml.safe_dump({
        'seed': 7,
        'model': {'name': 'lstm', 'units': 99},
        'train': {'epochs': 3},
    }))
    cfg = load_config(str(p))
    assert cfg.seed == 7
    assert cfg.model.name == 'lstm'
    assert cfg.model.units == 99
    assert cfg.train.epochs == 3
    # untouched values keep defaults
    assert cfg.data.train_ratio == 0.8


def test_cli_override_coercion():
    cfg = load_config(None, ['seed=11', 'train.epochs=2',
                             'data.synthetic=true', 'model.l1_reg=0.01'])
    assert cfg.seed == 11 and isinstance(cfg.seed, int)
    assert cfg.train.epochs == 2
    assert cfg.data.synthetic is True
    assert abs(cfg.model.l1_reg - 0.01) < 1e-9


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        load_config(None, ['train.nope=1'])


def test_to_dict_from_dict():
    cfg = Config()
    cfg.model.name = 'gru'
    d = cfg.to_dict()
    cfg2 = Config.from_dict(d)
    assert cfg2.model.name == 'gru'
    assert cfg2.to_dict() == d
