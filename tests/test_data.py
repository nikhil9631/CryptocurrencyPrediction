import numpy as np
import pytest

from PastSampler import PastSampler
from crypto_prediction.config import DataConfig
from crypto_prediction.data import build_dataset, save_dataset, load_dataset, split


def test_past_sampler_tumbling_shapes():
    A = np.arange(40, dtype=float).reshape(-1, 1, 1)
    X, Y = PastSampler(5, 3, sliding_window=False).transform(A)
    assert X.shape == (5, 5, 1)
    assert Y.shape == (5, 3, 1)
    # first sample's target is the next 3 values after the first 5
    np.testing.assert_array_equal(Y[0, :, 0], np.array([5.0, 6.0, 7.0]))


def test_past_sampler_sliding_count():
    A = np.arange(20, dtype=float).reshape(-1, 1, 1)
    X, Y = PastSampler(5, 3, sliding_window=True).transform(A)
    assert X.shape[0] == 20 - (5 + 3) + 1
    assert Y.shape[1] == 3


def test_build_dataset_shapes_and_scaling(tiny_data_cfg, tiny_dataset):
    ds = tiny_dataset
    assert ds.inputs.ndim == 3
    assert ds.inputs.shape[1] == tiny_data_cfg.input_steps
    assert ds.outputs.shape[1] == tiny_data_cfg.output_steps
    assert ds.inputs.shape[2] == 1
    # MinMax scaled into [0, 1]
    assert ds.inputs.min() >= 0.0 and ds.inputs.max() <= 1.0
    assert ds.original_datas.shape == (tiny_data_cfg.tail, 1)


def test_build_dataset_is_seeded(tiny_data_cfg):
    a = build_dataset(tiny_data_cfg, seed=123)
    b = build_dataset(tiny_data_cfg, seed=123)
    np.testing.assert_array_equal(a.inputs, b.inputs)
    np.testing.assert_array_equal(a.outputs, b.outputs)


def test_save_load_roundtrip(tmp_path, tiny_dataset):
    p = tmp_path / 'ds.h5'
    save_dataset(tiny_dataset, str(p))
    loaded = load_dataset(str(p))
    np.testing.assert_array_equal(loaded.inputs, tiny_dataset.inputs)
    np.testing.assert_array_equal(loaded.original_outputs,
                                  tiny_dataset.original_outputs)


def test_load_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_dataset('does/not/exist.h5')


def test_split_ratio_and_label_feature(tiny_dataset):
    tx, ty, vx, vy = split(tiny_dataset, ratio=0.75, label_feature=None)
    n = tiny_dataset.inputs.shape[0]
    assert tx.shape[0] == int(0.75 * n)
    assert tx.shape[0] + vx.shape[0] == n
    assert ty.ndim == 3

    _, ty2, _, vy2 = split(tiny_dataset, ratio=0.75, label_feature=0)
    assert ty2.ndim == 2
    assert ty2.shape[1] == tiny_dataset.outputs.shape[1]
