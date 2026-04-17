"""Dataset construction, persistence and loading."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from PastSampler import PastSampler
from .config import DataConfig
from .logging import get_logger

if TYPE_CHECKING:
    import tensorflow as tf

log = get_logger('data')

H5_KEYS = (
    'inputs', 'outputs', 'input_times', 'output_times',
    'original_inputs', 'original_outputs', 'original_datas',
)

REQUIRED_CSV_COLUMNS = ('Close', 'Timestamp')


class DatasetError(RuntimeError):
    """Dataset is malformed, corrupt, or missing required fields."""


@dataclass
class Dataset:
    inputs: np.ndarray
    outputs: np.ndarray
    input_times: np.ndarray
    output_times: np.ndarray
    original_inputs: np.ndarray
    original_outputs: np.ndarray
    original_datas: np.ndarray

    @property
    def step_size(self) -> int:
        return self.inputs.shape[1]

    @property
    def nb_features(self) -> int:
        return self.inputs.shape[2]

    @property
    def n_samples(self) -> int:
        return self.inputs.shape[0]


def _load_prices(cfg: DataConfig, rng: np.random.Generator):
    """Return (prices, timestamps). Synthetic if requested, else real CSV.

    Unlike a permissive fallback, missing CSVs raise here when
    ``cfg.synthetic=False`` — silent synthetic substitution would cause the
    caller to train on fake data without knowing.
    """
    if cfg.synthetic:
        log.info('Generating synthetic random-walk series (n=%d).', cfg.tail)
        n = cfg.tail
        prices = 1000 + np.cumsum(rng.normal(0, 5, size=n))
        times = np.arange(n, dtype=np.int64) * 60 + 1_325_376_000
        return prices.astype(np.float32), times

    if not os.path.exists(cfg.csv):
        raise FileNotFoundError(
            f"CSV '{cfg.csv}' not found. Either point data.csv at a real file "
            f"or pass data.synthetic=true to opt into synthetic data."
        )

    log.info('Loading prices from %s (tail=%d).', cfg.csv, cfg.tail)
    df = pd.read_csv(cfg.csv).dropna()
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(
            f"CSV '{cfg.csv}' is missing required columns: {missing}. "
            f"Need {list(REQUIRED_CSV_COLUMNS)}."
        )
    df = df.tail(cfg.tail)
    if len(df) < cfg.input_steps + cfg.output_steps:
        raise DatasetError(
            f"CSV has {len(df)} rows after dropna/tail, but "
            f"input_steps+output_steps={cfg.input_steps + cfg.output_steps} are needed."
        )
    return (df['Close'].to_numpy(dtype=np.float32),
            df['Timestamp'].to_numpy(dtype=np.int64))


def build_dataset(cfg: DataConfig, seed: int = 42) -> Dataset:
    """Window raw prices into model-ready arrays (does not write to disk)."""
    rng = np.random.default_rng(seed)
    prices, times = _load_prices(cfg, rng)

    A = prices.reshape(-1, 1, 1)
    T = times.reshape(-1, 1, 1)

    sampler = PastSampler(cfg.input_steps, cfg.output_steps,
                          sliding_window=cfg.sliding)
    original_inputs, original_outputs = sampler.transform(A)
    input_times, output_times = sampler.transform(T)

    if original_inputs.shape[0] == 0:
        raise DatasetError(
            f"Windowing produced 0 samples. input_steps={cfg.input_steps} "
            f"output_steps={cfg.output_steps} len(prices)={len(prices)} "
            f"sliding={cfg.sliding}."
        )

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1, 1)).reshape(-1, 1, 1)
    inputs, outputs = sampler.transform(scaled)

    log.info('Built dataset: inputs=%s outputs=%s', inputs.shape, outputs.shape)
    return Dataset(
        inputs=inputs, outputs=outputs,
        input_times=input_times, output_times=output_times,
        original_inputs=original_inputs, original_outputs=original_outputs,
        original_datas=prices.reshape(-1, 1),
    )


def save_dataset(ds: Dataset, path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with h5py.File(path, 'w') as f:
        for k in H5_KEYS:
            f.create_dataset(k, data=getattr(ds, k))
    log.info('Wrote dataset to %s', path)


def load_dataset(path: str) -> Dataset:
    """Load an HDF5 dataset into memory. See :func:`stream_batches` for large
    files that don't fit in RAM."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset '{path}' not found. Generate it with "
            f"`python DataProcessor.py` (see README)."
        )
    try:
        with h5py.File(path, 'r') as f:
            missing = [k for k in H5_KEYS if k not in f]
            if missing:
                raise DatasetError(
                    f"HDF5 '{path}' is missing required keys: {missing}. "
                    f"This usually means it was built by an older version; "
                    f"regenerate with DataProcessor.py."
                )
            arrays = {k: f[k][:] for k in H5_KEYS}
    except OSError as e:
        raise DatasetError(f"Could not open '{path}' as HDF5: {e}") from e
    return Dataset(**arrays)


def split(ds: Dataset, ratio: float, label_feature: int | None):
    """Return (train_x, train_y, val_x, val_y).

    ``label_feature=None`` keeps the full (samples, steps, features) label
    tensor (used by CNN). An int selects a single feature column (GRU/LSTM
    predict close only). Raises if the split would produce an empty side.
    """
    n = int(ratio * ds.n_samples)
    if n <= 0 or n >= ds.n_samples:
        raise DatasetError(
            f"Split ratio {ratio} on {ds.n_samples} samples yields "
            f"train={n}, val={ds.n_samples - n}; both sides must be non-empty. "
            f"Either lower data.train_ratio or build a larger dataset."
        )
    if label_feature is not None and not (0 <= label_feature < ds.outputs.shape[-1]):
        raise DatasetError(
            f"label_feature={label_feature} is out of range for outputs with "
            f"{ds.outputs.shape[-1]} features."
        )
    labels = ds.outputs if label_feature is None else ds.outputs[:, :, label_feature]
    return ds.inputs[:n], labels[:n], ds.inputs[n:], labels[n:]


def to_tf_dataset(x: np.ndarray, y: np.ndarray, *, batch_size: int,
                  shuffle: bool = False, seed: int = 42) -> 'tf.data.Dataset':
    """Wrap numpy arrays in a tf.data pipeline with prefetch (and optional
    shuffle). Gives a measurable training-throughput win over feeding raw
    numpy to ``model.fit``."""
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 10_000), seed=seed,
                        reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def stream_batches(path: str, *, keys: tuple[str, str] = ('inputs', 'outputs'),
                   batch_size: int = 32, label_feature: int | None = None,
                   start: int = 0, stop: int | None = None):
    """Generator yielding ``(x_batch, y_batch)`` from an HDF5 file without
    loading the whole thing into memory. Useful when the dataset exceeds RAM —
    pair with :func:`tf.data.Dataset.from_generator` for training at scale.

    ``start``/``stop`` define a sample-index window (e.g. for train/val split).
    """
    x_key, y_key = keys
    with h5py.File(path, 'r') as f:
        for k in keys:
            if k not in f:
                raise DatasetError(f"Key '{k}' not in '{path}'")
        n = f[x_key].shape[0] if stop is None else min(stop, f[x_key].shape[0])
        for i in range(start, n, batch_size):
            j = min(i + batch_size, n)
            x = f[x_key][i:j]
            y = f[y_key][i:j]
            if label_feature is not None:
                y = y[:, :, label_feature]
            yield x, y
