"""Load trained weights and produce inverse-scaled predictions."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .config import Config
from .seed import seed_everything
from .data import load_dataset, Dataset
from .models import build_model


def predict(cfg: Config, weights_path: str):
    """Return (times, ground_truth_close, predicted_close) for the val split."""
    seed_everything(cfg.seed, cfg.deterministic)
    ds: Dataset = load_dataset(cfg.data.path)

    n = int(cfg.data.train_ratio * ds.inputs.shape[0])
    val_x = ds.inputs[n:]
    val_out_times = ds.output_times[n:]
    val_true = ds.original_outputs[n:]

    model = build_model(ds.step_size, ds.nb_features, cfg.model)
    model.load_weights(weights_path)
    pred = model.predict(val_x, verbose=0)
    if pred.ndim == 2:  # (samples, steps) -> add feature axis
        pred = pred[:, :, np.newaxis]

    scaler = MinMaxScaler()
    scaler.fit(ds.original_datas[:, 0].reshape(-1, 1))
    pred_close = scaler.inverse_transform(pred[:, :, 0]).reshape(-1)

    return (val_out_times.reshape(-1),
            val_true[:, :, 0].reshape(-1),
            pred_close)
