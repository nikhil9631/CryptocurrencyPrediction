"""Config-driven training entry point shared by all model scripts."""
from __future__ import annotations

import os
import json

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

from .config import Config, parse_cli
from .seed import seed_everything
from .data import load_dataset, split
from .models import build_model


def _configure_gpu() -> None:
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def _ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _callbacks(cfg: Config):
    t = cfg.train
    ckpt = os.path.join(
        t.weights_dir, t.output_name + '-{epoch:02d}-{val_loss:.5f}.weights.h5'
    )
    cbs = [
        CSVLogger(os.path.join(t.result_dir, t.output_name + '.csv'), append=True),
        ModelCheckpoint(ckpt, monitor='val_loss', mode='min', verbose=1,
                        save_best_only=t.save_best_only, save_weights_only=True),
    ]
    if t.early_stopping_patience > 0:
        cbs.append(EarlyStopping(monitor='val_loss', mode='min',
                                 patience=t.early_stopping_patience))
    return cbs


def run(cfg: Config, verbose: int = 1):
    """Train ``cfg.model.name`` on ``cfg.data.path``. Returns the Keras History."""
    seed_everything(cfg.seed, cfg.deterministic)
    _configure_gpu()
    _ensure_dirs(cfg.train.weights_dir, cfg.train.result_dir)

    ds = load_dataset(cfg.data.path)
    train_x, train_y, val_x, val_y = split(
        ds, cfg.data.train_ratio, cfg.data.label_feature
    )

    model = build_model(ds.step_size, ds.nb_features, cfg.model)
    if verbose:
        model.summary()

    # Persist the resolved config alongside results for provenance.
    with open(os.path.join(cfg.train.result_dir,
                           cfg.train.output_name + '.config.json'), 'w') as f:
        json.dump(cfg.to_dict(), f, indent=2)

    history = model.fit(
        train_x, train_y,
        batch_size=cfg.train.batch_size,
        epochs=cfg.train.epochs,
        validation_data=(val_x, val_y),
        verbose=verbose,
        callbacks=_callbacks(cfg),
    )
    return history


def main(default_config: str) -> None:
    cfg = parse_cli(default_config)
    run(cfg)
