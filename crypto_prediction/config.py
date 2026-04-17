"""Centralized configuration.

A single :class:`Config` dataclass drives data building, training and
inference. Configs are loaded from YAML and can be overridden on the command
line with ``key=value`` pairs (dotted keys for nested sections), e.g.::

    python CNN.py --config configs/cnn.yaml train.epochs=5 seed=123
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any

import yaml


@dataclass
class DataConfig:
    path: str = 'data/bitcoin2015to2017_close.h5'
    csv: str = 'data/bitstampUSD_1-min_data_2012-01-01_to_2017-05-31.csv'
    input_steps: int = 256
    output_steps: int = 16
    tail: int = 1_000_000
    sliding: bool = False
    synthetic: bool = False
    train_ratio: float = 0.8
    label_feature: int | None = None  # None -> keep all label features (CNN)


@dataclass
class ModelConfig:
    name: str = 'cnn'            # cnn | gru | lstm | gru_wf
    units: int = 50
    output_size: int = 16
    dropout: float = 0.2
    l1_reg: float = 0.0


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 8
    output_name: str = 'run'
    weights_dir: str = 'weights'
    result_dir: str = 'result'
    save_best_only: bool = False
    early_stopping_patience: int = 0   # 0 disables
    tensorboard: bool = False
    log_level: str = 'INFO'
    log_file: str | None = None
    use_tf_data: bool = True           # wrap arrays in tf.data for prefetch


_VALID_MODELS = ('cnn', 'gru', 'lstm', 'gru_wf')
_VALID_LOG_LEVELS = ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')


class ConfigValidationError(ValueError):
    """Raised when a :class:`Config` fails validation."""


@dataclass
class Config:
    seed: int = 42
    deterministic: bool = True
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> 'Config':
        cfg = Config()
        _merge(cfg, d)
        return cfg

    def validate(self) -> 'Config':
        """Check ranges and enums. Returns self so calls can be chained."""
        errs: list[str] = []
        d, m, t = self.data, self.model, self.train

        if not (0.0 < d.train_ratio < 1.0):
            errs.append(f'data.train_ratio must be in (0, 1); got {d.train_ratio}')
        for name in ('input_steps', 'output_steps', 'tail'):
            v = getattr(d, name)
            if not isinstance(v, int) or v <= 0:
                errs.append(f'data.{name} must be a positive int; got {v!r}')
        if d.label_feature is not None and d.label_feature < 0:
            errs.append(f'data.label_feature must be >= 0 or null; got {d.label_feature}')

        if m.name not in _VALID_MODELS:
            errs.append(
                f"model.name must be one of {_VALID_MODELS}; got {m.name!r}"
            )
        if m.units <= 0:
            errs.append(f'model.units must be > 0; got {m.units}')
        if m.output_size <= 0:
            errs.append(f'model.output_size must be > 0; got {m.output_size}')
        if not (0.0 <= m.dropout < 1.0):
            errs.append(f'model.dropout must be in [0, 1); got {m.dropout}')
        if m.l1_reg < 0:
            errs.append(f'model.l1_reg must be >= 0; got {m.l1_reg}')

        if t.epochs <= 0:
            errs.append(f'train.epochs must be > 0; got {t.epochs}')
        if t.batch_size <= 0:
            errs.append(f'train.batch_size must be > 0; got {t.batch_size}')
        if t.early_stopping_patience < 0:
            errs.append(
                f'train.early_stopping_patience must be >= 0; got {t.early_stopping_patience}'
            )
        if t.log_level.upper() not in _VALID_LOG_LEVELS:
            errs.append(
                f'train.log_level must be one of {_VALID_LOG_LEVELS}; got {t.log_level!r}'
            )

        if errs:
            raise ConfigValidationError(
                'Invalid configuration:\n  - ' + '\n  - '.join(errs)
            )
        return self


_SECTIONS = {'data': DataConfig, 'model': ModelConfig, 'train': TrainConfig}


def _coerce(current: Any, value: Any) -> Any:
    """Cast a CLI override string to the type of the existing field value."""
    if not isinstance(value, str):
        return value
    if isinstance(current, bool):
        return value.lower() in ('1', 'true', 'yes', 'on')
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if current is None:
        # best effort: try int, then float, else leave as str/None
        if value.lower() in ('none', 'null', ''):
            return None
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                continue
        return value
    return value


def _merge(cfg: Config, d: dict) -> None:
    for key, value in d.items():
        if key in _SECTIONS:
            section = getattr(cfg, key)
            valid = {f.name for f in fields(section)}
            for sk, sv in (value or {}).items():
                if sk not in valid:
                    raise KeyError(f"Unknown config key: {key}.{sk}")
                setattr(section, sk, _coerce(getattr(section, sk), sv))
        elif key in {f.name for f in fields(cfg)}:
            setattr(cfg, key, _coerce(getattr(cfg, key), value))
        else:
            raise KeyError(f"Unknown config key: {key}")


def _apply_overrides(cfg: Config, overrides: list[str]) -> None:
    for ov in overrides:
        if '=' not in ov:
            raise ValueError(f"Override '{ov}' must be key=value")
        key, value = ov.split('=', 1)
        parts = key.split('.')
        if len(parts) == 1:
            _merge(cfg, {parts[0]: value})
        elif len(parts) == 2:
            _merge(cfg, {parts[0]: {parts[1]: value}})
        else:
            raise KeyError(f"Config keys may be at most two levels deep: {key}")


def load_config(path: str | None = None,
                overrides: list[str] | None = None,
                validate: bool = True) -> Config:
    """Load a Config from YAML, apply ``key=value`` overrides, validate."""
    cfg = Config()
    if path:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _merge(cfg, data)
    if overrides:
        _apply_overrides(cfg, overrides)
    if validate:
        cfg.validate()
    return cfg


def parse_cli(default_config: str | None,
              description: str = 'crypto-prediction') -> Config:
    """Standard CLI: ``--config <yaml>`` plus free-form ``key=value`` overrides."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', default=default_config,
                        help='Path to YAML config file')
    parser.add_argument('overrides', nargs='*',
                        help='key=value overrides, e.g. train.epochs=5 seed=7')
    args = parser.parse_args()
    return load_config(args.config, args.overrides)
