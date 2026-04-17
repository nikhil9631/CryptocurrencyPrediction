"""Model factories. Each returns a compiled ``tf.keras`` model."""
from __future__ import annotations

from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Activation, Conv1D, GRU, LSTM, LeakyReLU,
    MaxPooling1D,
)

from .config import ModelConfig


def build_cnn(step_size: int, nb_features: int, cfg: ModelConfig) -> Sequential:
    model = Sequential([
        Input(shape=(step_size, nb_features)),
        Conv1D(filters=8, kernel_size=20, strides=3, activation='relu'),
        Dropout(0.5),
        Conv1D(filters=nb_features, kernel_size=16, strides=4),
    ])
    model.compile(loss='mse', optimizer='adam')
    return model


def build_gru(step_size: int, nb_features: int, cfg: ModelConfig) -> Sequential:
    model = Sequential([
        Input(shape=(step_size, nb_features)),
        GRU(units=cfg.units, return_sequences=False),
        Activation('tanh'),
        Dropout(cfg.dropout),
        Dense(cfg.output_size),
        Activation('relu'),
    ])
    model.compile(loss='mse', optimizer='adam')
    return model


def build_lstm(step_size: int, nb_features: int, cfg: ModelConfig) -> Sequential:
    reg = regularizers.l1(cfg.l1_reg) if cfg.l1_reg > 0 else None
    model = Sequential([
        Input(shape=(step_size, nb_features)),
        LSTM(units=cfg.units, activity_regularizer=reg, return_sequences=False),
        Activation('tanh'),
        Dropout(cfg.dropout),
        Dense(cfg.output_size),
        LeakyReLU(),
    ])
    model.compile(loss='mse', optimizer='adam')
    return model


def build_gru_wf(step_size: int, nb_features: int, cfg: ModelConfig) -> Sequential:
    model = Sequential([
        Input(shape=(step_size, nb_features)),
        GRU(units=cfg.units, return_sequences=True),
        Activation('tanh'),
        Dropout(cfg.dropout),
        MaxPooling1D(pool_size=16),
        Dense(1),
        LeakyReLU(),
    ])
    model.compile(loss='mse', optimizer='adam')
    return model


MODEL_BUILDERS = {
    'cnn': build_cnn,
    'gru': build_gru,
    'lstm': build_lstm,
    'gru_wf': build_gru_wf,
}


def build_model(step_size: int, nb_features: int, cfg: ModelConfig) -> Sequential:
    try:
        builder = MODEL_BUILDERS[cfg.name]
    except KeyError as e:
        raise ValueError(
            f"Unknown model '{cfg.name}'. Choose from {sorted(MODEL_BUILDERS)}."
        ) from e
    return builder(step_size, nb_features, cfg)
