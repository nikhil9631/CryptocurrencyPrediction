"""Sweep L1/L2 regularizers on the LSTM (bias, kernel, activity, recurrent)
and write per-sweep MSE summaries to result/ (TF2 / Keras 3)."""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, LSTM, LeakyReLU
from tensorflow.keras import regularizers

from crypto_prediction.seed import seed_everything
from crypto_prediction.data import load_dataset
from crypto_prediction.train import _configure_gpu, _ensure_dirs

UNITS = 50
OUTPUT_SIZE = 16
REG_TARGETS = ('bias', 'kernel', 'activity', 'recurrent')


def make_regularizers():
    return [
        regularizers.l1(0), regularizers.l1(0.1), regularizers.l1(0.01),
        regularizers.l1(0.001), regularizers.l1(0.0001),
        regularizers.l2(0.1), regularizers.l2(0.01),
        regularizers.l2(0.001), regularizers.l2(0.0001),
    ]


def reg_name(reg):
    cfg = reg.get_config()
    l1 = float(cfg.get('l1', 0.0))
    l2 = float(cfg.get('l2', 0.0))
    return f'l1 {l1:.4f},l2 {l2:.4f}'


def fit_lstm(target, reg, train_x, train_y, step_size, nb_features,
             batch_size, epochs):
    kwargs = {f'{target}_regularizer': reg}
    model = Sequential([
        Input(shape=(step_size, nb_features)),
        LSTM(units=UNITS, return_sequences=False, **kwargs),
        Activation('tanh'),
        Dropout(0.2),
        Dense(OUTPUT_SIZE),
        LeakyReLU(),
    ])
    model.compile(loss='mse', optimizer='adam')
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=0)
    return model


def experiment(target, reg, nb_repeat, train_x, train_y, val_x,
               original_datas, val_original_outputs, step_size, nb_features,
               batch_size, epochs):
    scaler = MinMaxScaler()
    scaler.fit(original_datas[:, 0].reshape(-1, 1))
    truth = val_original_outputs[:, :, 0].reshape(-1)

    scores = []
    for _ in range(nb_repeat):
        model = fit_lstm(target, reg, train_x, train_y, step_size,
                         nb_features, batch_size, epochs)
        predicted = model.predict(val_x, verbose=0)
        predicted_inv = scaler.inverse_transform(predicted).reshape(-1)
        scores.append(mean_squared_error(truth, predicted_inv))
    return scores


def main():
    parser = argparse.ArgumentParser(description='LSTM regularization sweep')
    parser.add_argument('--data', default='data/bitcoin2015to2017_close.h5')
    parser.add_argument('--result-dir', default='result')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--targets', nargs='+', default=list(REG_TARGETS),
                        choices=REG_TARGETS)
    args = parser.parse_args()

    seed_everything(args.seed)
    _configure_gpu()
    _ensure_dirs(args.result_dir)

    ds = load_dataset(args.data)
    datas, labels = ds.inputs, ds.outputs
    original_datas = ds.original_datas
    original_outputs = ds.original_outputs

    training_size = int(0.8 * datas.shape[0])
    train_x = datas[:training_size, :, :]
    train_y = labels[:training_size, :, 0]
    val_x = datas[training_size:, :, :]
    val_original_outputs = original_outputs[training_size:, :, :]

    step_size, nb_features = datas.shape[1], datas.shape[2]

    for target in args.targets:
        results = pd.DataFrame()
        for reg in make_regularizers():
            name = reg_name(reg)
            print(f'[{target}] Training {name}')
            results[name] = experiment(
                target, reg, args.repeats, train_x, train_y, val_x,
                original_datas, val_original_outputs, step_size, nb_features,
                args.batch_size, args.epochs,
            )
        out_csv = os.path.join(args.result_dir, f'lstm_{target}_reg.csv')
        results.describe().to_csv(out_csv)
        print(f'Saved {out_csv}')
        print(results.describe())


if __name__ == '__main__':
    main()
