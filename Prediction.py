"""Load trained weights, predict on the validation split, and plot predicted
vs ground-truth close price. Thin entry point — see crypto_prediction.inference.

Usage::

    python Prediction.py --config configs/cnn.yaml --weights weights/<ckpt>.weights.h5
"""
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crypto_prediction.config import load_config
from crypto_prediction.inference import predict


def main():
    parser = argparse.ArgumentParser(description='Plot predictions vs ground truth')
    parser.add_argument('--config', default='configs/cnn.yaml')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--out', default='prediction.png')
    parser.add_argument('--tail', type=int, default=1000)
    parser.add_argument('overrides', nargs='*')
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    os.makedirs(cfg.train.result_dir, exist_ok=True)

    times, truth, pred = predict(cfg, args.weights)
    print(times.shape, truth.shape, pred.shape)

    n = args.tail
    plt.figure(figsize=(12, 6))
    plt.plot(times[-n:], truth[-n:], label='ground truth')
    plt.plot(times[-n:], pred[-n:], label='predicted')
    plt.legend()
    out_path = os.path.join(cfg.train.result_dir, args.out)
    plt.savefig(out_path)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
