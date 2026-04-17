"""Walk-forward GRU training across partitions.

This script consumes a 4-D partitioned dataset (partitions, samples, steps,
features) so it has its own loop rather than reusing crypto_prediction.train.
"""
import os
import h5py
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

from crypto_prediction.config import parse_cli
from crypto_prediction.seed import seed_everything
from crypto_prediction.models import build_gru_wf
from crypto_prediction.train import _configure_gpu, _ensure_dirs

TRAIN_SPLIT = 8000 * 4


def main():
    cfg = parse_cli(default_config=None,
                    description='Walk-forward GRU over partitioned dataset')
    if cfg.data.path == 'data/bitcoin2015to2017_close.h5':
        cfg.data.path = 'data/allcoin2015to2017_wf.h5'
    cfg.train.output_name = cfg.train.output_name if cfg.train.output_name != 'run' \
        else 'allcoin2015to2017_WF_GRU_tanh_leaky'

    seed_everything(cfg.seed, cfg.deterministic)
    _configure_gpu()
    _ensure_dirs(cfg.train.weights_dir, cfg.train.result_dir)

    with h5py.File(cfg.data.path, 'r') as hf:
        datas, labels = hf['inputs'][:], hf['outputs'][:]

    nb_partitions, _, step_size, nb_features = datas.shape
    log_path = os.path.join(cfg.train.result_dir, cfg.train.output_name + '.csv')

    for partition in range(nb_partitions):
        train_x = datas[partition, :TRAIN_SPLIT, :, :]
        val_x = datas[partition, TRAIN_SPLIT:, :, :]
        train_y = labels[partition, :TRAIN_SPLIT, :, :]
        val_y = labels[partition, TRAIN_SPLIT:, :, :]
        print(f'[partition {partition}] train_x={train_x.shape} val_x={val_x.shape}')

        ckpt = os.path.join(
            cfg.train.weights_dir,
            f'{cfg.train.output_name}_partition_{partition}.weights.h5',
        )
        cbs = [
            EarlyStopping(monitor='val_loss', mode='min', patience=10),
            CSVLogger(log_path, append=True),
            ModelCheckpoint(ckpt, monitor='val_loss', verbose=1,
                            save_best_only=True, mode='min',
                            save_weights_only=True),
        ]
        model = build_gru_wf(step_size, nb_features, cfg.model)
        model.fit(train_x, train_y,
                  batch_size=cfg.train.batch_size,
                  epochs=cfg.train.epochs,
                  validation_data=(val_x, val_y),
                  callbacks=cbs, verbose=0)


if __name__ == '__main__':
    main()
