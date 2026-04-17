"""Build the windowed .h5 dataset. Thin entry point — see crypto_prediction.data.

Usage::

    python DataProcessor.py --config configs/default.yaml data.synthetic=true data.tail=30000
"""
from crypto_prediction.config import parse_cli
from crypto_prediction.data import build_dataset, save_dataset


def main():
    cfg = parse_cli(default_config='configs/default.yaml',
                    description='Build windowed .h5 dataset')
    ds = build_dataset(cfg.data, seed=cfg.seed)
    print(f'inputs={ds.inputs.shape} outputs={ds.outputs.shape}')
    save_dataset(ds, cfg.data.path)
    print(f'Wrote {cfg.data.path}')


if __name__ == '__main__':
    main()

