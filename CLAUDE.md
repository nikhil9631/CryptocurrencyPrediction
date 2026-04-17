 # CLAUDE.md

## Project

Cryptocurrency price prediction using deep learning models (CNN, GRU, LSTM).
Python 3 / TensorFlow 2 (`tf.keras`). Config-driven, seeded, tested.

## Architecture

- `crypto_prediction/` — core package
  - `config.py` — `Config` dataclass, YAML loader, `key=value` CLI overrides
  - `seed.py` — `seed_everything()` (python/numpy/TF + deterministic ops)
  - `data.py` — `build_dataset` / `save_dataset` / `load_dataset` / `split`
  - `models.py` — `MODEL_BUILDERS` registry: cnn, gru, lstm, gru_wf
  - `train.py` — `run(cfg)` generic training loop
  - `inference.py` — `predict(cfg, weights)` → (times, truth, pred)
- `configs/*.yaml` — declarative experiment configs
- `tests/` — pytest suite (config, data, models, seed, train+infer)
- Top-level `CNN.py` / `GRU.py` / `LSTM.py` / `DataProcessor.py` /
  `Prediction.py` are 5-line wrappers over the package — keep them thin.

## Conventions

- All behaviour flows through `Config`. Add new knobs there, not ad-hoc argparse.
- Override on CLI with dotted keys: `train.epochs=5 seed=7 model.units=64`
- New model → add factory to `crypto_prediction/models.py` and register in
  `MODEL_BUILDERS`; add a `configs/<name>.yaml`; add a shape test.
- Array shape: `(samples, time_steps, features)`
- Import Keras via `tensorflow.keras`, never standalone `keras`
- Call `seed_everything(cfg.seed)` before any model/data construction

## Do Not

- Do not add per-script argparse — extend `Config` instead
- Do not hardcode `CUDA_VISIBLE_DEVICES` — set it in the shell if needed
- Do not use TF1 APIs (`tf.Session`, `tf.ConfigProto`, `set_session`)
- Do not commit large datasets or weights

## Workflow

- Build dataset: `python DataProcessor.py [data.synthetic=true]`
- Train: `python CNN.py [--config configs/cnn.yaml] [key=value ...]`
- Evaluate: `python Prediction.py --config <cfg> --weights weights/<ckpt>.weights.h5`
- Test: `pytest -q`

## Testing

- `pytest -q` — 21 tests, ~15s on CPU
- Covers: config load/override/reject-unknown, PastSampler shapes, dataset
  build seeding + HDF5 round-trip, model forward shapes, weight-init
  determinism, 1-epoch train reproducibility, train→inference round-trip
- New code should add or update tests in `tests/`

## Environment

- Python 3.9+
- `pip install -r requirements.txt` (tensorflow, numpy, pandas, h5py,
  scikit-learn, matplotlib, pyyaml, pytest)

## Gotchas

- `.h5` datasets and `weights/` are gitignored — generate them first
- `data.synthetic=true` gives a runnable smoke-test dataset
- CNN architecture requires `input_steps >= 256` (fixed kernel/stride)
- Checkpoint extension is `.weights.h5` (Keras 3 requirement)
- Every training run dumps `result/<name>.config.json` for provenance