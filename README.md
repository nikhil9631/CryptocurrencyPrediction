# 📈 Cryptocurrency Price Prediction using Deep Learning

A deep learning-based project for forecasting cryptocurrency prices using historical time-series data. This project implements and compares multiple neural network architectures including **CNN, GRU, and LSTM** to model market behavior.

---

## 🚀 Overview

Cryptocurrency markets are highly volatile and difficult to predict. This project applies deep learning techniques to capture temporal patterns and improve prediction accuracy over traditional methods.

---

## 🧠 Models Implemented

- **CNN (Convolutional Neural Network)**  
  Captures short-term temporal patterns

- **GRU (Gated Recurrent Unit)**  
  Efficient sequence modeling with fewer parameters

- **LSTM (Long Short-Term Memory)**  
  Handles long-term dependencies in time-series data

---

## 📊 Baseline Models

- Linear Regression  
- Random Walk (Naive Model)

These are used to evaluate whether deep learning models provide real improvement.

---

## 🗂️ Project Structure

```
crypto_prediction/      Core package
├── config.py           Centralised Config dataclass + YAML loader + CLI overrides
├── seed.py             seed_everything() for full reproducibility
├── data.py             Build / save / load / split datasets
├── models.py           build_cnn / build_gru / build_lstm / build_gru_wf
├── train.py            Config-driven training loop
└── inference.py        Load weights → inverse-scaled predictions
configs/                YAML configs (default, cnn, gru, lstm)
tests/                  pytest suite (config, data, models, seed, train)
CNN.py GRU.py LSTM.py   Thin entry points → crypto_prediction.train
DataProcessor.py        Thin entry point → crypto_prediction.data
Prediction.py           Thin entry point → crypto_prediction.inference
PastSampler.py          Vectorised windowing helper
result/                 Plots, training CSVs, config snapshots
```

## Tech Stack

- Python 3.9+
- TensorFlow 2.x (bundled `tf.keras`)
- NumPy, Pandas, scikit-learn
- h5py
- Matplotlib

---

## Workflow

1. **Data Collection**
2. **Data Preprocessing**
3. **Time-series Windowing**
4. **Model Training (CNN / GRU / LSTM)**
5. **Evaluation & Visualization**

## ▶️ How to Run

### 1. Install
```bash
git clone https://github.com/nikhil9631/CryptocurrencyPrediction.git
cd CryptocurrencyPrediction
pip install -r requirements.txt
```

### 2. Build the dataset

The raw CSV and trained weights are **not** committed (too large). Either drop a
1-min OHLCV CSV (with `Timestamp` and `Close` columns) into `data/`, or generate
a synthetic random-walk series so the pipeline runs end-to-end:

```bash
# real data (path comes from configs/default.yaml → data.csv)
python DataProcessor.py

# or: synthetic data for a quick smoke test
python DataProcessor.py data.synthetic=true data.tail=50000
```

This writes `data/bitcoin2015to2017_close.h5`.

### 3. Train

Everything is driven by a single `Config` (see `configs/*.yaml`). Override any
field on the CLI with dotted `key=value` pairs:

```bash
python CNN.py                               # uses configs/cnn.yaml
python GRU.py  train.epochs=200 seed=7
python LSTM.py model.l1_reg=0.001 train.batch_size=16
python CNN.py  --config configs/cnn.yaml data.path=data/my_other.h5
```

Checkpoints land in `weights/`, training CSV + a JSON snapshot of the resolved
config land in `result/`.

### 4. Evaluate

```bash
python Prediction.py --config configs/cnn.yaml \
    --weights weights/<your-checkpoint>.weights.h5
```

### 5. Tests

```bash
pytest -q
```

The suite covers config loading/overrides, dataset windowing & HDF5 round-trip,
model output shapes, seed determinism, and a 1-epoch train→infer smoke test.

### Reproducibility

`crypto_prediction.seed.seed_everything()` seeds Python, NumPy and TensorFlow,
and enables TF deterministic ops. Two runs with the same `seed` and config
produce identical weights and loss curves (verified by
`tests/test_seed.py` / `tests/test_train.py`).

### GPU selection

GPU IDs are **not** hardcoded. To pin a device, set the env var yourself:

```bash
CUDA_VISIBLE_DEVICES=0 python LSTM.py
```

CPU-only machines work without changes — TF2 picks the cuDNN LSTM/GRU kernel
automatically when a compatible GPU is present.