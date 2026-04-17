 # CLAUDE.md

## Project

Cryptocurrency price prediction using deep learning models (CNN, GRU, LSTM).
Built with Python, TensorFlow (v1), and Keras.

## Key Files

- `CNN.py` — CNN-based prediction model
- `GRU.py` — GRU-based prediction model
- `LSTM.py` — LSTM-based prediction model
- `DataProcessor.py` — prepares dataset (windowing)
- `PastSampler.py` — generates past/future samples
- `Prediction.py` — runs predictions and plotting
- `PlotRegularization.py` — regularization experiments

## Conventions

- Use Python scripts for training models (not notebooks)
- Keep dataset paths consistent (`data/*.h5`)
- Use numpy arrays with shape: (samples, time_steps, features)
- Always split dataset into train/validation
- Normalize data before training

## Do Not

- Do not hardcode GPU IDs (`CUDA_VISIBLE_DEVICES`)
- Do not change dataset structure without updating all models
- Do not mix notebook code inside `.py` files
- Do not commit large datasets or weights

## Workflow

- Train model using: `python CNN.py` or `GRU.py` or `LSTM.py`
- Evaluate using `Prediction.py`
- Use `Plot*.ipynb` for visualization only

## Testing

- No automated tests available
- Validate models using loss and prediction plots
- Compare with baseline models (Linear Regression / Random Walk)

## Environment

- Python 2.7 (legacy)
- TensorFlow 1.x
- Keras 2.1
- h5py for dataset handling

## Gotchas

- Dataset files (`.h5`) are not included — must be generated first
- `.ipynb_checkpoints` are not part of actual code
- Some scripts assume specific dataset shapes
- Old TensorFlow version may not work on modern systems