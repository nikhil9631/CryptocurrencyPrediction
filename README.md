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

├── CNN.py # CNN model training
├── GRU.py # GRU model training
├── LSTM.py # LSTM model training
├── GRU_WF.py # Walk-forward GRU model
├── DataProcessor.py # Data preprocessing and window generation
├── PastSampler.py # Time-series sampling utility
├── Prediction.py # Prediction and evaluation
├── PlotRegularization.py # Regularization experiments
├── result/ # Model output plots
├── *.ipynb # Experiment notebooks


## Tech Stack

- Python 2.7 (Legacy)
- TensorFlow 1.x
- Keras 2.1
- NumPy, Pandas
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

### 1. Clone the repository
```bash
git clone https://github.com/nikhil9631/CryptocurrencyPrediction.git
cd CryptocurrencyPrediction

python DataProcessor.py

python CNN.py
python GRU.py
python LSTM.py

python Prediction.py