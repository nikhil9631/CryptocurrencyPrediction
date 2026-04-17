"""Train the LSTM model. Thin entry point — see crypto_prediction.train."""
from crypto_prediction.train import main

if __name__ == '__main__':
    main(default_config='configs/lstm.yaml')
