"""Cryptocurrency price prediction package.

Layout:
    config     -- centralized Config dataclass + YAML loader
    seed       -- reproducibility (python/numpy/tensorflow seeding)
    data       -- dataset building, loading, train/val split
    models     -- model factories (cnn, gru, lstm, gru_wf)
    train      -- generic training entry point driven by Config
    inference  -- weight loading, prediction, inverse-scaling
"""
from .config import Config, load_config
from .seed import seed_everything
