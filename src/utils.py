"""
utils.py
Shared helpers (random seed setting, directory creation, etc.).
"""
import os
import pathlib
import random

import numpy as np

SEED = 42


def set_seeds(seed: int = SEED):
    """
    Fix all random seeds we control so that experiments are reproducible.

    We deliberately do NOT rely on global sklearn or PyTorch randomness.
    Instead, all scripts call this once at the beginning and then use
    explicit `random_state` / `generator` arguments where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # Torch is only needed for the NN script; ignore if unavailable.
        pass


def ensure_dir(path: str):
    """Create a directory (and parents) if it does not already exist."""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
