"""
utils.py
Shared helpers (random seed setting, logging, etc.).
"""
import os, random, numpy as np, pathlib

SEED = 42

def set_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def ensure_dir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
