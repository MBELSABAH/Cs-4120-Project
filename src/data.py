"""
data.py
Handles loading, cleaning, and splitting the Heart Disease dataset.

Key points for the final project:
- load_and_clean applies the same cleaning rules for all experiments.
- make_stratified_splits creates a single set of train/val/test indices,
  stratified on the binary classification label `target`.
- split_cls and split_reg both reuse those indices so that every model
  (classical and NN) sees exactly the same rows in each split.
"""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from features import add_engineered_features


def _load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "num" in df.columns and "target" not in df.columns:
        # Clean '?' from the UCI file
        df = df.replace("?", pd.NA)
        for c in ["ca", "thal"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # Convert original 0..4 label to binary disease present / not present
        df["target"] = (df["num"] > 0).astype(int)
    return df


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Load the raw CSV, clean missing values, and normalise dtypes.

    This is shared by all experiments (baselines + neural nets) so that
    data leakage and cleaning decisions are consistent.
    """
    df = _load_raw(csv_path).copy()

    # Simple imputation: mode for categorical-like columns that have a few '?'
    impute_mode = ["ca", "thal"]
    for c in impute_mode:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode(dropna=True)[0])

    # Remove original multiclass label if present (we only use binary `target`)
    if "num" in df.columns:
        df = df.drop(columns=["num"])

    # Ensure integer dtypes for categorical codes
    for c in ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]:
        if c in df.columns:
            df[c] = df[c].astype(int)

    df = add_engineered_features(df)

    return df


def make_stratified_splits(df: pd.DataFrame, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a single train/val/test split, stratified on `target`.

    Returns
    -------
    idx_train, idx_val, idx_test : np.ndarray of row indices

    These indices are reused for both the classification and regression tasks
    so that the comparison between classical models and NNs is fair.
    """
    if "target" not in df.columns:
        raise ValueError("Expected 'target' column for stratified split.")

    idx = np.arange(len(df))
    y = df["target"].values

    idx_tmp, idx_test, y_tmp, y_test = train_test_split(
        idx,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_tmp,
        y_tmp,
        test_size=0.25,  # 0.25 of 0.8 = 0.2 overall -> 60/20/20 split
        random_state=seed,
        stratify=y_tmp,
    )
    return idx_train, idx_val, idx_test


def split_cls(df: pd.DataFrame, seed: int = 42):
    """
    Classification split: disease presence (target 0/1).

    Uses the shared indices from make_stratified_splits so that all
    classification models see the same rows in each split.
    """
    idx_train, idx_val, idx_test = make_stratified_splits(df, seed=seed)
    X = df.drop(columns=["target"])
    y = df["target"]
    return (
        (X.iloc[idx_train], y.iloc[idx_train]),
        (X.iloc[idx_val], y.iloc[idx_val]),
        (X.iloc[idx_test], y.iloc[idx_test]),
    )


def split_reg(df: pd.DataFrame, seed: int = 42, y_col: str = "thalach"):
    """
    Regression split: predict a continuous target (default: thalach).

    We reuse the same row indices as the classification task and drop
    both the regression target and the classification label from the
    feature matrix.
    """
    if y_col not in df.columns:
        raise ValueError(f"Expected regression target column '{y_col}'.")

    idx_train, idx_val, idx_test = make_stratified_splits(df, seed=seed)

    feature_cols = [c for c in df.columns if c not in [y_col, "target"]]
    X = df[feature_cols]
    y = df[y_col]

    return (
        (X.iloc[idx_train], y.iloc[idx_train]),
        (X.iloc[idx_val], y.iloc[idx_val]),
        (X.iloc[idx_test], y.iloc[idx_test]),
    )
