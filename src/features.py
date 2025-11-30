"""
features.py
Preprocessing and feature engineering helpers.

We keep the numeric / categorical split fixed across all experiments and
wrap it in a ColumnTransformer so that the same preprocessing is reused
for classical models and neural networks.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Base numeric and categorical columns from the cleaned heart dataset
BASE_NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BASE_CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

# Engineered features shared by all experiments
ENGINEERED_NUM_COLS = ["bp_chol_ratio", "stress_index"]
ENGINEERED_CAT_COLS = ["age_band"]

NUM_COLS = BASE_NUM_COLS + ENGINEERED_NUM_COLS
CAT_COLS = BASE_CAT_COLS + ENGINEERED_CAT_COLS


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create reproducible engineered features used by every model."""
    df = df.copy()

    # Ratio of cholesterol to resting blood pressure (guard against zero/NaN)
    safe_trestbps = df["trestbps"].replace(0, np.nan)
    df["bp_chol_ratio"] = (df["chol"] / safe_trestbps).fillna(0.0)

    # Simple stress indicator: ST depression scaled by exercise-induced angina flag
    df["stress_index"] = df["oldpeak"] * (1 + df["exang"])

    # Age bucket for categorical risk-group modelling
    bins = [0, 40, 50, 60, 70, 200]
    labels = ["<40", "40-49", "50-59", "60-69", "70+"]
    df["age_band"] = pd.cut(df["age"], bins=bins, labels=labels, right=False).astype(str)

    return df


def build_preprocessor(scale_numeric: bool = True, drop_cols=None) -> ColumnTransformer:
    """
    Construct a ColumnTransformer that:
    - optionally standardises numeric features, and
    - one-hot encodes categorical features.

    Parameters
    ----------
    scale_numeric : bool
        If True, apply StandardScaler to numeric columns.
    drop_cols : list or None
        Optional list of column names to exclude from the transformers
        (useful for dropping the regression target).

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Transformer that can be fit on the training split and then
        reused (via .transform) on validation and test splits.
    """
    drop = set(drop_cols or [])

    num = [c for c in NUM_COLS if c not in drop]
    cat = [c for c in CAT_COLS if c not in drop]

    numeric_tf = StandardScaler() if scale_numeric and len(num) > 0 else "passthrough"
    categorical_tf = OneHotEncoder(handle_unknown="ignore") if len(cat) > 0 else "drop"

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num),
            ("cat", categorical_tf, cat),
        ],
        remainder="drop",
    )
    return pre
