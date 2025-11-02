"""
features.py
Preprocessing and feature engineering functions (to be implemented).
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_COLS = ["age","trestbps","chol","thalach","oldpeak","ca"]
CAT_COLS = ["sex","cp","fbs","restecg","exang","slope","thal"]

def build_preprocessor(scale_numeric: bool = True, drop_cols=None):
    """Return a ColumnTransformer that skips any columns in drop_cols."""
    drop = set(drop_cols or [])
    num = [c for c in NUM_COLS if c not in drop]
    cat = [c for c in CAT_COLS if c not in drop]
    return ColumnTransformer([
        ("num", StandardScaler() if scale_numeric else "passthrough", num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ], remainder="drop")
