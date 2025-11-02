"""
data.py
Handles loading, cleaning, and splitting the Heart Disease dataset.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

def _load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "num" in df.columns and "target" not in df.columns:
        # Clean '?' from the UCI file
        df = df.replace("?", pd.NA)
        for c in ["ca","thal"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["target"] = (df["num"] > 0).astype(int)
    return df

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = _load_raw(csv_path).copy()
    # ignore missingness
    impute_mode = ["ca","thal"]
    for c in impute_mode:
        if c in df.columns and df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode(dropna=True)[0])
    # Remove original 'num' if present
    if "num" in df.columns:
        df = df.drop(columns=["num"])
    # Ensure dtypes sane
    for c in ["sex","cp","fbs","restecg","exang","slope","thal"]:
        if c in df.columns:
            df[c] = df[c].astype(int)
    return df

def split_cls(df: pd.DataFrame, seed: int = 42):
    X = df.drop(columns=["target"])
    y = df["target"]
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=seed, stratify=y_tmp
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def split_reg(df: pd.DataFrame, seed: int = 42, y_col: str = "thalach"):
    # same indices classification split if you want perfect parity
    X = df.drop(columns=[y_col])
    y = df[y_col]
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=seed)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
