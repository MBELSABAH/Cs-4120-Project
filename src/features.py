"""
features.py
Preprocessing and feature engineering helpers.

We keep the numeric / categorical split fixed across all experiments and
wrap it in a ColumnTransformer so that the same preprocessing is reused
for classical models and neural networks.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Numeric and categorical column groups in the cleaned heart dataset
NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]


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
    # Make OHE dense so it’s easy to feed into PyTorch
    categorical_tf = OneHotEncoder(handle_unknown="ignore") if len(cat) > 0 else "drop"

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, num),
            ("cat", categorical_tf, cat),
        ],
        remainder="drop",
    )
    return pre
