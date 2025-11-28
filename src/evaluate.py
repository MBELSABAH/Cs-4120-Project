"""
evaluate.py

Utilities to generate the 5 required plots and 2 required tables for the FINAL report:

Plot 1 – Learning curve for classification NN (training and validation F1 vs epochs)
Plot 2 – Learning curve for regression NN (training and validation RMSE vs epochs)
Plot 3 – Confusion matrix for the best final classification model (classical or NN) on the test set
Plot 4 – Residuals vs predicted for the best final regression model (classical or NN) on the test set
Plot 5 – Permutation feature importance for a classical model

Table 1 – Classification comparison: best classical vs NN
Table 2 – Regression comparison: best classical vs NN
"""
import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data import load_and_clean, split_cls, split_reg, make_stratified_splits
from utils import ensure_dir


def plot_nn_cls_learning_curve(curve_path: str, outdir: str):
    df = pd.read_csv(curve_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["epoch"], df["train_f1"], label="Train F1")
    ax.plot(df["epoch"], df["val_f1"], label="Val F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 score")
    ax.set_title("NN Classification Learning Curve")
    ax.legend()
    ensure_dir(outdir)
    fig.savefig(os.path.join(outdir, "plot1_nn_cls_learning_curve.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_nn_reg_learning_curve(curve_path: str, outdir: str):
    df = pd.read_csv(curve_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["epoch"], df["train_rmse"], label="Train RMSE")
    ax.plot(df["epoch"], df["val_rmse"], label="Val RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title("NN Regression Learning Curve")
    ax.legend()
    ensure_dir(outdir)
    fig.savefig(os.path.join(outdir, "plot2_nn_reg_learning_curve.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_nn_cls(outdir: str):
    """Rebuild NN classification model + preprocessor for evaluation."""
    from train_nn import MLP  # avoid circular import at module level

    models_dir = os.path.join(outdir, "models")
    ckpt = torch.load(os.path.join(models_dir, "nn_cls.pt"), map_location="cpu")
    pre = joblib.load(os.path.join(models_dir, "nn_cls_preprocessor.pkl"))

    model = MLP(
        input_dim=ckpt["input_dim"],
        hidden_dims=ckpt["hidden_dims"],
        output_dim=1,
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, pre


def _load_nn_reg(outdir: str):
    """Rebuild NN regression model + preprocessor for evaluation."""
    from train_nn import MLP

    models_dir = os.path.join(outdir, "models")
    ckpt = torch.load(os.path.join(models_dir, "nn_reg.pt"), map_location="cpu")
    pre = joblib.load(os.path.join(models_dir, "nn_reg_preprocessor.pkl"))

    model = MLP(
        input_dim=ckpt["input_dim"],
        hidden_dims=ckpt["hidden_dims"],
        output_dim=1,
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, pre


def plot_confusion_best(df, outdir: str, seed: int = 42):
    """
    Plot 3 – Confusion matrix for the best final classification model.

    We choose between:
    - best classical baseline (best_cls.pkl, selected by validation F1), and
    - NN MLP (metrics stored in nn_classification_metrics.csv)

    based on validation F1.
    """
    cls_table = pd.read_csv(os.path.join(outdir, "classification_metrics.csv"))
    nn_table = pd.read_csv(os.path.join(outdir, "nn_classification_metrics.csv"))

    # Best classical row
    best_cls_row = cls_table.sort_values("F1_val", ascending=False).iloc[0]
    best_cls_f1_val = best_cls_row["F1_val"]

    # NN row (single)
    best_nn_row = nn_table.iloc[0]
    best_nn_f1_val = best_nn_row["F1_val"]

    # Decide which model is "best final"
    use_nn = best_nn_f1_val > best_cls_f1_val

    _, _, (Xt_cls, yt_cls) = split_cls(df, seed=seed)

    if not use_nn:
        # Classical path: load best_cls pipeline
        model = joblib.load(os.path.join(outdir, "models", "best_cls.pkl"))
        yhat = model.predict(Xt_cls)
        chosen_name = best_cls_row["Model"]
    else:
        # NN path: load NN + preprocessor and predict
        model, pre = _load_nn_cls(outdir)
        X_all = df.drop(columns=["target"])
        idx_train, idx_val, idx_test = make_stratified_splits(df, seed=seed)
        X_test = X_all.iloc[idx_test]
        X_test_proc = pre.transform(X_test)
        x_tensor = torch.as_tensor(X_test_proc, dtype=torch.float32)
        with torch.no_grad():
            logits = model(x_tensor).squeeze(1)
            probs = torch.sigmoid(logits).numpy().ravel()
        yhat = (probs >= 0.5).astype(int)
        yt_cls = df["target"].iloc[idx_test]
        chosen_name = "NN_MLP"

    cm = confusion_matrix(yt_cls, yhat)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix – Best Final Classifier ({chosen_name})")
    ensure_dir(outdir)
    fig.savefig(os.path.join(outdir, "plot3_confusion_best.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_residuals_best(df, outdir: str, seed: int = 42, y_col: str = "thalach"):
    """
    Plot 4 – Residuals vs predicted for the best final regression model.

    We choose between best classical regression baseline and NN MLP
    based on validation RMSE.
    """
    reg_table = pd.read_csv(os.path.join(outdir, "regression_metrics.csv"))
    nn_table = pd.read_csv(os.path.join(outdir, "nn_regression_metrics.csv"))

    best_cls_row = reg_table.sort_values("RMSE_val").iloc[0]
    best_cls_rmse_val = best_cls_row["RMSE_val"]

    best_nn_row = nn_table.iloc[0]
    best_nn_rmse_val = best_nn_row["RMSE_val"]

    use_nn = best_nn_rmse_val < best_cls_rmse_val

    (_, _), (_, _), (Xt_reg, yt_reg) = split_reg(df, seed=seed, y_col=y_col)

    if not use_nn:
        model = joblib.load(os.path.join(outdir, "models", "best_reg.pkl"))
        yhat = model.predict(Xt_reg)
        chosen_name = best_cls_row["Model"]
    else:
        model, pre = _load_nn_reg(outdir)
        feature_cols = [c for c in df.columns if c not in [y_col, "target"]]
        X_all = df[feature_cols]
        idx_train, idx_val, idx_test = make_stratified_splits(df, seed=seed)
        X_test = X_all.iloc[idx_test]
        X_test_proc = pre.transform(X_test)
        x_tensor = torch.as_tensor(X_test_proc, dtype=torch.float32)
        with torch.no_grad():
            preds = model(x_tensor).squeeze(1).numpy().ravel()
        yhat = preds
        yt_reg = df[y_col].iloc[idx_test]
        chosen_name = "NN_MLP"

    resid = yt_reg - yhat
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.scatter(yhat, resid, s=16, alpha=0.7)
    ax.axhline(0, ls="--", lw=1)
    ax.set_xlabel(f"Predicted {y_col}")
    ax.set_ylabel("Residuals (y - ŷ)")
    ax.set_title(f"Residuals vs Predicted – Best Final Regressor ({chosen_name})")
    ensure_dir(outdir)
    fig.savefig(os.path.join(outdir, "plot4_residuals_best.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(df, outdir: str, seed: int = 42):
    """
    Plot 5 – Permutation feature importance for the best classical classifier.

    We compute permutation importance on the validation set for interpretability.
    """
    (_, _), (Xv, yv), (_, _) = split_cls(df, seed=seed)
    model = joblib.load(os.path.join(outdir, "models", "best_cls.pkl"))

    r = permutation_importance(
        model, Xv, yv, n_repeats=50, random_state=seed, n_jobs=-1
    )
    importances = r.importances_mean
    std = r.importances_std
    feature_names = Xv.columns

    order = np.argsort(importances)[::-1]
    top_k = min(10, len(order))
    order = order[:top_k]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(top_k), importances[order], yerr=std[order], align="center")
    ax.set_xticks(range(top_k))
    ax.set_xticklabels(feature_names[order], rotation=45, ha="right")
    ax.set_ylabel("Permutation importance (mean decrease in score)")
    ax.set_title("Top Feature Importances – Best Classical Classifier")
    plt.tight_layout()
    ensure_dir(outdir)
    fig.savefig(os.path.join(outdir, "plot5_feature_importance.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_tables(outdir: str):
    """
    Build Table 1 and Table 2 for the report:

    Table 1 – Classification comparison: best classical vs NN (val & test, Accuracy & F1)
    Table 2 – Regression comparison: best classical vs NN (val & test, MAE & RMSE)
    """
    cls_table = pd.read_csv(os.path.join(outdir, "classification_metrics.csv"))
    reg_table = pd.read_csv(os.path.join(outdir, "regression_metrics.csv"))
    nn_cls_table = pd.read_csv(os.path.join(outdir, "nn_classification_metrics.csv"))
    nn_reg_table = pd.read_csv(os.path.join(outdir, "nn_regression_metrics.csv"))

    best_cls_row = cls_table.sort_values("F1_val", ascending=False).iloc[0]
    best_cls_row = best_cls_row.copy()
    best_cls_row["Model"] = f"{best_cls_row['Model']} (classical)"

    nn_row = nn_cls_table.iloc[0].copy()
    nn_row["Model"] = "NN_MLP"

    table1 = pd.DataFrame([best_cls_row, nn_row])
    table1.to_csv(os.path.join(outdir, "table1_classification_comparison.csv"), index=False)

    best_reg_row = reg_table.sort_values("RMSE_val").iloc[0]
    best_reg_row = best_reg_row.copy()
    best_reg_row["Model"] = f"{best_reg_row['Model']} (classical)"

    nn_reg_row = nn_reg_table.iloc[0].copy()
    nn_reg_row["Model"] = "NN_MLP"

    table2 = pd.DataFrame([best_reg_row, nn_reg_row])
    table2.to_csv(os.path.join(outdir, "table2_regression_comparison.csv"), index=False)


def main(args):
    df = load_and_clean(args.data)
    outdir = args.outdir
    ensure_dir(outdir)

    # Plot 1 & 2 – learning curves from CSVs saved by train_nn.py
    plot_nn_cls_learning_curve(os.path.join(outdir, "nn_cls_learning_curve.csv"), outdir)
    plot_nn_reg_learning_curve(os.path.join(outdir, "nn_reg_learning_curve.csv"), outdir)

    # Plot 3 & 4 – best final models (classical vs NN)
    plot_confusion_best(df, outdir, seed=args.seed)
    plot_residuals_best(df, outdir, seed=args.seed, y_col="thalach")

    # Plot 5 – feature importance
    plot_feature_importance(df, outdir, seed=args.seed)

    # Tables 1 & 2 – comparison tables
    build_tables(outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/heart.csv")
    parser.add_argument("--outdir", default="reports/final")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
