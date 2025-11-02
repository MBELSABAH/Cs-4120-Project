"""
evaluate.py
Functions for metrics, plots, confusion matrices, and residual analysis.
"""

import argparse, joblib, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data import load_and_clean, split_cls, split_reg
from utils import ensure_dir

def plot_target_distribution(df, outdir):
    counts = df["target"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5,3.2))
    ax.bar(['Absent (0)','Present (1)'], counts.values)
    ax.set_title("Target Distribution (Classification)")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v+1, str(int(v)), ha='center', va='bottom', fontsize=9)
    ensure_dir(outdir); fig.savefig(f"{outdir}/plot1_target_distribution.png", dpi=200, bbox_inches="tight"); plt.close(fig)

def plot_corr_heatmap(df, outdir):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != "target"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6.2,5))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, square=False)
    ax.set_title("Feature Correlation Heatmap")
    fig.savefig(f"{outdir}/plot2_corr_heatmap.png", dpi=200, bbox_inches="tight"); plt.close(fig)

def plot_confusion(best_cls_path, df, outdir, seed=42):
    (_, _), (_, _), (Xt, yt) = split_cls(df, seed=seed)
    model = joblib.load(best_cls_path)
    yhat = model.predict(Xt)
    cm = confusion_matrix(yt, yhat)
    fig, ax = plt.subplots(figsize=(4.8,3.6))
    disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
    disp.plot(ax=ax, values_format='d', colorbar=False)
    ax.set_title("Confusion Matrix, Best Classification Baseline (Test)")
    fig.savefig(f"{outdir}/plot3_confusion.png", dpi=200, bbox_inches="tight"); plt.close(fig)

def plot_residuals(best_reg_path, df, outdir, seed=42):
    (_, _), (_, _), (Xt, yt) = split_reg(df, seed=seed, y_col="thalach")
    model = joblib.load(best_reg_path)
    yhat = model.predict(Xt)
    resid = yt - yhat
    fig, ax = plt.subplots(figsize=(5.5,3.5))
    ax.scatter(yhat, resid, s=16, alpha=0.7)
    ax.axhline(0, ls="--", lw=1)
    ax.set_xlabel("Predicted thalach"); ax.set_ylabel("Residuals (y - ŷ)")
    ax.set_title("Residuals vs Predicted, Best Regression Baseline (Test)")
    fig.savefig(f"{outdir}/plot4_residuals.png", dpi=200, bbox_inches="tight"); plt.close(fig)

def _rounded_table_data(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    """Round only numeric columns; keep string columns (e.g., 'Model') unchanged."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(float).round(digits)
    return df

def main(args):
    df = load_and_clean(args.data)
    outdir = args.outdir
    # Plots 1–4
    plot_target_distribution(df, outdir)
    plot_corr_heatmap(df, outdir)
    plot_confusion(f"{outdir}/models/best_cls.pkl", df, outdir, seed=42)
    plot_residuals(f"{outdir}/models/best_reg.pkl", df, outdir, seed=42)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/heart.csv")
    ap.add_argument("--outdir", default="reports/midpoint")
    ap.add_argument("--group-name", default="GroupName")
    args = ap.parse_args()
    main(args)
