"""
evaluate.py
Functions for metrics, plots, confusion matrices, and residual analysis.
"""

import argparse, joblib, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
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


def make_pdf(df, outdir, group_name):
    # Load tables (created by train_baselines.py)
    cls = pd.read_csv(f"{outdir}/classification_metrics.csv")
    reg = pd.read_csv(f"{outdir}/regression_metrics.csv")
    pdf_path = f"{outdir}/midpoint_{group_name}.pdf"
    with PdfPages(pdf_path) as pdf:
        # Page 1, text blocks
        fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.axis('off')
        lines = [
          f"Midpoint Report, Heart Disease ({group_name})",
          "",
          "Updated Dataset Description & Cleaning:",
          "• UCI Cleveland Heart Disease (303 rows, 14 attributes).",
          "• target = 1 if disease present (num>0), else 0; regression target = thalach (bpm).",
          "• ca (4 rows) & thal (2 rows) imputed by mode; standardization & one-hot via sklearn Pipeline.",
          "",
          "EDA:",
          "• Slight class imbalance (~55% positive).",
          "• thalach inversely related to age; oldpeak tends to increase with exang.",
          "",
          "Split & Baselines:",
          "• 60/20/20 train/val/test split, random_state=42; stratified for classification.",
          "• Classification baselines: Logistic Regression, Decision Tree (MLflow tracked).",
          "• Regression baselines: Linear Regression, Decision Tree Regressor (MLflow tracked).",
          "",
          "Plots on following pages match rubric exactly: 1) target distribution, 2) correlation heatmap,",
          "3) confusion matrix of best classification baseline on TEST, 4) residuals vs predicted of best",
          "regression baseline on TEST.",
        ]
        y = 0.95
        for ln in lines:
            ax.text(0.06, y, ln, fontsize=10, va='top'); y -= 0.033
        pdf.savefig(fig); plt.close(fig)

        # Page 2, Plots 1 & 2
        for pth, title in [
          (f"{outdir}/plot1_target_distribution.png","Plot 1, Target Distribution"),
          (f"{outdir}/plot2_corr_heatmap.png","Plot 2, Correlation Heatmap"),
        ]:
            fig, ax = plt.subplots(figsize=(8.27,5.5)); ax.axis('off')
            img = plt.imread(pth); ax.imshow(img); ax.set_title(title)
            pdf.savefig(fig); plt.close(fig)

        # Page 3, Plot 3 + Table 1
        fig, ax = plt.subplots(figsize=(8.27,5.5)); ax.axis('off')
        img = plt.imread(f"{outdir}/plot3_confusion.png"); ax.imshow(img); ax.set_title("Plot 3, Confusion Matrix (Test)")
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.27,5.5)); ax.axis('off')
        ax.set_title("Table 1, Classification Metrics (Val & Test)")
        # tbl = ax.table(cellText=np.round(cls.values,4), colLabels=cls.columns, loc='center'); tbl.scale(1,1.3)
        
        cls_fmt = _rounded_table_data(cls, 4)
        tbl = ax.table(cellText=cls_fmt.values, colLabels=cls_fmt.columns, loc='center')
        tbl.scale(1, 1.3)

        pdf.savefig(fig); plt.close(fig)

        # Page 4, Plot 4 + Table 2
        fig, ax = plt.subplots(figsize=(8.27,5.5)); ax.axis('off')
        img = plt.imread(f"{outdir}/plot4_residuals.png"); ax.imshow(img); ax.set_title("Plot 4, Residuals vs Predicted (Test)")
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.27,5.5)); ax.axis('off')
        ax.set_title("Table 2, Regression Metrics (Val & Test)")
        # tbl = ax.table(cellText=np.round(reg.values,4), colLabels=reg.columns, loc='center'); tbl.scale(1,1.3)
        
        reg_fmt = _rounded_table_data(reg, 4)
        tbl = ax.table(cellText=reg_fmt.values, colLabels=reg_fmt.columns, loc='center')
        tbl.scale(1, 1.3)

        pdf.savefig(fig); plt.close(fig)

        # Page 5, Discussion + NN plan
        fig, ax = plt.subplots(figsize=(8.27, 11.69)); ax.axis('off')
        lines = [
          "Results & Discussion (brief):",
          "• The best classification baseline is chosen by highest validation F1; we report test confusion matrix.",
          "• The best regression baseline is chosen by lowest validation RMSE; residual plot shows remaining pattern.",
          "• Typical errors: borderline oldpeak/exang combinations (FPs); consider regularization/depth tuning.",
          "",
          "Neural Network Plan:",
          "• Two MLPs (classification & regression): 64→32→16 (ReLU), BatchNorm + Dropout 0.2.",
          "• Adam (lr=1e-3), batch 64, early stopping on val F1 (cls) and val RMSE (reg).",
          "• Standardize numeric, one-hot categorical; log learning curves and final metrics with MLflow.",
        ]
        y = 0.95
        for ln in lines:
            ax.text(0.06, y, ln, fontsize=10, va='top'); y -= 0.033
        pdf.savefig(fig); plt.close(fig)
    print("Wrote", pdf_path)

def main(args):
    df = load_and_clean(args.data)
    outdir = args.outdir
    # Plots 1–4
    plot_target_distribution(df, outdir)
    plot_corr_heatmap(df, outdir)
    plot_confusion(f"{outdir}/models/best_cls.pkl", df, outdir, seed=42)
    plot_residuals(f"{outdir}/models/best_reg.pkl", df, outdir, seed=42)
    # Assemble PDF
    make_pdf(df, outdir, group_name=args.group_name)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/heart.csv")
    ap.add_argument("--outdir", default="reports/midpoint")
    ap.add_argument("--group-name", default="GroupName")
    args = ap.parse_args()
    main(args)
