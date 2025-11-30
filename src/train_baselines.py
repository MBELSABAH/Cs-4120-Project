"""
train_baselines.py
Classical ML models (Logistic Regression, Decision Tree, Linear, DecisionTreeRegressor).

Entry point (Final project):
    python src/train_baselines.py --data data/heart.csv --outdir reports/final
"""
import argparse
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             root_mean_squared_error, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from data import load_and_clean, split_cls, split_reg
from features import build_preprocessor
from utils import ensure_dir, set_seeds


def train_classifiers(df: pd.DataFrame, outdir: str, seed: int = 42, use_auc: bool = False):
    """
    Train classical classification baselines and select the best model
    by validation F1 (or AUC if use_auc=True).

    Returns
    -------
    metrics_table : pd.DataFrame
        Per-model metrics on val & test splits.
    best_name : str
        Name of the best-performing model.
    """
    (Xtr, ytr), (Xv, yv), (Xt, yt) = split_cls(df, seed=seed)

    pre = build_preprocessor(scale_numeric=True)
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear"),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=seed),
    }

    rows = []
    ensure_dir(outdir)
    models_dir = os.path.join(outdir, "models")
    ensure_dir(models_dir)

    for name, est in models.items():
        with mlflow.start_run(run_name=f"CLS_{name}"):
            pipe = Pipeline([("prep", pre), ("clf", est)])
            pipe.fit(Xtr, ytr)

            # Validation metrics
            yv_hat = pipe.predict(Xv)
            acc_v = accuracy_score(yv, yv_hat)
            f1_v = f1_score(yv, yv_hat, average="binary")
            auc_v = (
                roc_auc_score(yv, pipe.predict_proba(Xv)[:, 1])
                if use_auc and hasattr(pipe[-1], "predict_proba")
                else None
            )

            # Test metrics
            yt_hat = pipe.predict(Xt)
            acc_t = accuracy_score(yt, yt_hat)
            f1_t = f1_score(yt, yt_hat, average="binary")
            auc_t = (
                roc_auc_score(yt, pipe.predict_proba(Xt)[:, 1])
                if use_auc and hasattr(pipe[-1], "predict_proba")
                else None
            )

            mlflow.log_metric("acc_val", acc_v)
            mlflow.log_metric("f1_val", f1_v)
            mlflow.log_metric("acc_test", acc_t)
            mlflow.log_metric("f1_test", f1_t)
            if use_auc and auc_v is not None:
                mlflow.log_metric("auc_val", auc_v)
                mlflow.log_metric("auc_test", auc_t)

            mlflow.sklearn.log_model(pipe, "model")

        rows.append(
            {
                "Model": name,
                "Accuracy_val": acc_v,
                "F1_val": f1_v if not use_auc else auc_v,
                "Accuracy_test": acc_t,
                "F1_test": f1_t if not use_auc else auc_t,
            }
        )

    table = pd.DataFrame(rows)

    # Select best classical model based on validation F1 (or AUC)
    key = "F1_val"
    best_name = table.sort_values(key, ascending=False).iloc[0]["Model"]

    # Refit best baseline on Train+Val and save for evaluation scripts
    X_trval = pd.concat([Xtr, Xv])
    y_trval = pd.concat([ytr, yv])
    best_est = models[best_name]
    best_pipe = Pipeline([("prep", pre), ("clf", best_est)]).fit(X_trval, y_trval)

    joblib.dump(best_pipe, os.path.join(models_dir, "best_cls.pkl"))
    table.to_csv(os.path.join(outdir, "classification_metrics.csv"), index=False)
    return table, best_name


def train_regressors(df: pd.DataFrame, outdir: str, seed: int = 42, y_var: str = "thalach"):
    """
    Train classical regression baselines and select the best model by
    validation RMSE.
    """
    (Xtr, ytr), (Xv, yv), (Xt, yt) = split_reg(df, seed=seed, y_col=y_var)

    pre = build_preprocessor(scale_numeric=True, drop_cols=[y_var])

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=seed),
    }

    rows = []
    ensure_dir(outdir)
    models_dir = os.path.join(outdir, "models")
    ensure_dir(models_dir)

    for name, est in models.items():
        with mlflow.start_run(run_name=f"REG_{name}"):
            pipe = Pipeline([("prep", pre), ("reg", est)])
            pipe.fit(Xtr, ytr)

            # Validation metrics
            yv_hat = pipe.predict(Xv)
            mae_v = mean_absolute_error(yv, yv_hat)
            rmse_v = root_mean_squared_error(yv, yv_hat)

            # Test metrics
            yt_hat = pipe.predict(Xt)
            mae_t = mean_absolute_error(yt, yt_hat)
            rmse_t = root_mean_squared_error(yt, yt_hat)

            mlflow.log_metric("mae_val", mae_v)
            mlflow.log_metric("rmse_val", rmse_v)
            mlflow.log_metric("mae_test", mae_t)
            mlflow.log_metric("rmse_test", rmse_t)
            mlflow.sklearn.log_model(pipe, "model")

        rows.append(
            {
                "Model": name,
                "MAE_val": mae_v,
                "RMSE_val": rmse_v,
                "MAE_test": mae_t,
                "RMSE_test": rmse_t,
            }
        )

    table = pd.DataFrame(rows)

    # Best classical regressor = lowest validation RMSE
    best_name = table.sort_values("RMSE_val").iloc[0]["Model"]

    X_trval = pd.concat([Xtr, Xv])
    y_trval = pd.concat([ytr, yv])
    best_est = models[best_name]
    best_pipe = Pipeline([("prep", pre), ("reg", best_est)]).fit(X_trval, y_trval)

    joblib.dump(best_pipe, os.path.join(models_dir, "best_reg.pkl"))
    table.to_csv(os.path.join(outdir, "regression_metrics.csv"), index=False)
    return table, best_name


def main(args):
    set_seeds(args.seed)
    df = load_and_clean(args.data)
    ensure_dir(args.outdir)

    cls_table, best_cls = train_classifiers(df, args.outdir, seed=args.seed, use_auc=False)
    reg_table, best_reg = train_regressors(df, args.outdir, seed=args.seed, y_var="thalach")

    print("Saved metrics to:")
    print(f"  {os.path.join(args.outdir, 'classification_metrics.csv')}")
    print(f"  {os.path.join(args.outdir, 'regression_metrics.csv')}")
    print("Best classical models:", best_cls, best_reg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/heart.csv")
    p.add_argument("--outdir", default="reports/final")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
