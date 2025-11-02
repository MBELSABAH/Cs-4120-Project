"""
train_baselines.py
Classical ML models (Logistic Regression, Decision Tree, Linear, Ridge).
Entry point: python src/train_baselines.py
"""

import argparse, joblib, pandas as pd
import mlflow, mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from features import build_preprocessor
from data import load_and_clean, split_cls, split_reg
from utils import set_seeds, ensure_dir

def train_classifiers(df, outdir, seed=42, use_auc=False):
    (Xtr,ytr), (Xv,yv), (Xt,yt) = split_cls(df, seed=seed)
    pre = build_preprocessor(scale_numeric=True)
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear"),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=seed),
    }
    rows = []
    for name, est in models.items():
        with mlflow.start_run(run_name=f"CLS_{name}"):
            pipe = Pipeline([("prep", pre), ("clf", est)])
            pipe.fit(Xtr, ytr)
            # VAL
            yv_hat = pipe.predict(Xv)
            acc_v = accuracy_score(yv, yv_hat)
            f1_v  = f1_score(yv, yv_hat, average="macro")
            auc_v = roc_auc_score(yv, pipe.predict_proba(Xv)[:,1]) if use_auc and hasattr(pipe[-1],"predict_proba") else None
            # TEST
            yt_hat = pipe.predict(Xt)
            acc_t = accuracy_score(yt, yt_hat)
            f1_t  = f1_score(yt, yt_hat, average="macro")
            auc_t = roc_auc_score(yt, pipe.predict_proba(Xt)[:,1]) if use_auc and hasattr(pipe[-1],"predict_proba") else None
            # log
            mlflow.log_metric("acc_val", acc_v); mlflow.log_metric("f1_val", f1_v)
            mlflow.log_metric("acc_test", acc_t); mlflow.log_metric("f1_test", f1_t)
            if use_auc and auc_v is not None:
                mlflow.log_metric("auc_val", auc_v); mlflow.log_metric("auc_test", auc_t)
            mlflow.sklearn.log_model(pipe, "model")
        rows.append({
            "Model": name,
            "Accuracy_val": acc_v, "F1_val": f1_v if not use_auc else auc_v,
            "Accuracy_test": acc_t, "F1_test": f1_t if not use_auc else auc_t
        })
    table = pd.DataFrame(rows)
    # Select best by validation F1 (or AUC)
    key = "F1_val" if not use_auc else "F1_val"  # same column alias
    best_name = table.sort_values(key, ascending=False).iloc[0]["Model"]
    # Refit best on Train+Val, save artifact for evaluation plots
    X_trval = pd.concat([Xtr, Xv]); y_trval = pd.concat([ytr, yv])
    best_est = models[best_name]
    best_pipe = Pipeline([("prep", pre), ("clf", best_est)]).fit(X_trval, y_trval)
    ensure_dir(f"{outdir}/models"); joblib.dump(best_pipe, f"{outdir}/models/best_cls.pkl")
    table.to_csv(f"{outdir}/classification_metrics.csv", index=False)
    return table, best_name

def train_regressors(df, outdir, seed=42):
    (Xtr,ytr), (Xv,yv), (Xt,yt) = split_reg(df, seed=seed, y_col="thalach")

    y_var = "thalach"
    (Xtr,ytr), (Xv,yv), (Xt,yt) = split_reg(df, seed=seed, y_col=y_var)
    pre = build_preprocessor(scale_numeric=True, drop_cols=[y_var])

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=seed),
    }
    rows = []
    for name, est in models.items():
        with mlflow.start_run(run_name=f"REG_{name}"):
            pipe = Pipeline([("prep", pre), ("reg", est)])
            pipe.fit(Xtr, ytr)
            # VAL
            yv_hat = pipe.predict(Xv)
            mae_v = mean_absolute_error(yv, yv_hat)
            rmse_v = mean_squared_error(yv, yv_hat, squared=False)
            # TEST
            yt_hat = pipe.predict(Xt)
            mae_t = mean_absolute_error(yt, yt_hat)
            rmse_t = mean_squared_error(yt, yt_hat, squared=False)
            # log
            mlflow.log_metric("mae_val", mae_v); mlflow.log_metric("rmse_val", rmse_v)
            mlflow.log_metric("mae_test", mae_t); mlflow.log_metric("rmse_test", rmse_t)
            mlflow.sklearn.log_model(pipe, "model")
        rows.append({
            "Model": name,
            "MAE_val": mae_v, "RMSE_val": rmse_v,
            "MAE_test": mae_t, "RMSE_test": rmse_t
        })
    table = pd.DataFrame(rows)
    # Best by lowest RMSE_val
    best_name = table.sort_values("RMSE_val").iloc[0]["Model"]
    # Refit best on Train+Val, save
    X_trval = pd.concat([Xtr, Xv]); y_trval = pd.concat([ytr, yv])
    best_est = models[best_name]
    best_pipe = Pipeline([("prep", pre), ("reg", best_est)]).fit(X_trval, y_trval)
    ensure_dir(f"{outdir}/models"); joblib.dump(best_pipe, f"{outdir}/models/best_reg.pkl")
    table.to_csv(f"{outdir}/regression_metrics.csv", index=False)
    return table, best_name

def main(args):
    set_seeds(args.seed); ensure_dir(args.outdir)
    df = load_and_clean(args.data)
    cls_table, best_cls = train_classifiers(df, args.outdir, seed=args.seed, use_auc=False)
    reg_table, best_reg = train_regressors(df, args.outdir, seed=args.seed)
    print("Saved:", f"{args.outdir}/classification_metrics.csv", f"{args.outdir}/regression_metrics.csv")
    print("Best:", best_cls, best_reg)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/heart.csv")
    p.add_argument("--outdir", default="reports/midpoint")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
