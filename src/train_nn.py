"""
train_nn.py
Neural network training for classification and regression using the Heart Disease dataset.

Usage (example):
    python src/train_nn.py --data data/heart.csv --outdir reports/final --seed 42
"""
import argparse
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

from data import load_and_clean, make_stratified_splits
from features import build_preprocessor
from utils import set_seeds, ensure_dir


class MLP(nn.Module):
    """Simple feed-forward MLP used for both tasks."""

    def __init__(self, input_dim: int, hidden_dims, output_dim: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _to_tensor(x: np.ndarray, y: np.ndarray, is_classification: bool):
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    # For both tasks we store y as a column vector [N,1]
    y_tensor = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
    return x_tensor, y_tensor


def _make_loaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size, is_classification: bool):
    xtr, ytr = _to_tensor(x_train, y_train, is_classification)
    xv, yv = _to_tensor(x_val, y_val, is_classification)
    xt, yt = _to_tensor(x_test, y_test, is_classification)

    train_ds = TensorDataset(xtr, ytr)
    val_ds = TensorDataset(xv, yv)
    test_ds = TensorDataset(xt, yt)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def _run_epoch_cls(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss = 0.0
    n_samples = 0
    all_logits = []
    all_targets = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(xb).squeeze(1)  # [B]
        loss = criterion(logits, yb.squeeze(1))

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(yb.detach().cpu())

    avg_loss = total_loss / max(n_samples, 1)
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0).numpy().ravel()
    probs = torch.sigmoid(logits).numpy().ravel()
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="binary")
    return avg_loss, acc, f1


def _run_epoch_reg(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss = 0.0
    n_samples = 0
    all_preds = []
    all_targets = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if train:
            optimizer.zero_grad()

        preds = model(xb).squeeze(1)
        loss = criterion(preds, yb.squeeze(1))

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)
        all_preds.append(preds.detach().cpu())
        all_targets.append(yb.detach().cpu())

    avg_loss = total_loss / max(n_samples, 1)
    preds = torch.cat(all_preds, dim=0).numpy().ravel()
    targets = torch.cat(all_targets, dim=0).numpy().ravel()

    mae = mean_absolute_error(targets, preds)
    rmse = mean_squared_error(targets, preds, squared=False)
    return avg_loss, mae, rmse


def train_nn_classification(df: pd.DataFrame, outdir: str, seed: int = 42,
                            hidden_dims=(64, 32), dropout: float = 0.2,
                            lr: float = 1e-3, weight_decay: float = 1e-4,
                            batch_size: int = 32, epochs: int = 100):
    """
    Train an MLP for the classification task (predicting heart disease presence).

    Preprocessing:
    - Use the same numeric/categorical split as classical baselines.
    - Fit the ColumnTransformer on TRAIN only, then transform val/test.
    """
    idx_train, idx_val, idx_test = make_stratified_splits(df, seed=seed)
    feature_cols = [c for c in df.columns if c != "target"]
    X_all = df[feature_cols]
    y_all = df["target"].values

    pre = build_preprocessor(scale_numeric=True)
    X_train = pre.fit_transform(X_all.iloc[idx_train])
    X_val = pre.transform(X_all.iloc[idx_val])
    X_test = pre.transform(X_all.iloc[idx_test])

    y_train = y_all[idx_train]
    y_val = y_all[idx_val]
    y_test = y_all[idx_test]

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(input_dim, hidden_dims, output_dim=1, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    train_loader, val_loader, test_loader = _make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size, is_classification=True
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_f1 = -np.inf
    best_val_acc = 0.0
    best_state = None
    best_epoch = 0

    ensure_dir(outdir)
    models_dir = os.path.join(outdir, "models")
    ensure_dir(models_dir)

    with mlflow.start_run(run_name="NN_classification"):
        mlflow.log_param("task", "classification")
        mlflow.log_param("hidden_dims", hidden_dims)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("lr", lr)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc, train_f1 = _run_epoch_cls(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_acc, val_f1 = _run_epoch_cls(
                model, val_loader, criterion, optimizer=None, device=device, train=False
            )

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["train_f1"].append(train_f1)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            mlflow.log_metric("train_f1", train_f1, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                best_state = model.state_dict()
                best_epoch = epoch

        # Load best epoch weights before evaluating on test
        if best_state is not None:
            model.load_state_dict(best_state)

        test_loss, test_acc, test_f1 = _run_epoch_cls(
            model, test_loader, criterion, optimizer=None, device=device, train=False
        )
        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_param("best_epoch", best_epoch)

        # Save PyTorch model weights and preprocessor for later evaluation
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dims": list(hidden_dims),
                "dropout": dropout,
            },
            os.path.join(models_dir, "nn_cls.pt"),
        )
        joblib.dump(pre, os.path.join(models_dir, "nn_cls_preprocessor.pkl"))

        # Save learning curve for Plot 1 (Final report)
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(os.path.join(outdir, "nn_cls_learning_curve.csv"), index=False)

        # Save summary metrics for Table 1 (NN row)
        metrics_df = pd.DataFrame(
            [
                {
                    "Model": "NN_MLP",
                    "Accuracy_val": best_val_acc,
                    "F1_val": best_val_f1,
                    "Accuracy_test": test_acc,
                    "F1_test": test_f1,
                }
            ]
        )
        metrics_df.to_csv(os.path.join(outdir, "nn_classification_metrics.csv"), index=False)

    return best_val_acc, best_val_f1, test_acc, test_f1


def train_nn_regression(df: pd.DataFrame, outdir: str, seed: int = 42, y_col: str = "thalach",
                        hidden_dims=(64, 32), dropout: float = 0.1,
                        lr: float = 1e-3, weight_decay: float = 1e-4,
                        batch_size: int = 32, epochs: int = 100):
    """
    Train an MLP for the regression task (predicting thalach).

    Preprocessing:
    - Use the same numeric/categorical split, but drop the regression target from features.
    - Fit the ColumnTransformer on TRAIN only, then transform val/test.
    """
    idx_train, idx_val, idx_test = make_stratified_splits(df, seed=seed)
    if y_col not in df.columns:
        raise ValueError(f"Regression target column '{y_col}' not found.")

    feature_cols = [c for c in df.columns if c not in [y_col, "target"]]
    X_all = df[feature_cols]
    y_all = df[y_col].values

    pre = build_preprocessor(scale_numeric=True, drop_cols=[y_col])
    X_train = pre.fit_transform(X_all.iloc[idx_train])
    X_val = pre.transform(X_all.iloc[idx_val])
    X_test = pre.transform(X_all.iloc[idx_test])

    y_train = y_all[idx_train]
    y_val = y_all[idx_val]
    y_test = y_all[idx_test]

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(input_dim, hidden_dims, output_dim=1, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_loader, val_loader, test_loader = _make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size, is_classification=False
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_mae": [],
        "train_rmse": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
    }

    best_val_rmse = np.inf
    best_val_mae = np.inf
    best_state = None
    best_epoch = 0

    ensure_dir(outdir)
    models_dir = os.path.join(outdir, "models")
    ensure_dir(models_dir)

    with mlflow.start_run(run_name="NN_regression"):
        mlflow.log_param("task", "regression")
        mlflow.log_param("y_col", y_col)
        mlflow.log_param("hidden_dims", hidden_dims)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("lr", lr)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)

        for epoch in range(1, epochs + 1):
            train_loss, train_mae, train_rmse = _run_epoch_reg(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_mae, val_rmse = _run_epoch_reg(
                model, val_loader, criterion, optimizer=None, device=device, train=False
            )

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["train_mae"].append(train_mae)
            history["train_rmse"].append(train_rmse)
            history["val_loss"].append(val_loss)
            history["val_mae"].append(val_mae)
            history["val_rmse"].append(val_rmse)

            mlflow.log_metric("train_rmse", train_rmse, step=epoch)
            mlflow.log_metric("val_rmse", val_rmse, step=epoch)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_val_mae = val_mae
                best_state = model.state_dict()
                best_epoch = epoch

        if best_state is not None:
            model.load_state_dict(best_state)

        test_loss, test_mae, test_rmse = _run_epoch_reg(
            model, test_loader, criterion, optimizer=None, device=device, train=False
        )
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_param("best_epoch", best_epoch)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dims": list(hidden_dims),
                "dropout": dropout,
            },
            os.path.join(models_dir, "nn_reg.pt"),
        )
        joblib.dump(pre, os.path.join(models_dir, "nn_reg_preprocessor.pkl"))

        # Save learning curve for Plot 2 (Final report)
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(os.path.join(outdir, "nn_reg_learning_curve.csv"), index=False)

        # Save summary metrics for Table 2 (NN row)
        metrics_df = pd.DataFrame(
            [
                {
                    "Model": "NN_MLP",
                    "MAE_val": best_val_mae,
                    "RMSE_val": best_val_rmse,
                    "MAE_test": test_mae,
                    "RMSE_test": test_rmse,
                }
            ]
        )
        metrics_df.to_csv(os.path.join(outdir, "nn_regression_metrics.csv"), index=False)

    return best_val_mae, best_val_rmse, test_mae, test_rmse


def main(args):
    set_seeds(args.seed)
    df = load_and_clean(args.data)
    ensure_dir(args.outdir)

    # Classification NN – a bit deeper MLP, tuned for F1
    train_nn_classification(
        df,
        outdir=args.outdir,
        seed=args.seed,
        hidden_dims=(64, 32, 16),
        dropout=0.2,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=args.batch_size,
        epochs=args.epochs_cls,
    )

    # Regression NN – slightly smaller MLP, tuned for RMSE
    train_nn_regression(
        df,
        outdir=args.outdir,
        seed=args.seed,
        y_col="thalach",
        hidden_dims=(64, 32),
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=args.batch_size,
        epochs=args.epochs_reg,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/heart.csv")
    parser.add_argument("--outdir", default="reports/final")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs_cls", type=int, default=100)
    parser.add_argument("--epochs_reg", type=int, default=100)
    args = parser.parse_args()
    main(args)
