# Heart Disease Classification + Regression

End-to-end CS-4120 project repo for the UCI Cleveland Heart Disease dataset. We tackle:

- **Classification:** disease present vs absent (`target`).
- **Regression:** predict maximum heart rate achieved (`thalach`).

Everything needed to reproduce the final submission—data download helper, classical baselines, neural networks, plots, and MLflow tracking—is in this repository.

## Environment Setup

1. Create/activate a Python 3.11 environment.
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
   If PyTorch wheels are not found automatically, use the official index, e.g.:
   ```powershell
   pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
   ```
3. (Optional) launch MLflow tracking UI in another terminal:
   ```powershell
   mlflow ui
   ```

## Data

Raw UCI files are not committed. Use the helper script to fetch, clean, and save the canonical CSV used throughout the project:

```powershell
python src/make_heartcsv.py
```
This writes `data/heart.csv` (303 rows, 15 columns). The `data/README.md` reiterates the manual download link if needed.

## Engineered Features

We extend the original attributes with three reproducible engineered features shared across baselines and NNs:

1. `bp_chol_ratio` – ratio of serum cholesterol to resting blood pressure (stabilised for zero/NaN).
2. `stress_index` – ST depression (`oldpeak`) scaled by the exercise-induced angina flag (`exang`).
3. `age_band` – categorical age bucket (<40, 40-49, 50-59, 60-69, 70+).

These are added inside `load_and_clean` so every script consumes the same columns with no leakage.

## Reproducing Final Metrics & Artifacts

After installing requirements and creating `data/heart.csv`, run the pipeline below (seeds fixed via `src/utils.py`):

```powershell
# 1) Classical baselines (logs metrics + best pipelines into reports/final)
python src/train_baselines.py --data data/heart.csv --outdir reports/final --seed 42

# 2) Neural network training for both tasks (saves weights + learning curves)
python src/train_nn.py --data data/heart.csv --outdir reports/final --seed 42 --batch_size 32 --epochs_cls 100 --epochs_reg 100

# 3) Final evaluation + plots/tables (consumes results from steps 1 & 2)
python src/evaluate.py --data data/heart.csv --outdir reports/final --seed 42
```

Outputs (`reports/final/`):
- `classification_metrics.csv`, `regression_metrics.csv`, `nn_*_metrics.csv` – raw metrics for grading tables.
- `plot1_nn_cls_learning_curve.png` … `plot5_feature_importance.png` – the 5 required final figures.
- `table1_classification_comparison.csv`, `table2_regression_comparison.csv` – Final Report tables comparing classical vs NN.
- `models/` – serialized classical pipelines and NN checkpoints + preprocessors.

## MLflow Tracking

Every training script wraps experiments in MLflow runs (saved locally under `mlruns/`). You can inspect them via `mlflow ui` to verify parameters, metrics, and serialized models.

## Repository Layout

```
project/
+-- data/                # lightweight CSV + README with download instructions
+-- mlruns/              # local MLflow tracking dir (auto-generated)
+-- reports/
¦   +-- proposal/        # Week 4 PDF
¦   +-- midpoint/        # Week 8 PDF + required plots/tables
¦   +-- final/           # Final metrics, plots, and saved models
+-- notebooks/           # optional exploratory notebooks
+-- src/
¦   +-- data.py          # load + clean + consistent splits
¦   +-- features.py      # preprocessing + engineered features
¦   +-- train_baselines.py
¦   +-- train_nn.py
¦   +-- evaluate.py
¦   +-- make_heartcsv.py
¦   +-- utils.py
+-- requirements.txt
```

## Notes

- Random seeds are set in every entry-point via `utils.set_seeds` to keep Train/Val/Test splits and model initialisation deterministic.
- The repo ignores raw data, MLflow artifacts, and other large files via `.gitignore`.
- For poster/final write-ups, reference the generated plots/tables above so the PDFs stay lightweight.
