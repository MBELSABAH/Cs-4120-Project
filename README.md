# Heart Disease Project Proposal

This repository contains our **Week 4 project proposal** using the **UCI Cleveland Heart Disease dataset**.  
We aim to (1) classify the presence of heart disease and (2) predict continuous cardiac measures such as serum cholesterol and maximum heart rate.  
The repo includes our proposal PDF, dataset references, baseline model plans, and reproducibility setup.

---

## Problem & Motivation

Cardiovascular disease is a leading global cause of death. Early detection using routine clinical features can help clinicians flag high-risk patients and guide interventions.  
Our project builds predictive models that support both **diagnosis** (classification of disease vs. no disease) and **risk stratification** (regression of continuous health measures).

---

## Dataset

- **Name:** Heart Disease (Cleveland subset)  
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)  
- **License/Terms:** As listed on the UCI dataset page  
- **Size:** 303 rows × 14 attributes  
- **Missing values:** 4 in `ca`, 2 in `thal`  
- **Sensitive attributes:** Age and sex are present, used carefully  

Raw data is **not** included here. Please download from UCI and place it under a local `data/` folder. A small `README.md` in that folder provides download instructions.

---

## Tasks

- **Classification:** Predict whether a patient has heart disease (binary: 0 = no disease, 1 = disease).  
- **Regression A:** Predict serum cholesterol (`chol`, mg/dL).  
- **Regression B:** Predict maximum heart rate (`thalach`, bpm).  

---

## Planned Metrics

- **Classification:** Accuracy, F1-score, and ROC-AUC.  
- **Regression:** Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).  

---

## Baseline Models

- **Classification:** Logistic Regression, Decision Tree (with optional k-NN or Naive Bayes).  
- **Regression:** Linear Regression and Ridge Regression (with optional Decision Tree Regressor).  

---

## Reproducibility Plan

- **Dependencies:** All requirements are pinned in `requirements.txt` (e.g., numpy, pandas, scikit-learn, mlflow, matplotlib).  
- **MLflow Tracking:** We will log experiments, metrics, and artifacts locally. To view results, run `mlflow ui`.  
- **Data Handling:** Raw dataset is excluded from version control. Instructions are provided so anyone can download it.  
- **Determinism:** Random seeds will be fixed for reproducibility.  


## How to Run

After setting up your environment (`pip install -r requirements.txt`):

- Train classical ML baselines:  
  ```bash
  python src/train_baselines.py
---

