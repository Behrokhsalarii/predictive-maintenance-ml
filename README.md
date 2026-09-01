# Predictive Maintenance — Machine Failure Classification

An end-to-end machine learning pipeline for predicting industrial machine failures from sensor telemetry. The project covers exploratory data analysis, domain-driven feature engineering, class-imbalance handling, model benchmarking, hyperparameter tuning, and model interpretability.

## Overview

Unplanned equipment downtime is one of the most costly problems in manufacturing. This project frames predictive maintenance as a binary classification task, predicting whether a machine will fail based on real-time sensor readings such as temperature, rotational speed, torque, and tool wear.

## Features

- Exploratory data analysis with distribution, correlation, and failure-mode breakdowns
- Domain-driven feature engineering (mechanical power, thermal differential, overstrain index)
- Class-imbalance handling with SMOTE oversampling
- Benchmarking of four models: Logistic Regression, Random Forest, XGBoost, LightGBM
- Hyperparameter tuning via stratified cross-validated grid search
- Evaluation with ROC-AUC, PR-AUC, F1-score, confusion matrix, and ROC/PR curves
- Model interpretability using SHAP
- Trained model persistence for deployment

## Project Structure

```
.
├── data/
│   └── (place your dataset CSV here)
├── predictive_maintenance.ipynb
├── predictive_maintenance_xgb.pkl
├── scaler.pkl
├── label_encoder.pkl
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/predictive-maintenance.git
cd predictive-maintenance
pip install -r requirements.txt
```

## Usage

1. Place your dataset CSV file inside the `data/` directory.
2. Launch the notebook:

```bash
jupyter notebook predictive_maintenance.ipynb
```

3. Run all cells to reproduce the full pipeline, from EDA through model training and evaluation.

## Model Performance

The tuned XGBoost classifier achieved the strongest ROC-AUC and PR-AUC among the benchmarked models on the held-out test set, with feature importance and SHAP analysis identifying torque, tool wear, and the derived overstrain index as the dominant predictors of failure risk.

## Tech Stack

- Python, pandas, NumPy
- scikit-learn, XGBoost, LightGBM
- imbalanced-learn (SMOTE)
- SHAP
- Matplotlib, Seaborn

## License

This project is released under the MIT License.
