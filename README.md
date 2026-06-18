# Churn-Prediction

# End-to-End Machine Learning Architecture

This repository implements a modular, production-ready Machine Learning pipeline designed to handle high-dimensional data, optimize model performance, and evaluate predictions using robust statistical metrics.

The project is structured into three decoupled core modules: **Preprocess**, **Train**, and **Evaluate**, ensuring high maintainability and clean code practices.

## 🛠️ Architecture & Pipeline Modules

### 1. Preprocessing Module (`preprocess.py`)
Responsible for data cleaning, sanitization, and feature engineering.
- Automated handling of missing values and outlier detection.
- Categorical variable encoding and numerical feature scaling.
- Feature selection to reduce dimensionality and avoid overfitting.

### 2. Training Module (`train.py`)
Handles model selection, hyperparameter tuning, and training tracking.
- Implements robust algorithms (such as XGBoost / Scikit-Learn ensembles).
- Stratified K-Fold Cross-Validation to ensure generalization across data splits.
- Hyperparameter optimization pipeline.

### 3. Evaluation Module (`evaluate.py`)
Computes deep performance metrics to validate model reliability before deployment.
- Core metrics tracked: Classification Report (Precision, Recall, F1-Score).
- Generation of confusion matrices and performance distribution curves.

---

## 📂 Project Structure
```text
├── preprocess.py             # Data cleaning and feature engineering
├── train.py                  # Model training and cross-validation
├── evaluate.py               # Performance metrics and evaluation plots
└── README.md                 # Project documentation
