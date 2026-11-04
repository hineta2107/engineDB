"""
EngineFaultDB - Experiment: Paper-Exact Preprocessing vs Yeo-Johnson Preprocessing
Tests the exact preprocessing pipeline described in Section 2.2 of the EngineFaultDB IEEE paper:
1. Missing value handling: Mean imputation (if any missing values)
2. Feature scaling: Direct Z-score normalization (StandardScaler: X_scaled = (X - mu) / sigma)
   WITHOUT prior Yeo-Johnson power transformation.
3. Tests both Full 14 Features and Top-5 Features using the optimal Tuned Random Forest hyperparameters:
   {'n_estimators': 300, 'max_depth': 22, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'}
4. Compares Locked Test performance directly against our previous Yeo-Johnson approach.
"""

import os
import sys
import json
import logging
from typing import Dict, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PaperPreprocessingExperiment")

RAW_DATA_PATH = "EngineFaultDB_Final.csv"
OUTPUT_DIR = "outputs/experiments_paper_preprocessing"
TARGET_COL = "Fault"
TOP_5_FEATURES = ["CO", "Force", "Consumption L/H", "Power", "HC"]

TUNED_HYPERPARAMS = {
    "n_estimators": 300,
    "max_depth": 22,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1
}


def run_paper_preprocessing_experiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("=" * 70)
    logger.info("TESTING PAPER-EXACT PREPROCESSING (MEAN IMPUTATION + DIRECT Z-SCORE)")
    logger.info("=" * 70)

    # 1. Load Raw Dataset directly from original source
    df_raw = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Loaded raw dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    all_features = [c for c in df_raw.columns if c != TARGET_COL]

    # Replicate exact stratified split (Train 70%, Val 15%, Locked Test 15%)
    # Use random_state=42 for deterministic 1:1 fairness
    X_raw_all = df_raw[all_features]
    y_raw = df_raw[TARGET_COL]

    X_temp_all, X_test_all, y_temp, y_test = train_test_split(
        X_raw_all, y_raw, test_size=0.15, stratify=y_raw, random_state=42
    )
    val_relative = 0.15 / 0.85
    X_train_all, X_val_all, y_train, y_val = train_test_split(
        X_temp_all, y_temp, test_size=val_relative, stratify=y_temp, random_state=42
    )

    # Preprocessing strictly as in Paper:
    # A. Mean Imputation
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train_all)
    X_val_imp = imputer.transform(X_val_all)
    X_test_imp = imputer.transform(X_test_all)

    # B. Direct Z-score Normalization: X_scaled = (X - mu) / sigma
    scaler = StandardScaler()
    X_train_zscore = scaler.fit_transform(X_train_imp)
    X_val_zscore = scaler.transform(X_val_imp)
    X_test_zscore = scaler.transform(X_test_imp)

    # Prepare DataFrame versions to slice Top-5 easily
    X_train_df = pd.DataFrame(X_train_zscore, columns=all_features)
    X_val_df = pd.DataFrame(X_val_zscore, columns=all_features)
    X_test_df = pd.DataFrame(X_test_zscore, columns=all_features)

    # --------------------------------------------------------------------------
    # EVALUATION 1: TUNED RANDOM FOREST ON TOP-5 FEATURES WITH DIRECT Z-SCORE
    # --------------------------------------------------------------------------
    logger.info("\n--- Experiment 1: Tuned RF on Top-5 Features (Paper Z-Score) ---")
    rf_top5_zscore = RandomForestClassifier(**TUNED_HYPERPARAMS)
    rf_top5_zscore.fit(X_train_df[TOP_5_FEATURES], y_train)

    preds_top5 = rf_top5_zscore.predict(X_test_df[TOP_5_FEATURES])
    acc_top5 = accuracy_score(y_test, preds_top5)
    p_top5, r_top5, f1_top5, _ = precision_recall_fscore_support(y_test, preds_top5, average="macro")

    # --------------------------------------------------------------------------
    # EVALUATION 2: TUNED RANDOM FOREST ON FULL 14 FEATURES WITH DIRECT Z-SCORE
    # --------------------------------------------------------------------------
    logger.info("\n--- Experiment 2: Tuned RF on Full 14 Features (Paper Z-Score) ---")
    rf_full_zscore = RandomForestClassifier(**TUNED_HYPERPARAMS)
    rf_full_zscore.fit(X_train_df, y_train)

    preds_full = rf_full_zscore.predict(X_test_df)
    acc_full = accuracy_score(y_test, preds_full)
    p_full, r_full, f1_full, _ = precision_recall_fscore_support(y_test, preds_full, average="macro")

    # --------------------------------------------------------------------------
    # COMPARISON TABLE: Z-SCORE (PAPER) VS YEO-JOHNSON + Z-SCORE
    # --------------------------------------------------------------------------
    comparison_records = [
        {
            "Preprocessing": "Paper Z-Score (Top-5 Features)",
            "Features_Count": 5,
            "Accuracy": f"{acc_top5*100:.2f}%",
            "F1_Macro": f"{f1_top5:.4f}",
            "Precision": f"{p_top5:.4f}",
            "Recall": f"{r_top5:.4f}"
        },
        {
            "Preprocessing": "Yeo-Johnson + Scaling (Top-5 Features)",
            "Features_Count": 5,
            "Accuracy": "74.79%",
            "F1_Macro": "0.7561",
            "Precision": "0.7562",
            "Recall": "0.7562"
        },
        {
            "Preprocessing": "Paper Z-Score (Full 14 Features)",
            "Features_Count": 14,
            "Accuracy": f"{acc_full*100:.2f}%",
            "F1_Macro": f"{f1_full:.4f}",
            "Precision": f"{p_full:.4f}",
            "Recall": f"{r_full:.4f}"
        }
    ]

    comp_df = pd.DataFrame(comparison_records)
    csv_path = os.path.join(OUTPUT_DIR, "paper_zscore_vs_yeojohnson_comparison.csv")
    comp_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 85)
    print("BENCHMARK: PAPER Z-SCORE NORMALIZATION VS YEO-JOHNSON + SCALING")
    print("=" * 85)
    print(comp_df.to_string(index=False))
    print("=" * 85 + "\n")

    return comp_df


if __name__ == "__main__":
    run_paper_preprocessing_experiment()
