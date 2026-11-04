"""
EngineFaultDB - Experiment: Full Features (14) vs Top-5 Features
Compares Random Forest Classifier performance between:
1. Full Dataset (All 14 features)
2. Top-5 Most Important Features: ['CO', 'Force', 'Consumption L/H', 'Power', 'HC']
Evaluates on Stratified Train/Val/Test (Locked Test) with full classification metrics.
"""

import os
import sys
import json
import logging
from typing import Dict, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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
logger = logging.getLogger("RF_Experiment")

SOURCE_DATA_PATH = "outputs/common_foundation/prepared_feature_base_normalized.csv"
OUTPUT_DIR = "outputs/experiments_rf"
DATASETS_DIR = "outputs/datasets"
TARGET_COL = "Fault"

TOP_5_FEATURES = ["CO", "Force", "Consumption L/H", "Power", "HC"]


def prepare_and_save_datasets(source_path: str = SOURCE_DATA_PATH):
    os.makedirs(DATASETS_DIR, exist_ok=True)
    df = pd.read_csv(source_path)

    # 1. Full Dataset (14 features + Target)
    full_path = os.path.join(DATASETS_DIR, "data_full_14_features.csv")
    df.to_csv(full_path, index=False)
    logger.info(f"Dataset 1 (Full 14 features) saved to: {full_path}")

    # 2. Top-5 Features Dataset (5 features + Target)
    top5_df = df[TOP_5_FEATURES + [TARGET_COL]]
    top5_path = os.path.join(DATASETS_DIR, "data_top5_features.csv")
    top5_df.to_csv(top5_path, index=False)
    logger.info(f"Dataset 2 (Top 5 features: {TOP_5_FEATURES}) saved to: {top5_path}")

    return df, top5_df


def train_and_evaluate_rf(df: pd.DataFrame, feature_cols: list, exp_name: str) -> Dict[str, Any]:
    logger.info(f"\n--- Running Random Forest on: {exp_name} ({len(feature_cols)} features) ---")
    X = df[feature_cols]
    y = df[TARGET_COL]

    # Split: Train 70%, Val 15%, Test 15% (Stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    val_relative_size = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, stratify=y_temp, random_state=42
    )

    # 5-Fold Stratified Cross-Validation on Train
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)

    # Fit on full training set
    rf.fit(X_train, y_train)

    # Validation Set Evaluation
    val_preds = rf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_p, val_r, val_f1, _ = precision_recall_fscore_support(y_val, val_preds, average="macro")

    # Locked Test Set Evaluation
    test_preds = rf.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_p, test_r, test_f1, _ = precision_recall_fscore_support(y_test, test_preds, average="macro")
    cls_report = classification_report(y_test, test_preds, output_dict=True)
    cm = confusion_matrix(y_test, test_preds).tolist()

    return {
        "experiment": exp_name,
        "n_features": len(feature_cols),
        "features_list": feature_cols,
        "cv_f1_macro_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_macro_std": round(float(cv_scores.std()), 4),
        "val_accuracy": round(float(val_acc), 4),
        "val_f1_macro": round(float(val_f1), 4),
        "test_accuracy": round(float(test_acc), 4),
        "test_f1_macro": round(float(test_f1), 4),
        "test_precision_macro": round(float(test_p), 4),
        "test_recall_macro": round(float(test_r), 4),
        "confusion_matrix": cm,
        "classification_report": cls_report
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Prepare and export both datasets
    df_full, df_top5 = prepare_and_save_datasets()

    all_features = [c for c in df_full.columns if c != TARGET_COL]

    # 2. Train RF on Dataset 1: Full 14 features
    res_full = train_and_evaluate_rf(df_full, all_features, "Full_14_Features")

    # 3. Train RF on Dataset 2: Top-5 features
    res_top5 = train_and_evaluate_rf(df_top5, TOP_5_FEATURES, "Top_5_Features")

    # 4. Comparative Summary
    comparison_records = [
        {
            "Dataset": "Full (14 Features)",
            "Features Count": res_full["n_features"],
            "CV F1-Macro": f"{res_full['cv_f1_macro_mean']:.4f} ± {res_full['cv_f1_macro_std']:.4f}",
            "Val Accuracy": f"{res_full['val_accuracy']*100:.2f}%",
            "Val F1-Macro": f"{res_full['val_f1_macro']:.4f}",
            "Test Accuracy": f"{res_full['test_accuracy']*100:.2f}%",
            "Test F1-Macro": f"{res_full['test_f1_macro']:.4f}",
            "Test Precision": f"{res_full['test_precision_macro']:.4f}",
            "Test Recall": f"{res_full['test_recall_macro']:.4f}"
        },
        {
            "Dataset": "Top-5 Features",
            "Features Count": res_top5["n_features"],
            "CV F1-Macro": f"{res_top5['cv_f1_macro_mean']:.4f} ± {res_top5['cv_f1_macro_std']:.4f}",
            "Val Accuracy": f"{res_top5['val_accuracy']*100:.2f}%",
            "Val F1-Macro": f"{res_top5['val_f1_macro']:.4f}",
            "Test Accuracy": f"{res_top5['test_accuracy']*100:.2f}%",
            "Test F1-Macro": f"{res_top5['test_f1_macro']:.4f}",
            "Test Precision": f"{res_top5['test_precision_macro']:.4f}",
            "Test Recall": f"{res_top5['test_recall_macro']:.4f}"
        }
    ]

    summary_df = pd.DataFrame(comparison_records)
    summary_csv_path = os.path.join(OUTPUT_DIR, "rf_comparison_full_vs_top5.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    detailed_json_path = os.path.join(OUTPUT_DIR, "rf_comparison_detailed_results.json")
    with open(detailed_json_path, "w") as f:
        json.dump({"full_features": res_full, "top5_features": res_top5}, f, indent=4)

    print("\n" + "=" * 90)
    print("RANDOM FOREST BENCHMARK: FULL 14 FEATURES VS TOP-5 FEATURES")
    print("=" * 90)
    print(summary_df.to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
