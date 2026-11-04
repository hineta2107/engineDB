"""
Branch B - Step B5 & B6: Advanced Hyperparameter Tuning for Champion Random Forest
Focuses on the winning model (RandomForest on Top-5 Features).
Performs Systematic Grid Search CV across:
- n_estimators: [100, 200, 300]
- max_depth: [12, 16, 20, None]
- min_samples_split: [2, 5, 10]
- min_samples_leaf: [1, 2, 4]
- max_features: ['sqrt', 'log2', None]
Evaluates baseline vs tuned model on Locked Test Set (8,400 samples).
Generates side-by-side Confusion Matrix and exports production artifacts.
"""

import os
import sys
import json
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
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
logger = logging.getLogger("RF_Tuning")

DATA_PATH = "outputs/datasets/data_top5_features.csv"
OUTPUT_DIR = "outputs/branch_b_top5/rf_hyperparameter_tuning"
TARGET_COL = "Fault"
TOP_5_FEATURES = ["CO", "Force", "Consumption L/H", "Power", "HC"]
CLASS_NAMES = ["Normal (0)", "Fault Type 1", "Fault Type 2", "Fault Type 3"]


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str, color_map: str = "Blues"):
    plt.figure(figsize=(8, 6.5), dpi=300)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            annot[i, j] = f"{c:,}\n({p:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap=color_map,
        cbar=True,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        annot_kws={"size": 11, "weight": "bold"}
    )

    plt.title(title, fontsize=13, weight="bold", pad=12)
    plt.xlabel("Predicted Label (Dự đoán)", fontsize=11, weight="bold")
    plt.ylabel("True Ground Truth Label (Thực tế)", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved confusion matrix: {save_path}")


def run_random_forest_tuning():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("=" * 65)
    logger.info("RUNNING GRID SEARCH HYPERPARAMETER TUNING FOR RANDOM FOREST")
    logger.info("=" * 65)

    df = pd.read_csv(DATA_PATH)
    X = df[TOP_5_FEATURES]
    y = df[TARGET_COL]

    # Replicate exact stratified Locked Test partition (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    val_relative = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 1. Baseline Random Forest (Defaults: n_estimators=100, max_depth=12)
    logger.info("1/3 Evaluating Baseline Random Forest...")
    rf_baseline = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_baseline.fit(X_train_scaled, y_train)

    base_preds = rf_baseline.predict(X_test_scaled)
    base_acc = accuracy_score(y_test, base_preds)
    base_p, base_r, base_f1, _ = precision_recall_fscore_support(y_test, base_preds, average="macro")
    base_cm = confusion_matrix(y_test, base_preds)

    plot_confusion_matrix(
        base_cm,
        title=f"Baseline Random Forest (Depth=12, N=100)\nLocked Test Accuracy: {base_acc*100:.2f}% | F1: {base_f1:.4f}",
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_rf_baseline.png"),
        color_map="Blues"
    )

    # 2. Grid Search CV on 5-Fold Stratified Split
    logger.info("2/3 Executing Systematic Hyperparameter Tuning Grid Search...")
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [14, 18, 22, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    logger.info(f"Best Hyperparameters Found: {best_params} (CV F1-Macro: {best_cv_score:.4f})")

    # 3. Fit Best Tuned Model on Train + Val, and evaluate on Locked Test Set
    logger.info("3/3 Evaluating Tuned Champion Model on Locked Test Set (8,400 samples)...")
    best_rf = grid_search.best_estimator_

    tuned_preds = best_rf.predict(X_test_scaled)
    tuned_acc = accuracy_score(y_test, tuned_preds)
    tuned_p, tuned_r, tuned_f1, _ = precision_recall_fscore_support(y_test, tuned_preds, average="macro")
    tuned_cm = confusion_matrix(y_test, tuned_preds)
    cls_report = classification_report(y_test, tuned_preds, target_names=CLASS_NAMES, output_dict=True)

    plot_confusion_matrix(
        tuned_cm,
        title=f"Tuned Champion Random Forest ({best_params['n_estimators']} trees, Depth={best_params['max_depth']})\nLocked Test Accuracy: {tuned_acc*100:.2f}% | F1: {tuned_f1:.4f}",
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_rf_tuned.png"),
        color_map="Greens"
    )

    # 4. Feature Importance of Tuned Champion Model
    importances = dict(zip(TOP_5_FEATURES, best_rf.feature_importances_))
    sorted_importances = {k: round(float(v), 4) for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)}

    # 5. Comparison Table
    comparison_df = pd.DataFrame([
        {
            "Stage": "Baseline RF (N=100, Depth=12)",
            "Test Accuracy": f"{base_acc*100:.2f}%",
            "Test F1-Macro": f"{base_f1:.4f}",
            "Test Precision": f"{base_p:.4f}",
            "Test Recall": f"{base_r:.4f}"
        },
        {
            "Stage": "Tuned Champion RF (Optimized)",
            "Test Accuracy": f"{tuned_acc*100:.2f}%",
            "Test F1-Macro": f"{tuned_f1:.4f}",
            "Test Precision": f"{tuned_p:.4f}",
            "Test Recall": f"{tuned_r:.4f}"
        }
    ])
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "tuning_comparison_metrics.csv"), index=False)

    # Export production artifacts
    joblib.dump(best_rf, os.path.join(OUTPUT_DIR, "champion_random_forest_tuned.joblib"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "champion_scaler.joblib"))

    # Save summary metadata
    meta = {
        "best_hyperparameters": best_params,
        "best_cross_val_f1": round(float(best_cv_score), 4),
        "test_accuracy": round(float(tuned_acc), 4),
        "test_f1_macro": round(float(tuned_f1), 4),
        "feature_importances": sorted_importances,
        "confusion_matrix": tuned_cm.tolist(),
        "classification_report": cls_report
    }
    with open(os.path.join(OUTPUT_DIR, "rf_tuning_metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print("\n" + "=" * 80)
    print("RANDOM FOREST HYPERPARAMETER TUNING: BEFORE VS AFTER COMPARISON:")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print(f"\nOptimal Hyperparameters: {best_params}")
    print(f"Tuned Feature Importances: {sorted_importances}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_random_forest_tuning()

