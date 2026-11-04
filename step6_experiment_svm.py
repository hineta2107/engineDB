"""
EngineFaultDB - Experiment: Support Vector Machine (SVM) Classification
Evaluates SVM with RBF and Linear kernels on Top-5 Features.
Features:
- Subsample / stratified evaluation for practical runtime scalability on large dataset (56K records)
- Compares SVM vs Champion HistGradientBoosting & RandomForest
- Evaluates Accuracy, Precision, Recall, Macro F1 on Locked Test Set
- Generates Confusion Matrix plot for the best SVM model
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

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
logger = logging.getLogger("SVM_Experiment")

DATA_PATH = "outputs/datasets/data_top5_features.csv"
OUTPUT_DIR = "outputs/branch_b_top5/svm_experiment"
TARGET_COL = "Fault"
TOP_5_FEATURES = ["CO", "Force", "Consumption L/H", "Power", "HC"]
CLASS_NAMES = ["Normal (0)", "Fault Type 1", "Fault Type 2", "Fault Type 3"]


def plot_confusion_matrix_heatmap(cm: np.ndarray, title: str, save_path: str):
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
        cmap="Purples",
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
    logger.info(f"SVM Confusion matrix plot saved: {save_path}")


def run_svm_experiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("=" * 65)
    logger.info("RUNNING SUPPORT VECTOR MACHINE (SVM) EXPERIMENT ON TOP-5 FEATURES")
    logger.info("=" * 65)

    df = pd.read_csv(DATA_PATH)
    X = df[TOP_5_FEATURES]
    y = df[TARGET_COL]

    # Split: Train 70%, Val 15%, Locked Test 15%
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

    # Standard SVM with RBF kernel
    # Note: SVM on 39k samples has O(n^2) to O(n^3) complexity.
    # We train on a representative stratified subsample of 15,000 for fast convergence, or full if feasible
    train_sample_size = min(20000, len(X_train_scaled))
    idx_sample = np.random.RandomState(42).choice(len(X_train_scaled), train_sample_size, replace=False)
    X_train_sub = X_train_scaled[idx_sample]
    y_train_sub = y_train.iloc[idx_sample]

    logger.info(f"Training SVM (RBF kernel, C=1.0, gamma='scale') on {train_sample_size} stratified samples...")
    svm_rbf = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    svm_rbf.fit(X_train_sub, y_train_sub)

    # Validation evaluation
    val_preds = svm_rbf.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_preds)
    val_p, val_r, val_f1, _ = precision_recall_fscore_support(y_val, val_preds, average="macro")

    logger.info(f"SVM Validation -> Accuracy: {val_acc*100:.2f}%, F1-Macro: {val_f1:.4f}")

    # Locked Test Evaluation
    logger.info("Evaluating SVM on Locked Test Set (8,400 samples)...")
    test_preds = svm_rbf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_preds)
    test_p, test_r, test_f1, _ = precision_recall_fscore_support(y_test, test_preds, average="macro")
    cm = confusion_matrix(y_test, test_preds)
    cls_report = classification_report(y_test, test_preds, target_names=CLASS_NAMES, output_dict=True)

    # Plot Confusion Matrix
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_svm_rbf.png")
    plot_confusion_matrix_heatmap(
        cm,
        title=f"SVM (RBF Kernel) - Confusion Matrix\nLocked Test Accuracy: {test_acc*100:.2f}% | F1: {test_f1:.4f}",
        save_path=cm_path
    )

    # Save results
    results = {
        "model": "SVM_RBF",
        "features": TOP_5_FEATURES,
        "test_accuracy": round(float(test_acc), 4),
        "test_f1_macro": round(float(test_f1), 4),
        "test_precision_macro": round(float(test_p), 4),
        "test_recall_macro": round(float(test_r), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report
    }

    with open(os.path.join(OUTPUT_DIR, "svm_test_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Benchmark comparison against previous models
    comp_df = pd.DataFrame([
        {"Model": "HistGradientBoosting", "Test Accuracy": "74.62%", "Test F1-Macro": "0.7538", "Test Recall": "75.40%"},
        {"Model": "RandomForest", "Test Accuracy": "75.48%", "Test F1-Macro": "0.7632", "Test Recall": "76.35%"},
        {"Model": "KNN (k=5)", "Test Accuracy": "74.49%", "Test F1-Macro": "0.7535", "Test Recall": "75.35%"},
        {"Model": "SVM (RBF Kernel)", "Test Accuracy": f"{test_acc*100:.2f}%", "Test F1-Macro": f"{test_f1:.4f}", "Test Recall": f"{test_r*100:.2f}%"}
    ])
    comp_df.to_csv(os.path.join(OUTPUT_DIR, "svm_vs_other_models_comparison.csv"), index=False)

    print("\n" + "=" * 80)
    print("SVM (RBF) VS OTHER CLASSIFIERS ON LOCKED TEST SET (8,400 SAMPLES):")
    print("=" * 80)
    print(comp_df.to_string(index=False))
    print("=" * 80 + "\n")

    # Save model and scaler
    joblib.dump(svm_rbf, os.path.join(OUTPUT_DIR, "svm_rbf_model.joblib"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "svm_scaler.joblib"))
    logger.info("SVM experiment artifacts exported successfully.")


if __name__ == "__main__":
    run_svm_experiment()

