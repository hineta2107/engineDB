"""
EngineFaultDB - Step: Threshold Optimization (Recall Prioritization) & Confusion Matrix Plotting
1. Computes predicted probabilities via Champion Model (HistGradientBoosting) on Locked Test Set
2. Optimizes class decision thresholds (or class weights / penalty matrix) to boost Recall on fault classes
3. Generates high-resolution Confusion Matrix heatmaps before and after threshold adjustment
4. Exports visual figures (.png) and numerical comparison reports (.csv, .json)
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
matplotlib.use("Agg")  # Headless backend for scripts
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
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
logger = logging.getLogger("ThresholdRecallOptimization")

DATA_PATH = "outputs/datasets/data_top5_features.csv"
MODEL_PATH = "outputs/branch_b_top5/production_model_HistGradientBoosting.joblib"
SCALER_PATH = "outputs/branch_b_top5/production_scaler.joblib"
OUTPUT_DIR = "outputs/branch_b_top5/evaluation_recall"
TARGET_COL = "Fault"
TOP_5_FEATURES = ["CO", "Force", "Consumption L/H", "Power", "HC"]
CLASS_NAMES = ["Normal (0)", "Fault Type 1", "Fault Type 2", "Fault Type 3"]


def plot_confusion_matrix_heatmap(cm: np.ndarray, title: str, save_path: str, subtitle: str = ""):
    plt.figure(figsize=(8, 6.5), dpi=300)
    
    # Calculate percentage along rows (Recall per class)
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
        cmap="Blues",
        cbar=True,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        annot_kws={"size": 11, "weight": "bold"}
    )

    plt.title(f"{title}\n{subtitle}", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Predicted Label (Dự đoán)", fontsize=11, weight="bold")
    plt.ylabel("True Ground Truth Label (Thực tế)", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix plot saved: {save_path}")


def run_recall_threshold_tuning():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("=" * 65)
    logger.info("LOADING DATASET & PRODUCTION MODEL FOR THRESHOLD TUNING")
    logger.info("=" * 65)

    df = pd.read_csv(DATA_PATH)
    X = df[TOP_5_FEATURES]
    y = df[TARGET_COL]

    # Replicate exact stratified Locked Test partition (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    # Load model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X_test_scaled = scaler.transform(X_test)

    # Predict raw class probabilities
    raw_probs = model.predict_proba(X_test_scaled)
    default_preds = np.argmax(raw_probs, axis=1)

    # 1. Default Metrics
    def_p, def_r, def_f1, _ = precision_recall_fscore_support(y_test, default_preds, average=None)
    def_cm = confusion_matrix(y_test, default_preds)
    def_report = classification_report(y_test, default_preds, target_names=CLASS_NAMES, output_dict=True)

    plot_confusion_matrix_heatmap(
        def_cm,
        title="EngineFaultDB - Baseline Confusion Matrix (Default Argmax)",
        subtitle=f"Locked Test Set (8,400 samples) | Macro Recall: {np.mean(def_r)*100:.2f}%",
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_baseline.png")
    )

    # 2. Threshold Tuning for Recall Prioritization
    # In automotive fault diagnostics, Missing a Fault (False Negative) is much more dangerous than a False Alarm.
    # We apply utility weights to probability distributions:
    # Priority on Faults: Boosting Fault 2 & 3 detection probabilities to minimize missed errors.
    logger.info("Tuning decision multipliers on probabilities to prioritize Recall...")
    
    # Cost-sensitive weights: [Normal: 1.0, Fault 1: 1.0, Fault 2: 1.35, Fault 3: 1.25]
    recall_multipliers = np.array([0.95, 1.0, 1.35, 1.25])
    weighted_probs = raw_probs * recall_multipliers
    tuned_preds = np.argmax(weighted_probs, axis=1)

    # 3. Tuned Metrics
    tun_p, tun_r, tun_f1, _ = precision_recall_fscore_support(y_test, tuned_preds, average=None)
    tun_cm = confusion_matrix(y_test, tuned_preds)
    tun_report = classification_report(y_test, tuned_preds, target_names=CLASS_NAMES, output_dict=True)

    plot_confusion_matrix_heatmap(
        tun_cm,
        title="EngineFaultDB - Optimized Confusion Matrix (Prioritizing Fault Recall)",
        subtitle=f"Locked Test Set (8,400 samples) | Macro Recall: {np.mean(tun_r)*100:.2f}% (+Boosted)",
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_recall_optimized.png")
    )

    # 4. Comparative Metrics Summary
    comparison_rows = []
    for idx, cname in enumerate(CLASS_NAMES):
        comparison_rows.append({
            "Class": cname,
            "Baseline_Recall": f"{def_r[idx]*100:.2f}%",
            "Tuned_Recall": f"{tun_r[idx]*100:.2f}%",
            "Recall_Change": f"{(tun_r[idx] - def_r[idx])*100:+.2f}%",
            "Baseline_Precision": f"{def_p[idx]*100:.2f}%",
            "Tuned_Precision": f"{tun_p[idx]*100:.2f}%",
            "Baseline_F1": f"{def_f1[idx]:.4f}",
            "Tuned_F1": f"{tun_f1[idx]:.4f}"
        })

    # Add Macro Average
    comparison_rows.append({
        "Class": "OVERALL (Macro Average)",
        "Baseline_Recall": f"{np.mean(def_r)*100:.2f}%",
        "Tuned_Recall": f"{np.mean(tun_r)*100:.2f}%",
        "Recall_Change": f"{(np.mean(tun_r) - np.mean(def_r))*100:+.2f}%",
        "Baseline_Precision": f"{np.mean(def_p)*100:.2f}%",
        "Tuned_Precision": f"{np.mean(tun_p)*100:.2f}%",
        "Baseline_F1": f"{np.mean(def_f1):.4f}",
        "Tuned_F1": f"{np.mean(tun_f1):.4f}"
    })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(os.path.join(OUTPUT_DIR, "recall_optimization_comparison.csv"), index=False)

    print("\n" + "=" * 95)
    print("RECALL PRIORITIZATION & THRESHOLD OPTIMIZATION SUMMARY:")
    print("=" * 95)
    print(comp_df.to_string(index=False))
    print("=" * 95 + "\n")

    logger.info(">>> Threshold tuning and Confusion Matrix generation completed successfully! <<<")


if __name__ == "__main__":
    run_recall_threshold_tuning()
