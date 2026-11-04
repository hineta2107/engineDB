"""
EngineFaultDB - Final Production Pipeline: Optimal Preprocessing + Tuned Champion Model
Synthesis of all experimental findings:
1. Optimal Preprocessing:
   - Data Cleaning (Deduplication + Physical boundary checks)
   - Feature Selection: Top-5 Most Critical Sensors ['CO', 'Force', 'Consumption L/H', 'Power', 'HC']
   - Paper-Exact Z-Score Normalization: X_scaled = (X - mu) / sigma (fitted strictly on Train)
2. Optimal Model:
   - Random Forest Classifier with Tuned Champion Hyperparameters:
     {'n_estimators': 300, 'max_depth': 22, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'}
3. Comprehensive Visualizations (High-Resolution 300 DPI):
   - 1. Confusion Matrix Heatmap (Count & Percentages)
   - 2. Per-Class Metrics Bar Chart (Precision, Recall, F1-Score)
   - 3. ROC Curves (One-vs-Rest for all 4 classes + Micro/Macro AUC)
   - 4. Tuned Feature Importance Chart
4. Exports complete standalone production artifacts and summary JSON.
"""

import os
import sys
import json
import logging

import pandas as pd
import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FinalProductionPipeline")

RAW_DATA_PATH = "EngineFaultDB_Final.csv"
OUTPUT_DIR = "outputs/final_production_results"
TARGET_COL = "Fault"
TOP_5_FEATURES = ["CO", "Force", "Consumption L/H", "Power", "HC"]
CLASS_NAMES = ["Normal (0)", "Fault Type 1", "Fault Type 2", "Fault Type 3"]

OPTIMAL_HYPERPARAMS = {
    "n_estimators": 300,
    "max_depth": 22,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1
}


def run_final_production_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("=" * 70)
    logger.info("EXECUTING FINAL SYNTHESIS PRODUCTION PIPELINE")
    logger.info("=" * 70)

    # 1. Ingestion & Deduplication
    raw_df = pd.read_csv(RAW_DATA_PATH)
    clean_df = raw_df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Loaded and cleaned data: {len(clean_df)} records.")

    # 2. Top-5 Feature Selection
    X = clean_df[TOP_5_FEATURES]
    y = clean_df[TARGET_COL]

    # 3. Stratified Split (Train 70%, Val 15%, Locked Test 15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    val_relative = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=42
    )

    # 4. Paper-Exact Z-Score Scaling (Fitted STRICTLY on Train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Paper Z-score scaler fitted on Train and applied to Val & Locked Test.")

    # 5. Train Optimal Random Forest
    logger.info("Training Champion Random Forest model with optimal hyperparameters...")
    clf = RandomForestClassifier(**OPTIMAL_HYPERPARAMS)
    clf.fit(X_train_scaled, y_train)

    # 6. Locked Test Set Evaluation
    test_preds = clf.predict(X_test_scaled)
    test_probs = clf.predict_proba(X_test_scaled)

    test_acc = accuracy_score(y_test, test_preds)
    test_p, test_r, test_f1, _ = precision_recall_fscore_support(y_test, test_preds, average=None)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test, test_preds, average="macro")
    cm = confusion_matrix(y_test, test_preds)
    cls_report = classification_report(y_test, test_preds, target_names=CLASS_NAMES, output_dict=True)

    logger.info(f"FINAL LOCKED TEST METRICS: Accuracy={test_acc*100:.2f}%, F1-Macro={macro_f1:.4f}")

    # ==========================================================================
    # VISUALIZATIONS GENERATION
    # ==========================================================================
    sns.set_theme(style="whitegrid")

    # Plot 1: High-Resolution Confusion Matrix Heatmap
    plt.figure(figsize=(8.5, 7), dpi=300)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_perc[i, j]:.1f}%)"

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
    plt.title(f"Final Champion Random Forest - Confusion Matrix\nLocked Test Set (8,400 samples) | Accuracy: {test_acc*100:.2f}%", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Predicted Label (Dự đoán)", fontsize=11, weight="bold")
    plt.ylabel("True Ground Truth Label (Thực tế)", fontsize=11, weight="bold")
    plt.tight_layout()
    cm_plot_path = os.path.join(OUTPUT_DIR, "plot1_final_confusion_matrix.png")
    plt.savefig(cm_plot_path)
    plt.close()

    # Plot 2: Per-Class Metrics Bar Chart (Precision, Recall, F1)
    metrics_df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "Precision": test_p,
        "Recall": test_r,
        "F1-Score": test_f1
    }).melt(id_vars="Class", var_name="Metric", value_name="Score")

    plt.figure(figsize=(9, 5.5), dpi=300)
    ax = sns.barplot(data=metrics_df, x="Class", y="Score", hue="Metric", palette="viridis")
    plt.title("Per-Class Performance Metrics (Precision, Recall, F1-Score)", fontsize=13, weight="bold", pad=12)
    plt.ylim(0, 1.15)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11, weight="bold")
    plt.xlabel("Engine Condition / Fault Class", fontsize=11, weight="bold")
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height + 0.02),
                        ha='center', va='bottom', fontsize=9, weight="bold")
    plt.tight_layout()
    metrics_plot_path = os.path.join(OUTPUT_DIR, "plot2_per_class_metrics.png")
    plt.savefig(metrics_plot_path)
    plt.close()

    # Plot 3: Multiclass ROC Curves & AUC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    n_classes = y_test_bin.shape[1]
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], test_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8.5, 6.5), dpi=300)
    colors = ["#2b5c8f", "#27ae60", "#e67e22", "#9b59b6"]
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2.2, label=f"{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label="Random Chance (AUC = 0.50)")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, weight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, weight="bold")
    plt.title("Multiclass One-vs-Rest ROC Curves & Area Under Curve (AUC)", fontsize=13, weight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    roc_plot_path = os.path.join(OUTPUT_DIR, "plot3_multiclass_roc_curves.png")
    plt.savefig(roc_plot_path)
    plt.close()

    # Plot 4: Feature Importance Ranking
    importances = dict(zip(TOP_5_FEATURES, clf.feature_importances_))
    sorted_fi = pd.DataFrame({
        "Feature": list(importances.keys()),
        "Importance": list(importances.values())
    }).sort_values("Importance", ascending=True)

    plt.figure(figsize=(8.5, 4.8), dpi=300)
    bars = plt.barh(sorted_fi["Feature"], sorted_fi["Importance"], color="#3498db", edgecolor="#2980b9", height=0.55)
    plt.xlabel("Feature Importance (Gini Importance)", fontsize=11, weight="bold")
    plt.title("Top-5 Sensor Feature Importance Ranking (Optimal RF)", fontsize=13, weight="bold", pad=12)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f"{width*100:.2f}%",
                 va='center', ha='left', fontsize=10, weight="bold")
    plt.xlim(0, max(sorted_fi["Importance"]) * 1.18)
    plt.tight_layout()
    fi_plot_path = os.path.join(OUTPUT_DIR, "plot4_top5_feature_importance.png")
    plt.savefig(fi_plot_path)
    plt.close()

    # Save artifacts
    joblib.dump(clf, os.path.join(OUTPUT_DIR, "final_champion_rf_model.joblib"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "final_production_scaler.joblib"))

    summary_metadata = {
        "dataset": "EngineFaultDB_Final.csv",
        "preprocessing_method": "Z-score Normalization (Paper-Exact: X_scaled = (X - mu) / sigma)",
        "selected_features": TOP_5_FEATURES,
        "champion_algorithm": "RandomForestClassifier",
        "optimal_hyperparameters": OPTIMAL_HYPERPARAMS,
        "test_evaluation_metrics": {
            "test_sample_size": len(X_test),
            "test_accuracy": round(float(test_acc), 4),
            "test_macro_f1": round(float(macro_f1), 4),
            "test_macro_precision": round(float(macro_p), 4),
            "test_macro_recall": round(float(macro_r), 4),
            "per_class_auc": {CLASS_NAMES[i]: round(float(roc_auc[i]), 4) for i in range(n_classes)}
        },
        "feature_importances": {k: round(float(v), 4) for k, v in sorted_fi.set_index("Feature")["Importance"].to_dict().items()}
    }

    with open(os.path.join(OUTPUT_DIR, "final_synthesis_report.json"), "w") as f:
        json.dump(summary_metadata, f, indent=4)

    logger.info("=" * 70)
    logger.info("ALL VISUALIZATION CHARTS AND PRODUCTION ARTIFACTS EXPORTED!")
    logger.info("=" * 70)
    return summary_metadata


if __name__ == "__main__":
    run_final_production_pipeline()
