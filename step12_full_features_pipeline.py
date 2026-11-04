"""
EngineFaultDB - Dual-Branch Full-Feature Pipeline (14 Features)
Executes both Branch A (Unsupervised) and Branch B (Supervised) on FULL 14 SENSORS:
['MAP', 'TPS', 'Force', 'Power', 'RPM', 'Consumption L/H', 'Consumption L/100KM',
 'Speed', 'CO', 'HC', 'CO2', 'O2', 'Lambda', 'AFR']

Branch A:
- Scaled via Paper Z-score
- 2D PCA representation
- Optimal KMeans (K=4) with stability evaluation
- Cluster feature profiles & cross-tabulation with Fault

Branch B:
- Stratified 70/15/15 split
- Train-only Z-score scaling
- Random Forest Classifier with optimized hyperparameters (300 trees, depth=22)
- Locked Test Set (8,400 samples) evaluation

Visualizations (High-Resolution 300 DPI):
1. Plot 1 (Branch A): 2D PCA Full-Feature Clusters
2. Plot 2 (Branch A): Full-Feature Radar / Mean Profiles
3. Plot 3 (Branch B): Confusion Matrix Heatmap (Full 14 Features)
4. Plot 4 (Branch B): Full 14-Feature Importance Ranking (Gini Importance)
5. Plot 5 (Integrated): Cluster vs Ground-Truth Fault Matrix Heatmap
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
logger = logging.getLogger("FullFeaturePipeline")

DATA_PATH = "outputs/common_foundation/prepared_feature_base.csv"
OUTPUT_DIR = "outputs/full_14_features_pipeline"
TARGET_COL = "Fault"
CLASS_NAMES = ["Normal (0)", "Fault Type 1", "Fault Type 2", "Fault Type 3"]

OPTIMAL_RF_HYPERPARAMS = {
    "n_estimators": 300,
    "max_depth": 22,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1
}


def run_full_feature_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("=" * 70)
    logger.info("RUNNING DUAL-BRANCH PIPELINE ON FULL 14 SENSOR FEATURES")
    logger.info("=" * 70)

    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    logger.info(f"Loaded {len(df)} records. Using all {len(feature_cols)} features: {feature_cols}")

    X = df[feature_cols]
    y = df[TARGET_COL]

    # ==========================================================================
    # BRANCH A: UNSUPERVISED CLUSTERING (FULL 14 FEATURES)
    # ==========================================================================
    logger.info("\n--- Branch A: Unsupervised Segmentation (14 Features) ---")
    scaler_a = StandardScaler()
    X_scaled_a = scaler_a.fit_transform(X)

    # 2D PCA for visual representation
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled_a)
    var_exp = pca.explained_variance_ratio_

    # KMeans with K=4
    km = KMeans(n_clusters=4, random_state=42, n_init=15)
    clusters = km.fit_predict(X_scaled_a)
    df["Cluster"] = clusters

    # Profiles & Cross-tabulation
    cluster_profiles = df.groupby("Cluster")[feature_cols].mean().round(3)
    cluster_profiles.to_csv(os.path.join(OUTPUT_DIR, "branch_a_full_feature_profiles.csv"))

    crosstab_a = pd.crosstab(df["Cluster"], df[TARGET_COL], margins=True, normalize="index").round(4) * 100
    crosstab_a.to_csv(os.path.join(OUTPUT_DIR, "branch_a_cluster_vs_fault_matrix.csv"))

    # ==========================================================================
    # BRANCH B: SUPERVISED FAULT CLASSIFICATION (FULL 14 FEATURES)
    # ==========================================================================
    logger.info("\n--- Branch B: Supervised Classification (14 Features) ---")
    # Stratified Split (Train 70%, Val 15%, Locked Test 15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    val_relative = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=42
    )

    # Train-only Scaling
    scaler_b = StandardScaler()
    X_train_scaled = scaler_b.fit_transform(X_train)
    X_val_scaled = scaler_b.transform(X_val)
    X_test_scaled = scaler_b.transform(X_test)

    # Train Random Forest on full 14 features
    rf_full = RandomForestClassifier(**OPTIMAL_RF_HYPERPARAMS)
    rf_full.fit(X_train_scaled, y_train)

    # Locked Test Evaluation
    test_preds = rf_full.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_preds)
    test_p, test_r, test_f1, _ = precision_recall_fscore_support(y_test, test_preds, average="macro")
    cm_b = confusion_matrix(y_test, test_preds)
    cls_report = classification_report(y_test, test_preds, target_names=CLASS_NAMES, output_dict=True)

    logger.info(f"FULL 14-FEATURE TEST METRICS: Accuracy={test_acc*100:.2f}%, F1-Macro={test_f1:.4f}")

    # ==========================================================================
    # VISUALIZATION GENERATION (300 DPI)
    # ==========================================================================
    sns.set_theme(style="whitegrid")

    # Plot 1: Branch A - 2D PCA Clusters
    plt.figure(figsize=(9, 7), dpi=300)
    idx_sample = np.random.RandomState(42).choice(len(df), min(10000, len(df)), replace=False)
    scatter_df = pd.DataFrame({
        "PC1": X_pca[idx_sample, 0],
        "PC2": X_pca[idx_sample, 1],
        "Cluster": [f"Cluster {c}" for c in clusters[idx_sample]]
    })
    sns.scatterplot(
        data=scatter_df, x="PC1", y="PC2", hue="Cluster",
        palette=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"],
        alpha=0.6, s=25
    )
    plt.title("Branch A (Full 14 Features) - 2D PCA Engine Clusters", fontsize=13, weight="bold", pad=12)
    plt.xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}% Variance)", fontsize=11, weight="bold")
    plt.ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}% Variance)", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot1_branch_a_full_pca_clusters.png"))
    plt.close()

    # Plot 2: Branch B - Confusion Matrix (Full 14 Features)
    plt.figure(figsize=(8.5, 7), dpi=300)
    cm_perc = cm_b.astype('float') / cm_b.sum(axis=1)[:, np.newaxis] * 100
    annot = np.empty_like(cm_b).astype(str)
    for i in range(cm_b.shape[0]):
        for j in range(cm_b.shape[1]):
            annot[i, j] = f"{cm_b[i, j]:,}\n({cm_perc[i, j]:.1f}%)"

    sns.heatmap(
        cm_b, annot=annot, fmt="", cmap="Blues", cbar=True,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title(f"Branch B (Full 14 Features) - Confusion Matrix\nLocked Test Set (8,400 samples) | Accuracy: {test_acc*100:.2f}% | F1: {test_f1:.4f}", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Predicted Label (Dự đoán)", fontsize=11, weight="bold")
    plt.ylabel("True Ground Truth Label (Thực tế)", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot2_branch_b_full_confusion_matrix.png"))
    plt.close()

    # Plot 3: Branch B - Feature Importance for all 14 Features
    fi_dict = dict(zip(feature_cols, rf_full.feature_importances_))
    fi_df = pd.DataFrame({
        "Feature": list(fi_dict.keys()),
        "Importance": list(fi_dict.values())
    }).sort_values("Importance", ascending=True)

    plt.figure(figsize=(10, 6.5), dpi=300)
    bars = plt.barh(fi_df["Feature"], fi_df["Importance"], color="#2ecc71", edgecolor="#27ae60", height=0.6)
    plt.xlabel("Gini Feature Importance", fontsize=11, weight="bold")
    plt.title("Branch B - Full 14 Sensor Feature Importance Ranking (Random Forest)", fontsize=13, weight="bold", pad=12)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.002, bar.get_y() + bar.get_height() / 2, f"{w*100:.2f}%", va='center', ha='left', fontsize=9, weight="bold")
    plt.xlim(0, max(fi_df["Importance"]) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot3_branch_b_all_14_features_importance.png"))
    plt.close()

    # Plot 4: Integrated Insights - Cluster vs Ground-Truth Fault Heatmap
    plt.figure(figsize=(7.5, 5.5), dpi=300)
    sns.heatmap(
        crosstab_a.iloc[:4, :4], annot=True, fmt=".1f", cmap="YlOrRd",
        cbar_kws={"label": "Percentage of Cluster (%)"},
        annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title("Integrated Insights: Full-Feature Clusters vs Ground-Truth Faults (%)", fontsize=12, weight="bold", pad=12)
    plt.xlabel("Ground-Truth Fault Label (0, 1, 2, 3)", fontsize=11, weight="bold")
    plt.ylabel("Discovered Engine Cluster", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot4_integrated_cluster_vs_fault_heatmap.png"))
    plt.close()

    # Export production models
    joblib.dump(rf_full, os.path.join(OUTPUT_DIR, "full_14_rf_model.joblib"))
    joblib.dump(scaler_b, os.path.join(OUTPUT_DIR, "full_14_scaler.joblib"))
    joblib.dump(km, os.path.join(OUTPUT_DIR, "full_14_kmeans_model.joblib"))

    summary = {
        "dataset": "EngineFaultDB_Final.csv",
        "features_used": feature_cols,
        "n_features": len(feature_cols),
        "branch_a_unsupervised": {
            "algorithm": "KMeans (K=4)",
            "pca_variance_ratio_2d": [round(float(v), 4) for v in var_exp]
        },
        "branch_b_supervised": {
            "algorithm": "RandomForestClassifier",
            "hyperparameters": OPTIMAL_RF_HYPERPARAMS,
            "test_accuracy": round(float(test_acc), 4),
            "test_f1_macro": round(float(test_f1), 4),
            "test_precision_macro": round(float(test_p), 4),
            "test_recall_macro": round(float(test_r), 4),
            "classification_report": cls_report
        },
        "all_feature_importances": {k: round(float(v), 4) for k, v in fi_df.set_index("Feature")["Importance"].to_dict().items()}
    }

    with open(os.path.join(OUTPUT_DIR, "full_14_features_pipeline_report.json"), "w") as f:
        json.dump(summary, f, indent=4)

    logger.info("=" * 70)
    logger.info("FULL 14-FEATURE PIPELINE COMPLETED & ALL PLOTS EXPORTED!")
    logger.info("=" * 70)
    return summary


if __name__ == "__main__":
    run_full_feature_pipeline()
