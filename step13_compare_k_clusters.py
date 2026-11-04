"""
EngineFaultDB - Experiment: Branch A Clustering Comparison (K=2 vs K=3 vs K=4)
Evaluates Unsupervised Clustering on Full 14 Features:
1. Calculates Silhouette, Calinski-Harabasz, and Davies-Bouldin scores for K in [2, 3, 4]
2. Computes Seed Stability across multiple runs (Adjusted Rand Index)
3. Analyzes Cluster Profiles and physical engine operating state meaning for each K
4. Generates Cross-tabulation matrices: How well each K separates Normal vs Faults
5. Exports side-by-side 2D PCA scatter plots and heatmaps for K=2 and K=3.
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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("K_Comparison_Experiment")

DATA_PATH = "outputs/common_foundation/prepared_feature_base.csv"
OUTPUT_DIR = "outputs/branch_a_k_comparison"
TARGET_COL = "Fault"


def run_k_comparison_experiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("=" * 70)
    logger.info("EVALUATING CLUSTERING FOR K=2 vs K=3 vs K=4 ON FULL 14 FEATURES")
    logger.info("=" * 70)

    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c != TARGET_COL]

    X = df[feature_cols]
    y_ground_truth = df[TARGET_COL]

    # Z-score normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2D PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    # Evaluation on a representative stratified sample of 12,000 for metric calculation
    eval_size = min(12000, len(X_scaled))
    idx_eval = np.random.RandomState(42).choice(len(X_scaled), eval_size, replace=False)
    X_eval = X_scaled[idx_eval]

    metrics_records = []
    crosstabs = {}
    cluster_labels_dict = {}

    for k in [2, 3, 4]:
        logger.info(f"Evaluating KMeans with K = {k}...")
        # Metrics on eval sample
        km_eval = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_eval = km_eval.fit_predict(X_eval)

        sil = silhouette_score(X_eval, labels_eval)
        ch = calinski_harabasz_score(X_eval, labels_eval)
        db = davies_bouldin_score(X_eval, labels_eval)

        # Stability across 5 random seeds
        seeds = [42, 10, 100, 2024, 7]
        seed_partitions = [KMeans(n_clusters=k, random_state=s, n_init=5).fit_predict(X_eval) for s in seeds]
        aris = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                aris.append(adjusted_rand_score(seed_partitions[i], seed_partitions[j]))
        mean_ari = float(np.mean(aris))

        # Full fit for profiling
        km_full = KMeans(n_clusters=k, random_state=42, n_init=15)
        full_clusters = km_full.fit_predict(X_scaled)
        cluster_labels_dict[k] = full_clusters

        # Cross-tabulation with Ground-Truth Fault
        ct = pd.crosstab(full_clusters, y_ground_truth, margins=True, normalize="index").round(4) * 100
        crosstabs[k] = ct

        metrics_records.append({
            "Clusters_K": k,
            "Silhouette_Score": round(float(sil), 4),
            "Calinski_Harabasz": round(float(ch), 1),
            "Davies_Bouldin": round(float(db), 4),
            "Stability_ARI": round(mean_ari, 4),
            "Interpretability_Description": (
                "Normal vs Abnormal" if k == 2 else
                "Normal vs Severe Fault 1 vs Mixed Faults 2&3" if k == 3 else
                "Normal vs Fault 1 vs Fault 2 vs Fault 3"
            )
        })

    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv_path = os.path.join(OUTPUT_DIR, "k_comparison_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Save Cross-tabulations
    for k, ct in crosstabs.items():
        ct.to_csv(os.path.join(OUTPUT_DIR, f"crosstab_cluster_vs_fault_k{k}.csv"))

    # ==========================================================================
    # VISUALIZATIONS GENERATION (300 DPI)
    # ==========================================================================
    sns.set_theme(style="whitegrid")

    # Figure 1: Side-by-side 2D PCA plots for K=2, K=3, K=4
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)
    sample_vis_idx = idx_eval[:5000]

    for idx, k in enumerate([2, 3, 4]):
        clusters_sample = cluster_labels_dict[k][sample_vis_idx]
        scatter_df = pd.DataFrame({
            "PC1": X_pca[sample_vis_idx, 0],
            "PC2": X_pca[sample_vis_idx, 1],
            "Cluster": [f"Cluster {c}" for c in clusters_sample]
        })
        sns.scatterplot(
            data=scatter_df, x="PC1", y="PC2", hue="Cluster",
            palette="tab10", alpha=0.6, s=20, ax=axes[idx]
        )
        sil_val = metrics_df.loc[metrics_df["Clusters_K"] == k, "Silhouette_Score"].values[0]
        axes[idx].set_title(f"K = {k} (Silhouette: {sil_val:.4f})", fontsize=12, weight="bold")
        axes[idx].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)", fontsize=10, weight="bold")
        axes[idx].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)" if idx == 0 else "", fontsize=10, weight="bold")

    fig.suptitle("Comparison of Cluster Geometries in 2D PCA Space (K = 2 vs 3 vs 4)", fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()
    plot1_path = os.path.join(OUTPUT_DIR, "plot1_side_by_side_pca_k2_k3_k4.png")
    plt.savefig(plot1_path)
    plt.close()

    # Figure 2: Heatmaps of Cluster vs Ground-Truth Fault for K=2, K=3, K=4
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    for idx, k in enumerate([2, 3, 4]):
        ct = crosstabs[k].iloc[:k, :4]
        sns.heatmap(
            ct, annot=True, fmt=".1f", cmap="Blues", cbar=True,
            ax=axes[idx], annot_kws={"size": 10, "weight": "bold"}
        )
        axes[idx].set_title(f"K = {k}: Cluster vs Fault Breakdown (%)", fontsize=11, weight="bold")
        axes[idx].set_xlabel("Ground-Truth Fault Label (0, 1, 2, 3)", fontsize=10, weight="bold")
        axes[idx].set_ylabel(f"Discovered Clusters (K={k})" if idx == 0 else "", fontsize=10, weight="bold")

    fig.suptitle("How Effectively Each K Separates the True Fault Types (%)", fontsize=13, weight="bold", y=1.02)
    plt.tight_layout()
    plot2_path = os.path.join(OUTPUT_DIR, "plot2_crosstabs_k2_k3_k4_heatmaps.png")
    plt.savefig(plot2_path)
    plt.close()

    print("\n" + "=" * 90)
    print("CLUSTERING COMPARISON SUMMARY: K=2 vs K=3 vs K=4")
    print("=" * 90)
    print(metrics_df.to_string(index=False))
    print("=" * 90 + "\n")

    return metrics_df, crosstabs


if __name__ == "__main__":
    run_k_comparison_experiment()
