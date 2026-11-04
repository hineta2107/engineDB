"""
EngineFaultDB - Experiment: Branch A Clustering with K=3 on Top-5 Features
Features used: ['CO', 'Force', 'Consumption L/H', 'Power', 'HC']
Steps:
1. Paper Z-score scaling on Top-5 features
2. KMeans with K=3 (Optimal cluster count)
3. Clustering evaluation metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin, Seed Stability ARI
4. Detailed cluster signatures and domain interpretation
5. Cross-tabulation against ground-truth Fault labels (0, 1, 2, 3)
6. High-Resolution Visualizations (300 DPI):
   - Plot 1: 2D PCA Cluster Distribution (Top-5 Features)
   - Plot 2: Cluster Mean Sensor Profiles Bar Chart
   - Plot 3: Cluster vs Ground-Truth Fault Heatmap
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
logger = logging.getLogger("BranchA_K3_Top5")

DATA_PATH = "outputs/common_foundation/prepared_feature_base.csv"
OUTPUT_DIR = "outputs/branch_a_k3_top5"
TARGET_COL = "Fault"
TOP_5_FEATURES = ["CO", "Force", "Consumption L/H", "Power", "HC"]


def run_branch_a_k3_top5():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("=" * 70)
    logger.info("RUNNING BRANCH A CLUSTERING (K=3) ON TOP-5 CRITICAL SENSORS")
    logger.info("=" * 70)

    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} records. Using Top-5 features: {TOP_5_FEATURES}")

    X = df[TOP_5_FEATURES]
    y_ground_truth = df[TARGET_COL]

    # 2. Z-Score Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 2D PCA for visual space
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    # 4. Metrics on stratified sample (12,000 samples)
    eval_size = min(12000, len(X_scaled))
    idx_eval = np.random.RandomState(42).choice(len(X_scaled), eval_size, replace=False)
    X_eval = X_scaled[idx_eval]

    # Evaluate KMeans K=3
    km_eval = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_eval = km_eval.fit_predict(X_eval)
    sil = silhouette_score(X_eval, labels_eval)
    ch = calinski_harabasz_score(X_eval, labels_eval)
    db = davies_bouldin_score(X_eval, labels_eval)

    # Seed stability across 5 runs
    seeds = [42, 10, 100, 2024, 7]
    seed_parts = [KMeans(n_clusters=3, random_state=s, n_init=5).fit_predict(X_eval) for s in seeds]
    aris = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            aris.append(adjusted_rand_score(seed_parts[i], seed_parts[j]))
    mean_ari = float(np.mean(aris))

    logger.info(f"Clustering Metrics for K=3 on Top-5:")
    logger.info(f"  - Silhouette Score: {sil:.4f}")
    logger.info(f"  - Davies-Bouldin Index: {db:.4f}")
    logger.info(f"  - Calinski-Harabasz: {ch:.1f}")
    logger.info(f"  - Seed Stability ARI: {mean_ari:.4f}")

    # 5. Fit full dataset with K=3
    km_full = KMeans(n_clusters=3, random_state=42, n_init=15)
    clusters = km_full.fit_predict(X_scaled)
    df["Cluster_K3"] = clusters

    # Cluster Profiles
    profiles = df.groupby("Cluster_K3")[TOP_5_FEATURES].mean().round(3)
    profiles.to_csv(os.path.join(OUTPUT_DIR, "k3_top5_cluster_feature_profiles.csv"))
    logger.info(f"Mean Sensor Profiles (K=3):\n{profiles}")

    # Cross-tabulation with Ground-Truth Fault
    crosstab = pd.crosstab(df["Cluster_K3"], y_ground_truth, margins=True, normalize="index").round(4) * 100
    crosstab.to_csv(os.path.join(OUTPUT_DIR, "k3_top5_cluster_vs_fault_matrix.csv"))
    logger.info(f"Cluster vs Fault Breakdown (%):\n{crosstab}")

    # ==========================================================================
    # VISUALIZATIONS GENERATION (300 DPI)
    # ==========================================================================
    sns.set_theme(style="whitegrid")

    # Plot 1: 2D PCA Cluster Scatter
    plt.figure(figsize=(9, 7), dpi=300)
    vis_idx = idx_eval[:6000]
    scatter_df = pd.DataFrame({
        "PC1": X_pca[vis_idx, 0],
        "PC2": X_pca[vis_idx, 1],
        "Cluster": [f"Cluster {c}" for c in clusters[vis_idx]]
    })
    sns.scatterplot(
        data=scatter_df, x="PC1", y="PC2", hue="Cluster",
        palette=["#3498db", "#2ecc71", "#e74c3c"],
        alpha=0.65, s=25
    )
    plt.title(f"Branch A: Engine State Clusters (K=3 on Top-5 Sensors)\nSilhouette: {sil:.4f} | Stability ARI: {mean_ari:.4f}", fontsize=13, weight="bold", pad=12)
    plt.xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}% Variance)", fontsize=11, weight="bold")
    plt.ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}% Variance)", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot1_k3_top5_pca_clusters.png"))
    plt.close()

    # Plot 2: Cluster Profiles Bar Chart
    norm_profiles = (profiles - profiles.min()) / (profiles.max() - profiles.min())
    plt.figure(figsize=(10, 5.5), dpi=300)
    norm_profiles.T.plot(kind="bar", figsize=(10, 5.5), colormap="Set1", width=0.75)
    plt.title("Cluster Sensor Profiles (Normalized Intensity across Top-5 Features)", fontsize=13, weight="bold", pad=12)
    plt.ylabel("Normalized Intensity (0.0 to 1.0)", fontsize=11, weight="bold")
    plt.xlabel("Top-5 Critical Sensors", fontsize=11, weight="bold")
    plt.xticks(rotation=0, weight="bold")
    plt.legend(title="Engine State (K=3)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot2_k3_top5_sensor_signatures.png"))
    plt.close()

    # Plot 3: Heatmap of Cluster vs Fault Type Breakdown
    plt.figure(figsize=(8, 5.5), dpi=300)
    sns.heatmap(
        crosstab.iloc[:3, :4], annot=True, fmt=".1f", cmap="Blues",
        cbar_kws={"label": "Percentage of Cluster (%)"},
        annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title("Integrated Insights: K=3 Clusters vs Ground-Truth Faults (%)", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Ground-Truth Fault Label (0, 1, 2, 3)", fontsize=11, weight="bold")
    plt.ylabel("Discovered Engine Cluster (K=3)", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot3_k3_top5_cluster_vs_fault_heatmap.png"))
    plt.close()

    # Save models
    joblib.dump(km_full, os.path.join(OUTPUT_DIR, "k3_top5_kmeans_model.joblib"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "k3_top5_scaler.joblib"))

    summary = {
        "clusters_k": 3,
        "features": TOP_5_FEATURES,
        "silhouette_score": round(float(sil), 4),
        "davies_bouldin_score": round(float(db), 4),
        "calinski_harabasz_score": round(float(ch), 1),
        "seed_stability_ari": round(float(mean_ari), 4),
        "cluster_distribution": df["Cluster_K3"].value_counts().to_dict(),
        "cluster_mean_profiles": profiles.to_dict()
    }
    with open(os.path.join(OUTPUT_DIR, "k3_top5_clustering_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    logger.info("=" * 70)
    logger.info("BRANCH A (K=3 ON TOP-5) COMPLETED & ALL ARTIFACTS EXPORTED!")
    logger.info("=" * 70)
    return summary


if __name__ == "__main__":
    run_branch_a_k3_top5()
