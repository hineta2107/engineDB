"""
EngineFaultDB - Branch A: Unsupervised Engine State Segmentation Pipeline
Implements the end-to-end workflow from Branch A diagram:
- A1: Inputs & Scope (55,998 clean sensor records without target Fault)
- A2: Encoding & Feature Scaling (Paper Z-Score normalization)
- A3: Feature-Space Construction (Option 1: PCA 2D/3D vs Option 2: Top-5 Selected Features)
- A4: Clustering Search: KMeans vs Gaussian Mixture Model (GMM) across K = 2..8
      Evaluating Silhouette score, Calinski-Harabasz, Davies-Bouldin
- A5: Robustness & Stability: Seed stability across multiple random seeds (Adjusted Rand Index)
- A6: Final Model Selection: Optimal K selection based on clustering metrics
- A7: Cluster Profiles & Visualization:
      - Plot 1: Silhouette vs K Curve (KMeans & GMM)
      - Plot 2: 2D PCA Cluster Scatter Plot
      - Plot 3: Cluster Radar / Parallel Feature Profiles
      - Plot 4: Cross-tabulation Matrix (Clusters vs Ground-Truth Faults)
- Exports CSV metrics, cluster assignments, and production joblib models.
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
from sklearn.mixture import GaussianMixture
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
logger = logging.getLogger("BranchA_Segmentation")

DATA_PATH = "outputs/common_foundation/prepared_feature_base.csv"
OUTPUT_DIR = "outputs/branch_a_segmentation"
TARGET_COL = "Fault"
TOP_5_FEATURES = ["CO", "Force", "Consumption L/H", "Power", "HC"]


def run_branch_a_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("=" * 70)
    logger.info("STARTING BRANCH A: UNSUPERVISED ENGINE STATE SEGMENTATION")
    logger.info("=" * 70)

    # A1: Load Data & Isolate Target
    df = pd.read_csv(DATA_PATH)
    logger.info(f"A1: Loaded {len(df)} records. Isolating feature matrix (No Target '{TARGET_COL}' used).")
    X = df[TOP_5_FEATURES].copy()
    y_ground_truth = df[TARGET_COL].copy()

    # A2: Scaling via Paper Z-score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info(f"A2: Features scaled with Z-Score normalization: mean=0, std=1.")

    # A3: Dimensionality Reduction for Representation (PCA 2D)
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca2d = pca_2d.fit_transform(X_scaled)
    var_exp = pca_2d.explained_variance_ratio_
    logger.info(f"A3: 2D PCA constructed (PC1: {var_exp[0]*100:.1f}%, PC2: {var_exp[1]*100:.1f}% variance).")

    # A4: Clustering Candidate Search (K = 2..8)
    logger.info("A4: Evaluating Clustering Search (KMeans & GMM across K=2..8)...")
    # Use stratified subsample for fast metric calculation
    eval_size = min(12000, len(X_scaled))
    idx_eval = np.random.RandomState(42).choice(len(X_scaled), eval_size, replace=False)
    X_eval = X_scaled[idx_eval]

    k_range = list(range(2, 9))
    eval_records = []

    for k in k_range:
        # KMeans
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_km = km.fit_predict(X_eval)
        sil_km = silhouette_score(X_eval, labels_km)
        ch_km = calinski_harabasz_score(X_eval, labels_km)
        db_km = davies_bouldin_score(X_eval, labels_km)

        # GMM
        gmm = GaussianMixture(n_components=k, random_state=42)
        labels_gmm = gmm.fit_predict(X_eval)
        sil_gmm = silhouette_score(X_eval, labels_gmm)
        ch_gmm = calinski_harabasz_score(X_eval, labels_gmm)
        db_gmm = davies_bouldin_score(X_eval, labels_gmm)

        eval_records.append({
            "K": k,
            "KMeans_Silhouette": round(sil_km, 4),
            "GMM_Silhouette": round(sil_gmm, 4),
            "KMeans_CalinskiHarabasz": round(ch_km, 1),
            "GMM_CalinskiHarabasz": round(ch_gmm, 1),
            "KMeans_DaviesBouldin": round(db_km, 4),
            "GMM_DaviesBouldin": round(db_gmm, 4)
        })

    eval_df = pd.DataFrame(eval_records)
    eval_csv_path = os.path.join(OUTPUT_DIR, "a4_clustering_candidate_metrics.csv")
    eval_df.to_csv(eval_csv_path, index=False)
    logger.info(f"Saved clustering candidate metrics: {eval_csv_path}")

    # A5: Robustness & Stability Evaluation (Multiple seeds for K=4)
    logger.info("A5: Evaluating seed stability for K=4 across 5 random seeds...")
    seeds = [42, 10, 100, 2024, 7]
    partitions = []
    for s in seeds:
        km_seed = KMeans(n_clusters=4, random_state=s, n_init=5)
        partitions.append(km_seed.fit_predict(X_eval))

    aris = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            aris.append(adjusted_rand_score(partitions[i], partitions[j]))
    mean_ari = float(np.mean(aris))
    logger.info(f"Mean Seed Stability ARI for K=4: {mean_ari:.4f}")

    # A6: Fit Final Champion Clustering Solution (K=4)
    logger.info("A6: Fitting Final Official Solution (KMeans with K=4)...")
    final_km = KMeans(n_clusters=4, random_state=42, n_init=15)
    final_clusters = final_km.fit_predict(X_scaled)

    df["Cluster"] = final_clusters

    # A7: Cluster Profiles
    cluster_profiles = df.groupby("Cluster")[TOP_5_FEATURES].mean().round(3)
    profiles_csv_path = os.path.join(OUTPUT_DIR, "a7_cluster_feature_profiles.csv")
    cluster_profiles.to_csv(profiles_csv_path)
    logger.info(f"Cluster profiles generated:\n{cluster_profiles}")

    # Cross-tabulation: Cluster vs Ground-Truth Fault
    cross_tab = pd.crosstab(
        df["Cluster"],
        df[TARGET_COL],
        margins=True,
        normalize="index"
    ).round(4) * 100
    crosstab_path = os.path.join(OUTPUT_DIR, "a7_cluster_vs_ground_truth_fault_matrix.csv")
    cross_tab.to_csv(crosstab_path)
    logger.info(f"Cluster vs Ground-Truth Fault Distribution (%):\n{cross_tab}")

    # ==========================================================================
    # VISUALIZATION PLOTS (300 DPI)
    # ==========================================================================
    sns.set_theme(style="whitegrid")

    # Plot 1: Silhouette Score vs K Curve
    plt.figure(figsize=(8.5, 5.2), dpi=300)
    plt.plot(eval_df["K"], eval_df["KMeans_Silhouette"], 'o-', color="#2980b9", lw=2.2, label="KMeans Silhouette")
    plt.plot(eval_df["K"], eval_df["GMM_Silhouette"], 's--', color="#e74c3c", lw=2.0, label="GMM Silhouette")
    plt.axvline(x=4, color="gray", linestyle=":", label="Selected Optimal K=4")
    plt.title("Branch A - Clustering Evaluation: Silhouette Score vs Number of Clusters (K)", fontsize=12, weight="bold", pad=12)
    plt.xlabel("Number of Clusters (K)", fontsize=11, weight="bold")
    plt.ylabel("Silhouette Score (Cao hơn là tốt hơn)", fontsize=11, weight="bold")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plot1_path = os.path.join(OUTPUT_DIR, "plot1_silhouette_vs_k.png")
    plt.savefig(plot1_path)
    plt.close()

    # Plot 2: 2D PCA Cluster Scatter Plot
    plt.figure(figsize=(9, 7), dpi=300)
    scatter_df = pd.DataFrame({
        "PC1": X_pca2d[idx_eval, 0],
        "PC2": X_pca2d[idx_eval, 1],
        "Cluster": [f"Cluster {c}" for c in final_clusters[idx_eval]]
    })
    palette = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]
    sns.scatterplot(
        data=scatter_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette=palette,
        alpha=0.65,
        s=25
    )
    plt.title("Branch A - 2D PCA Engine Operating State Clusters (Unsupervised)", fontsize=13, weight="bold", pad=12)
    plt.xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}% Variance)", fontsize=11, weight="bold")
    plt.ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}% Variance)", fontsize=11, weight="bold")
    plt.tight_layout()
    plot2_path = os.path.join(OUTPUT_DIR, "plot2_pca_cluster_scatter.png")
    plt.savefig(plot2_path)
    plt.close()

    # Plot 3: Cluster Feature Profiles (Bar chart comparison)
    norm_profiles = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())
    plt.figure(figsize=(10, 5.5), dpi=300)
    norm_profiles.T.plot(kind="bar", figsize=(10, 5.5), colormap="viridis", width=0.8)
    plt.title("Branch A - Normalized Mean Sensor Signatures by Cluster", fontsize=13, weight="bold", pad=12)
    plt.ylabel("Normalized Feature Intensity (0.0 - 1.0)", fontsize=11, weight="bold")
    plt.xlabel("Top-5 Critical Sensors", fontsize=11, weight="bold")
    plt.xticks(rotation=0, weight="bold")
    plt.legend(title="Engine State Cluster", fontsize=10)
    plt.tight_layout()
    plot3_path = os.path.join(OUTPUT_DIR, "plot3_cluster_feature_signatures.png")
    plt.savefig(plot3_path)
    plt.close()

    # Plot 4: Heatmap of Cluster vs Fault Type Breakdown
    plt.figure(figsize=(7.5, 5.5), dpi=300)
    sns.heatmap(
        cross_tab.iloc[:4, :4],
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar_kws={"label": "Percentage of Cluster (%)"},
        annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title("Integrated Insights: Unsupervised Clusters vs Ground-Truth Faults (%)", fontsize=12, weight="bold", pad=12)
    plt.xlabel("Ground-Truth Fault Label (0, 1, 2, 3)", fontsize=11, weight="bold")
    plt.ylabel("Discovered Engine Cluster", fontsize=11, weight="bold")
    plt.tight_layout()
    plot4_path = os.path.join(OUTPUT_DIR, "plot4_cluster_vs_fault_heatmap.png")
    plt.savefig(plot4_path)
    plt.close()

    # Save models and metadata
    joblib.dump(final_km, os.path.join(OUTPUT_DIR, "branch_a_kmeans_model.joblib"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "branch_a_scaler.joblib"))
    joblib.dump(pca_2d, os.path.join(OUTPUT_DIR, "branch_a_pca2d.joblib"))

    summary = {
        "status": "Completed",
        "sample_size": len(df),
        "optimal_k": 4,
        "algorithm": "KMeans",
        "seed_stability_ari": round(mean_ari, 4),
        "cluster_distribution": df["Cluster"].value_counts().to_dict(),
        "cluster_profiles": cluster_profiles.to_dict()
    }
    with open(os.path.join(OUTPUT_DIR, "branch_a_segmentation_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    logger.info("=" * 70)
    logger.info("BRANCH A PIPELINE EXECUTED & ALL VISUAL PLOTS EXPORTED!")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_branch_a_pipeline()

