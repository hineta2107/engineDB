"""
EngineFaultDB - Step: Full Correlation Matrix (14 Features + Target 'Fault')
Computes both Pearson (linear) and Spearman (rank-based/monotonic) correlation matrices.
Generates high-resolution annotated Heatmap plots and exports numerical CSVs.
"""

import os
import sys
import logging

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("CorrelationMatrix")

DATA_PATH = "outputs/common_foundation/prepared_feature_base.csv"
OUTPUT_DIR = "outputs/correlation_analysis"


def run_correlation_matrix_analysis():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("=" * 65)
    logger.info("COMPUTING FULL CORRELATION MATRIX (14 FEATURES + TARGET 'FAULT')")
    logger.info("=" * 65)

    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Reorder columns to put Fault at the end or at the front
    cols = [c for c in df.columns if c != "Fault"] + ["Fault"]
    df = df[cols]

    # 1. Pearson Correlation
    pearson_corr = df.corr(method="pearson").round(3)
    pearson_path = os.path.join(OUTPUT_DIR, "correlation_matrix_pearson.csv")
    pearson_corr.to_csv(pearson_path)
    logger.info(f"Saved Pearson correlation table: {pearson_path}")

    # 2. Spearman Rank Correlation (Better for non-linear monotonic relationships with discrete Fault)
    spearman_corr = df.corr(method="spearman").round(3)
    spearman_path = os.path.join(OUTPUT_DIR, "correlation_matrix_spearman.csv")
    spearman_corr.to_csv(spearman_path)
    logger.info(f"Saved Spearman correlation table: {spearman_path}")

    # 3. Plot High-Resolution Heatmap (Pearson)
    plt.figure(figsize=(12, 10), dpi=300)
    mask = np.triu(np.ones_like(pearson_corr, dtype=bool))  # Upper triangle mask
    
    sns.heatmap(
        pearson_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"},
        annot_kws={"size": 9, "weight": "bold"}
    )
    plt.title("EngineFaultDB - Full Correlation Matrix Heatmap\n(14 Features + Target 'Fault')", fontsize=14, weight="bold", pad=15)
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "correlation_matrix_heatmap.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved correlation heatmap image: {plot_path}")

    # 4. Extract correlation specifically with Target 'Fault'
    target_corr = pd.DataFrame({
        "Pearson_r": pearson_corr["Fault"],
        "Spearman_rho": spearman_corr["Fault"]
    }).drop("Fault").sort_values("Pearson_r", ascending=False)

    target_corr_path = os.path.join(OUTPUT_DIR, "features_correlation_with_target.csv")
    target_corr.to_csv(target_corr_path)

    print("\n" + "=" * 70)
    print("CORRELATION OF ALL 14 FEATURES WITH TARGET 'FAULT':")
    print("=" * 70)
    print(target_corr.to_string())
    print("=" * 70 + "\n")

    return pearson_corr, target_corr


if __name__ == "__main__":
    run_correlation_matrix_analysis()
