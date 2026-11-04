"""
EngineFaultDB - Step: Feature Importance & Explainability Analysis
Evaluates feature importance using multiple complementary approaches:
1. Tree-based MDI (Mean Decrease in Impurity) via Random Forest
2. Permutation Importance (Model-agnostic on held-out Validation set)
3. Mutual Information (Non-linear relationship with Target 'Fault')
4. Random Forest Feature Importance for predicting individual Fault classes
Exports comprehensive metrics and rankings to CSV and JSON.
"""

import os
import sys
import json
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FeatureImportance")

DATA_PATH = "outputs/common_foundation/prepared_feature_base_normalized.csv"
OUTPUT_DIR = "outputs/feature_importance"
TARGET_COL = "Fault"


def run_feature_importance_analysis(data_path: str = DATA_PATH, output_dir: str = OUTPUT_DIR):
    logger.info("=" * 65)
    logger.info("RUNNING MULTI-METHOD FEATURE IMPORTANCE & EXPLAINABILITY")
    logger.info("=" * 65)

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c != TARGET_COL]

    X = df[feature_cols]
    y = df[TARGET_COL]

    # Split train/val for permutation importance evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    logger.info(f"Dataset split: Train={len(X_train)}, Val={len(X_val)}")

    # 1. Mutual Information (Information Gain)
    logger.info("1/3 Calculating Mutual Information (non-linear dependency)...")
    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
    mi_series = pd.Series(mi_scores, index=feature_cols, name="Mutual_Information").round(4)

    # 2. Random Forest MDI (Gini Importance)
    logger.info("2/3 Fitting Random Forest Classifier and computing MDI Gini Importance...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_mdi_series = pd.Series(rf.feature_importances_, index=feature_cols, name="RandomForest_MDI").round(4)

    # 3. Permutation Feature Importance on Validation Set
    logger.info("3/3 Computing Permutation Importance on held-out validation set...")
    perm_res = permutation_importance(
        rf, X_val, y_val, n_repeats=5, random_state=42, scoring="f1_macro", n_jobs=-1
    )
    perm_mean_series = pd.Series(perm_res.importances_mean, index=feature_cols, name="Permutation_F1_Drop").round(4)
    perm_std_series = pd.Series(perm_res.importances_std, index=feature_cols, name="Permutation_Std").round(4)

    # Aggregate into comparison DataFrame
    importance_df = pd.concat([rf_mdi_series, perm_mean_series, perm_std_series, mi_series], axis=1)

    # Calculate overall rank (average rank across methods)
    importance_df["Rank_MDI"] = importance_df["RandomForest_MDI"].rank(ascending=False)
    importance_df["Rank_Perm"] = importance_df["Permutation_F1_Drop"].rank(ascending=False)
    importance_df["Rank_MI"] = importance_df["Mutual_Information"].rank(ascending=False)
    importance_df["Consensus_Rank"] = (
        (importance_df["Rank_MDI"] + importance_df["Rank_Perm"] + importance_df["Rank_MI"]) / 3.0
    ).round(2)

    importance_df = importance_df.sort_values("Consensus_Rank")

    # Save to CSV
    csv_path = os.path.join(output_dir, "feature_importance_rankings.csv")
    importance_df.to_csv(csv_path)
    logger.info(f"Feature importance table saved to: {csv_path}")

    # Save summary JSON
    top_features = importance_df.index.tolist()
    summary = {
        "top_5_critical_sensors": top_features[:5],
        "full_rankings": importance_df.to_dict(orient="index")
    }
    json_path = os.path.join(output_dir, "feature_importance_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Summary JSON saved to: {json_path}")

    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE & RANKING SUMMARY TABLE:")
    print("=" * 80)
    print(importance_df[["RandomForest_MDI", "Permutation_F1_Drop", "Mutual_Information", "Consensus_Rank"]].to_string())
    print("=" * 80 + "\n")

    return importance_df


if __name__ == "__main__":
    run_feature_importance_analysis()
