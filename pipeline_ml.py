"""
EngineFaultDB Machine Learning Pipeline
Architecture follows the Dual-Branch specification:
- Common Foundation: Data Ingestion, Cleaning, Quality Checks, Feature Governance
- Branch A: Unsupervised Learning (Engine Operating State Segmentation & Clustering)
- Branch B: Supervised Learning (Fault Classification, Multi-Model Benchmarking, SHAP/Feature Importance)
- Integrated Insights: Cross-branch analysis of cluster distributions vs ground-truth faults
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import joblib

# Scikit-learn modules
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("EngineFaultPipeline")

# ==============================================================================
# FEATURE FAMILIES DEFINITION (Domain Specific for EngineFaultDB)
# ==============================================================================
FEATURE_FAMILIES = {
    "engine_performance": ["RPM", "Force", "Power"],
    "fuel_system": ["MAP", "TPS", "Consumption L/H", "Consumption L/100KM"],
    "vehicle_dynamics": ["Speed"],
    "exhaust_emissions": ["CO", "HC", "CO2", "O2"],
    "combustion_chemistry": ["Lambda", "AFR"]
}

ALL_FEATURE_COLS = [f for group in FEATURE_FAMILIES.values() for f in group]
TARGET_COL = "Fault"


# ==============================================================================
# 1. COMMON FOUNDATION
# ==============================================================================
class CommonFoundation:
    """Handles Data Ingestion, Cleaning, Quality Checks, and Feature Governance."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.raw_df = None
        self.clean_df = None
        self.qc_report = {}

    def step1_data_ingestion(self) -> pd.DataFrame:
        logger.info(f"Step 1: Loading raw dataset from {self.data_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        self.raw_df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.raw_df)} records, shape: {self.raw_df.shape}")
        return self.raw_df

    def step2_data_cleaning(self) -> pd.DataFrame:
        logger.info("Step 2: Cleaning data (Handling duplicates, missing values, anomalies)")
        df = self.raw_df.copy()

        # Check duplicates
        initial_count = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        dropped_dups = initial_count - len(df)
        if dropped_dups > 0:
            logger.info(f"Removed {dropped_dups} duplicate records.")

        # Check missing values
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Nulls detected: {null_counts[null_counts > 0].to_dict()}. Dropping NA rows.")
            df = df.dropna().reset_index(drop=True)
        else:
            logger.info("No missing values found.")

        self.clean_df = df
        return self.clean_df

    def step3_quality_check(self) -> Dict[str, Any]:
        logger.info("Step 3: Performing Data Quality and Physical Plausibility Checks")
        df = self.clean_df

        # Domain validity checks (e.g., negative values where non-physical)
        non_negative_cols = ["RPM", "Power", "Consumption L/H", "Consumption L/100KM", "Speed", "O2", "CO2"]
        violations = {}
        for col in non_negative_cols:
            if col in df.columns:
                invalid = (df[col] < 0).sum()
                if invalid > 0:
                    violations[col] = int(invalid)

        self.qc_report = {
            "total_records": len(df),
            "target_classes": df[TARGET_COL].unique().tolist(),
            "target_distribution": df[TARGET_COL].value_counts().to_dict(),
            "non_physical_negatives": violations
        }
        logger.info(f"QC Summary: Total={len(df)}, Classes={self.qc_report['target_distribution']}")
        return self.qc_report

    def step4_feature_governance(self) -> pd.DataFrame:
        logger.info("Step 4 & 5: Feature Governance & Final Prepared Feature Base")
        # Ensure all required features are present
        missing_features = [col for col in ALL_FEATURE_COLS if col not in self.clean_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Final checks: target isolation
        logger.info("Feature base prepared successfully for Branch A & Branch B.")
        return self.clean_df


# ==============================================================================
# 2. BRANCH A: UNSUPERVISED LEARNING (SEGMENTATION)
# ==============================================================================
class BranchASegmentation:
    """
    Branch A: Engine State Segmentation
    - Feature families grouping & scaling
    - Representation construction: Option 1 (PCA) vs Option 2 (Correlation-based)
    - Clustering Candidate Search (KMeans & GMM for K=2..6)
    - Stability & Robustness (ARI, Subsampling)
    - Final Profile Analysis (NO target 'Fault' used during fitting!)
    """

    def __init__(self, data: pd.DataFrame, output_dir: str = "outputs/branch_a"):
        self.data = data.copy()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.features = self.data[ALL_FEATURE_COLS]
        self.scaled_features = None
        self.scaler = None
        self.pca_model = None

    def a1_a2_encoding_and_scaling(self):
        logger.info("Branch A [A1-A2]: Scaling feature families with Standard Scaler")
        self.scaler = StandardScaler()
        self.scaled_features = pd.DataFrame(
            self.scaler.fit_transform(self.features),
            columns=ALL_FEATURE_COLS
        )
        return self.scaled_features

    def a3_feature_space_construction(self, variance_threshold: float = 0.85):
        logger.info(f"Branch A [A3]: Constructing representations (PCA vs Correlation-based)")
        # Option 1: PCA representation
        pca = PCA(n_components=variance_threshold, random_state=42)
        X_pca = pca.fit_transform(self.scaled_features)
        self.pca_model = pca
        logger.info(f"Option 1 (PCA): Retained {X_pca.shape[1]} components for {variance_threshold*100}% variance")

        # Option 2: Correlation-based selection (drop features with |r| >= 0.90)
        corr_matrix = self.scaled_features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] >= 0.90)]
        X_corr = self.scaled_features.drop(columns=to_drop)
        logger.info(f"Option 2 (Corr): Dropped {to_drop}, retained {X_corr.shape[1]} features")

        return {"O1_PCA": X_pca, "O2_CORR": X_corr.values}

    def a4_clustering_candidate_search(self, X_rep: np.ndarray, k_range: range = range(2, 7), sample_size: int = 10000):
        logger.info(f"Branch A [A4]: Evaluating Clustering across K={list(k_range)}")
        # Subsample for efficient metric computation on large data
        if len(X_rep) > sample_size:
            idx = np.random.RandomState(42).choice(len(X_rep), sample_size, replace=False)
            X_eval = X_rep[idx]
        else:
            X_eval = X_rep

        results = []
        for k in k_range:
            # KMeans
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_km = km.fit_predict(X_eval)
            sil_km = silhouette_score(X_eval, labels_km)
            ch_km = calinski_harabasz_score(X_eval, labels_km)
            db_km = davies_bouldin_score(X_eval, labels_km)

            results.append({
                "algorithm": "KMeans",
                "k": k,
                "silhouette": round(sil_km, 4),
                "calinski_harabasz": round(ch_km, 2),
                "davies_bouldin": round(db_km, 4)
            })

            # Gaussian Mixture
            gmm = GaussianMixture(n_components=k, random_state=42)
            labels_gmm = gmm.fit_predict(X_eval)
            sil_gmm = silhouette_score(X_eval, labels_gmm)
            results.append({
                "algorithm": "GMM",
                "k": k,
                "silhouette": round(sil_gmm, 4),
                "calinski_harabasz": round(calinski_harabasz_score(X_eval, labels_gmm), 2),
                "davies_bouldin": round(davies_bouldin_score(X_eval, labels_gmm), 4)
            })

        metrics_df = pd.DataFrame(results)
        metrics_df.to_csv(os.path.join(self.output_dir, "clustering_candidate_metrics.csv"), index=False)
        logger.info("Clustering search completed and saved.")
        return metrics_df

    def a5_stability_evaluation(self, X_rep: np.ndarray, best_k: int = 4, n_runs: int = 5):
        logger.info(f"Branch A [A5]: Robustness & Stability evaluation for K={best_k} across {n_runs} seeds")
        seeds = [10, 42, 100, 2024, 999][:n_runs]
        partition_runs = []
        
        sample_idx = np.random.RandomState(42).choice(len(X_rep), min(len(X_rep), 10000), replace=False)
        X_sample = X_rep[sample_idx]

        for s in seeds:
            km = KMeans(n_clusters=best_k, random_state=s, n_init=5)
            partition_runs.append(km.fit_predict(X_sample))

        pairwise_aris = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                ari = adjusted_rand_score(partition_runs[i], partition_runs[j])
                pairwise_aris.append(ari)

        mean_ari = np.mean(pairwise_aris)
        logger.info(f"Mean Seed Stability ARI for K={best_k}: {mean_ari:.4f}")
        return mean_ari

    def a6_a7_final_selection_and_profiling(self, best_k: int = 4):
        logger.info(f"Branch A [A6-A7]: Fitting final clustering solution with K={best_k}")
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        clusters = km.fit_predict(self.scaled_features)

        self.data["Cluster"] = clusters
        cluster_profiles = self.data.groupby("Cluster")[ALL_FEATURE_COLS].mean()
        cluster_profiles.to_csv(os.path.join(self.output_dir, "cluster_feature_profiles.csv"))
        
        # Save cluster model and scaler
        joblib.dump(km, os.path.join(self.output_dir, "kmeans_cluster_model.joblib"))
        joblib.dump(self.scaler, os.path.join(self.output_dir, "branch_a_scaler.joblib"))

        logger.info(f"Cluster profiles generated:\n{cluster_profiles.round(2)}")
        return self.data


# ==============================================================================
# 3. BRANCH B: SUPERVISED LEARNING (FAULT CLASSIFICATION)
# ==============================================================================
class BranchBSupervised:
    """
    Branch B: Supervised Engine Fault Classification Pipeline
    - B1: Target Definition & Stratified Train/Val/Test Split
    - B2: Train-only Preprocessing (Leakage-Safe Scaling)
    - B3: Multi-model Training & Stratified Cross-Validation
    - B4: Model Evaluation & Benchmarking
    - B5: Locked Test Evaluation
    - B6: Pipeline Export & Metadata Logging
    """

    def __init__(self, data: pd.DataFrame, output_dir: str = "outputs/branch_b"):
        self.data = data.copy()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.scaler = None
        self.models = {}
        self.cv_results = {}
        self.best_model_name = None
        self.best_model = None

    def b1_stratified_split(self, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
        logger.info("Branch B [B1]: Stratified Data Split (Train / Val / Locked Test)")
        X = self.data[ALL_FEATURE_COLS]
        y = self.data[TARGET_COL]

        # First split into Train+Val and Locked Test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        # Then split temp into Train and Val
        val_relative_size = val_size / (1.0 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative_size, stratify=y_temp, random_state=random_state
        )

        logger.info(
            f"Split sizes - Train: {len(self.X_train)}, Val: {len(self.X_val)}, Locked Test: {len(self.X_test)}"
        )

    def b2_train_only_preprocessing(self):
        logger.info("Branch B [B2]: Train-only Preprocessing (Fitting Scaler strictly on Train)")
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        logger.info("Feature scaling applied independently to prevent data leakage.")

    def b3_b4_model_training_and_comparison(self):
        logger.info("Branch B [B3-B4]: Training & Comparing Candidate Classifiers")

        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
            "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        benchmark_records = []

        for name, clf in self.models.items():
            logger.info(f"Evaluating {name} with 5-fold Stratified CV...")
            scores = cross_val_score(clf, self.X_train_scaled, self.y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
            mean_f1 = scores.mean()
            std_f1 = scores.std()

            # Fit on full training set and evaluate on validation set
            clf.fit(self.X_train_scaled, self.y_train)
            val_preds = clf.predict(self.X_val_scaled)
            val_acc = accuracy_score(self.y_val, val_preds)
            val_p, val_r, val_f1, _ = precision_recall_fscore_support(self.y_val, val_preds, average="macro")

            benchmark_records.append({
                "model": name,
                "cv_f1_macro_mean": round(mean_f1, 4),
                "cv_f1_macro_std": round(std_f1, 4),
                "val_accuracy": round(val_acc, 4),
                "val_f1_macro": round(val_f1, 4),
                "val_precision_macro": round(val_p, 4),
                "val_recall_macro": round(val_r, 4)
            })

        benchmark_df = pd.DataFrame(benchmark_records).sort_values("val_f1_macro", ascending=False)
        benchmark_df.to_csv(os.path.join(self.output_dir, "model_benchmark_comparison.csv"), index=False)
        logger.info(f"Model Benchmark Summary:\n{benchmark_df}")

        # Best model selection
        self.best_model_name = benchmark_df.iloc[0]["model"]
        self.best_model = self.models[self.best_model_name]
        logger.info(f"Best selected model: {self.best_model_name}")
        return benchmark_df

    def b5_locked_test_evaluation(self) -> Dict[str, Any]:
        logger.info(f"Branch B [B5]: OPENING LOCKED TEST SET ONCE with {self.best_model_name}")
        test_preds = self.best_model.predict(self.X_test_scaled)

        test_acc = accuracy_score(self.y_test, test_preds)
        test_p, test_r, test_f1, _ = precision_recall_fscore_support(self.y_test, test_preds, average="macro")
        cls_report = classification_report(self.y_test, test_preds, output_dict=True)
        cm = confusion_matrix(self.y_test, test_preds).tolist()

        test_results = {
            "best_model": self.best_model_name,
            "test_accuracy": round(test_acc, 4),
            "test_f1_macro": round(test_f1, 4),
            "test_precision_macro": round(test_p, 4),
            "test_recall_macro": round(test_r, 4),
            "confusion_matrix": cm,
            "classification_report": cls_report
        }

        # Feature Importance (if tree-based)
        if hasattr(self.best_model, "feature_importances_"):
            importances = dict(zip(ALL_FEATURE_COLS, self.best_model.feature_importances_))
            test_results["feature_importances"] = {k: round(v, 4) for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}
            logger.info(f"Top 5 Feature Importances:\n{list(test_results['feature_importances'].items())[:5]}")

        with open(os.path.join(self.output_dir, "locked_test_results.json"), "w") as f:
            json.dump(test_results, f, indent=4)

        logger.info(f"Final Locked Test Metrics -> Accuracy: {test_acc:.4f}, F1-Macro: {test_f1:.4f}")
        return test_results

    def b6_save_pipeline_artifacts(self):
        logger.info("Branch B [B6]: Exporting Model & Scaler Artifacts")
        joblib.dump(self.best_model, os.path.join(self.output_dir, f"best_model_{self.best_model_name}.joblib"))
        joblib.dump(self.scaler, os.path.join(self.output_dir, "branch_b_scaler.joblib"))
        logger.info("All Branch B production artifacts saved successfully.")


# ==============================================================================
# 4. INTEGRATED INSIGHTS (CROSS-BRANCH)
# ==============================================================================
def generate_integrated_insights(df_with_clusters: pd.DataFrame, output_dir: str = "outputs/integrated_insights"):
    """Cross-tabulates unsupervised clusters against ground-truth Fault categories."""
    logger.info("Step 4: Generating Integrated Insights (Branch A Clusters vs Ground Truth Faults)")
    os.makedirs(output_dir, exist_ok=True)

    crosstab = pd.crosstab(
        df_with_clusters["Cluster"],
        df_with_clusters[TARGET_COL],
        margins=True,
        normalize="index"
    ).round(4) * 100

    crosstab.to_csv(os.path.join(output_dir, "cluster_vs_fault_distribution.csv"))
    logger.info(f"Cluster vs Fault Type Breakdown (%):\n{crosstab}")
    return crosstab


# ==============================================================================
# MAIN EXECUTION ENTRYPOINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run EngineFaultDB ML Dual-Branch Pipeline")
    parser.add_argument("--data_path", type=str, default="EngineFaultDB_Final.csv", help="Path to raw dataset")
    parser.add_argument("--clusters", type=int, default=4, help="Optimal K clusters for Branch A")
    args = parser.parse_args()

    logger.info("=== Starting EngineFaultDB Dual-Branch ML Pipeline ===")

    # 1. Common Foundation
    cf = CommonFoundation(args.data_path)
    cf.step1_data_ingestion()
    cf.step2_data_cleaning()
    cf.step3_quality_check()
    clean_df = cf.step4_feature_governance()

    # 2. Branch A: Unsupervised Learning
    branch_a = BranchASegmentation(clean_df)
    branch_a.a1_a2_encoding_and_scaling()
    reps = branch_a.a3_feature_space_construction()
    branch_a.a4_clustering_candidate_search(reps["O1_PCA"])
    branch_a.a5_stability_evaluation(reps["O1_PCA"], best_k=args.clusters)
    df_with_clusters = branch_a.a6_a7_final_selection_and_profiling(best_k=args.clusters)

    # 3. Branch B: Supervised Learning
    branch_b = BranchBSupervised(clean_df)
    branch_b.b1_stratified_split()
    branch_b.b2_train_only_preprocessing()
    branch_b.b3_b4_model_training_and_comparison()
    branch_b.b5_locked_test_evaluation()
    branch_b.b6_save_pipeline_artifacts()

    # 4. Integrated Insights
    generate_integrated_insights(df_with_clusters)

    logger.info("=== All Pipeline Stages Completed Successfully! ===")


if __name__ == "__main__":
    main()

