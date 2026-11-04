"""
EngineFaultDB - Step 1: Common Foundation Pipeline
Implements the 5 standard steps:
1. Data Ingestion: Load raw CSV and generate detailed initial profile
2. Data Cleaning: Detect/remove duplicates, handle missing values, inspect anomalies
3. Data Quality Check: Range validations, physical plausibility, class balance checks
4. Feature Governance: Check multicollinearity, target isolation, assign feature families
5. Prepared Feature Base: Export clean feature matrix ready for Branch A & B
"""

import os
import sys
import json
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("CommonFoundation")

# Domain-specific feature families for automotive engine diagnostics
FEATURE_FAMILIES = {
    "engine_performance": ["RPM", "Force", "Power"],
    "fuel_system": ["MAP", "TPS", "Consumption L/H", "Consumption L/100KM"],
    "vehicle_dynamics": ["Speed"],
    "exhaust_emissions": ["CO", "HC", "CO2", "O2"],
    "combustion_chemistry": ["Lambda", "AFR"]
}

TARGET_COLUMN = "Fault"
ALL_FEATURES = [feat for group in FEATURE_FAMILIES.values() for feat in group]


class CommonFoundationPipeline:
    def __init__(self, raw_data_path: str = "EngineFaultDB_Final.csv", output_dir: str = "outputs/common_foundation"):
        self.raw_data_path = raw_data_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.raw_df = None
        self.clean_df = None
        self.profile_report: Dict[str, Any] = {}

    # --------------------------------------------------------------------------
    # 1. DATA INGESTION & DATA PROFILING
    # --------------------------------------------------------------------------
    def run_step1_data_ingestion_and_profile(self) -> pd.DataFrame:
        logger.info("=" * 60)
        logger.info("STEP 1: DATA INGESTION & DETAILED PROFILING")
        logger.info("=" * 60)

        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"Source dataset not found at: {self.raw_data_path}")

        self.raw_df = pd.read_csv(self.raw_data_path)
        logger.info(f"Loaded raw dataset successfully: {self.raw_df.shape[0]} rows, {self.raw_df.shape[1]} columns.")

        # Compute summary statistics
        desc = self.raw_df.describe().T
        desc["missing_count"] = self.raw_df.isnull().sum()
        desc["missing_pct"] = (self.raw_df.isnull().mean() * 100).round(2)
        desc["skewness"] = self.raw_df.skew().round(3)
        desc["kurtosis"] = self.raw_df.kurt().round(3)

        # Save profile CSV
        profile_path = os.path.join(self.output_dir, "step1_initial_data_profile.csv")
        desc.to_csv(profile_path)
        logger.info(f"Detailed statistical profile saved to: {profile_path}")

        self.profile_report["raw_shape"] = list(self.raw_df.shape)
        self.profile_report["columns"] = list(self.raw_df.columns)
        return self.raw_df

    # --------------------------------------------------------------------------
    # 2. DATA CLEANING
    # --------------------------------------------------------------------------
    def run_step2_data_cleaning(self) -> pd.DataFrame:
        logger.info("=" * 60)
        logger.info("STEP 2: DATA CLEANING (DUPLICATES & MISSING VALUES)")
        logger.info("=" * 60)

        df = self.raw_df.copy()
        initial_len = len(df)

        # Check and handle duplicates
        duplicates_count = df.duplicated().sum()
        logger.info(f"Detected duplicate rows: {duplicates_count}")
        if duplicates_count > 0:
            df = df.drop_duplicates().reset_index(drop=True)
            logger.info(f"Deduplicated data: {initial_len} -> {len(df)} records.")

        # Check and handle missing values
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls > 0:
            logger.warning(f"Found missing values:\n{null_counts[null_counts > 0]}")
            df = df.dropna().reset_index(drop=True)
            logger.info(f"Dropped null rows. New row count: {len(df)}")
        else:
            logger.info("Zero missing/null values found across all columns.")

        self.clean_df = df
        self.profile_report["cleaning_stats"] = {
            "initial_rows": initial_len,
            "duplicates_removed": int(duplicates_count),
            "final_clean_rows": len(df)
        }
        return self.clean_df

    # --------------------------------------------------------------------------
    # 3. DATA QUALITY CHECK & PHYSICAL CONSTRAINTS
    # --------------------------------------------------------------------------
    def run_step3_data_quality_check(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("STEP 3: DATA QUALITY & PHYSICAL FEASIBILITY CHECKS")
        logger.info("=" * 60)

        df = self.clean_df

        # Physical domain checks for automotive combustion engine
        quality_issues = {}
        
        # Non-negative sensor checks
        strictly_non_negative = ["RPM", "Power", "Consumption L/H", "Consumption L/100KM", "Speed", "CO", "HC", "CO2", "O2"]
        for col in strictly_non_negative:
            if col in df.columns:
                invalid_count = int((df[col] < 0).sum())
                if invalid_count > 0:
                    quality_issues[f"{col}_negative_count"] = invalid_count

        # Check Lambda and AFR logical balance
        # Lambda = 1.0 corresponds to stoichiometric AFR (~14.7 for gasoline)
        lambda_afr_corr = float(df["Lambda"].corr(df["AFR"]))
        logger.info(f"Pearson Correlation between Lambda and AFR: {lambda_afr_corr:.4f}")

        # Target class balance
        target_counts = df[TARGET_COLUMN].value_counts().to_dict()
        target_proportions = (df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2).to_dict()

        logger.info(f"Target Distribution: {target_counts}")
        logger.info(f"Target Proportions (%): {target_proportions}")

        qc_summary = {
            "physical_constraint_violations": quality_issues if quality_issues else "None (All physically valid)",
            "lambda_afr_consistency_correlation": round(lambda_afr_corr, 4),
            "target_class_distribution": target_counts,
            "target_class_proportions_pct": target_proportions
        }
        
        self.profile_report["quality_checks"] = qc_summary
        
        qc_path = os.path.join(self.output_dir, "step3_data_quality_report.json")
        with open(qc_path, "w") as f:
            json.dump(qc_summary, f, indent=4)
        logger.info(f"Data quality report saved to: {qc_path}")
        return qc_summary

    # --------------------------------------------------------------------------
    # 4. FEATURE GOVERNANCE & MULTICOLLINEARITY ANALYSIS
    # --------------------------------------------------------------------------
    def run_step4_feature_governance(self) -> pd.DataFrame:
        logger.info("=" * 60)
        logger.info("STEP 4: FEATURE GOVERNANCE & MULTICOLLINEARITY")
        logger.info("=" * 60)

        # Check for presence of all required feature groups
        for family, cols in FEATURE_FAMILIES.items():
            missing = [c for c in cols if c not in self.clean_df.columns]
            if missing:
                raise ValueError(f"Feature family '{family}' is missing fields: {missing}")
            logger.info(f"Family '{family}': verified {len(cols)} features -> {cols}")

        # Compute full correlation matrix among features
        features_df = self.clean_df[ALL_FEATURES]
        corr_matrix = features_df.corr().round(4)
        corr_path = os.path.join(self.output_dir, "step4_feature_correlation_matrix.csv")
        corr_matrix.to_csv(corr_path)
        logger.info(f"Feature correlation matrix saved to: {corr_path}")

        # Detect high multicollinearity pairs (|r| >= 0.85)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                val = corr_matrix.iloc[i, j]
                if abs(val) >= 0.85:
                    high_corr_pairs.append({
                        "feature_1": col_i,
                        "feature_2": col_j,
                        "correlation": float(val)
                    })

        logger.info(f"High collinearity pairs (|r| >= 0.85): {len(high_corr_pairs)}")
        for pair in high_corr_pairs:
            logger.info(f"  - {pair['feature_1']} <-> {pair['feature_2']}: {pair['correlation']:.4f}")

        self.profile_report["high_correlation_pairs"] = high_corr_pairs
        return corr_matrix

    # --------------------------------------------------------------------------
    # 5. PREPARED FEATURE BASE EXPORT
    # --------------------------------------------------------------------------
    def run_step5_prepared_feature_base(self) -> str:
        logger.info("=" * 60)
        logger.info("STEP 5: EXPORT PREPARED FEATURE BASE")
        logger.info("=" * 60)

        clean_data_path = os.path.join(self.output_dir, "prepared_feature_base.csv")
        self.clean_df.to_csv(clean_data_path, index=False)
        logger.info(f"Saved Prepared Feature Base ({len(self.clean_df)} records) to: {clean_data_path}")

        # Save comprehensive pipeline execution metadata
        meta_path = os.path.join(self.output_dir, "common_foundation_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(self.profile_report, f, indent=4)
        logger.info(f"Full execution summary saved to: {meta_path}")
        return clean_data_path

    # --------------------------------------------------------------------------
    # RUN ALL 5 STEPS SEQUENTIALLY
    # --------------------------------------------------------------------------
    def run_full_common_foundation(self):
        self.run_step1_data_ingestion_and_profile()
        self.run_step2_data_cleaning()
        self.run_step3_data_quality_check()
        self.run_step4_feature_governance()
        self.run_step5_prepared_feature_base()
        logger.info(">>> COMMON FOUNDATION PIPELINE FINISHED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    pipeline = CommonFoundationPipeline(raw_data_path="EngineFaultDB_Final.csv")
    pipeline.run_full_common_foundation()
