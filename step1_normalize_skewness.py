"""
EngineFaultDB - Step 1.1: Data Normalization & Skewness Correction
Applies Yeo-Johnson Power Transformation to pull heavily skewed & high-kurtosis
sensor features back into near-normal Gaussian distributions.
Generates a side-by-side verification report (Before vs After).
"""

import os
import sys
import json
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import PowerTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DataNormalization")

INPUT_DATA_PATH = "outputs/common_foundation/prepared_feature_base.csv"
OUTPUT_DIR = "outputs/common_foundation"
TARGET_COL = "Fault"


def run_skewness_correction_and_verification(input_path: str = INPUT_DATA_PATH, output_dir: str = OUTPUT_DIR):
    logger.info("=" * 65)
    logger.info("APPLYING YEO-JOHNSON POWER TRANSFORMATION TO NORMALIZE FEATURES")
    logger.info("=" * 65)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Prepared feature base not found: {input_path}")

    df = pd.read_csv(input_path)
    feature_cols = [c for c in df.columns if c != TARGET_COL]

    # Initial stats (Before)
    skew_before = df[feature_cols].skew()
    kurt_before = df[feature_cols].kurt()

    # Fit Yeo-Johnson PowerTransformer
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    transformed_features = pt.fit_transform(df[feature_cols])

    df_transformed = pd.DataFrame(transformed_features, columns=feature_cols)
    df_transformed[TARGET_COL] = df[TARGET_COL].values

    # Stats after transformation
    skew_after = df_transformed[feature_cols].skew()
    kurt_after = df_transformed[feature_cols].kurt()

    # Build comparison summary table
    comparison_df = pd.DataFrame({
        "Skewness_Before": skew_before.round(3),
        "Skewness_After": skew_after.round(3),
        "Kurtosis_Before": kurt_before.round(3),
        "Kurtosis_After": kurt_after.round(3),
        "Status": ["Normalized (Gaussian)" if abs(s) < 0.5 else "Substantially Reduced" for s in skew_after]
    })

    # Save comparison CSV
    comparison_path = os.path.join(output_dir, "skewness_kurtosis_comparison.csv")
    comparison_df.to_csv(comparison_path)
    logger.info(f"Comparison report saved to: {comparison_path}")

    # Save normalized dataset
    normalized_data_path = os.path.join(output_dir, "prepared_feature_base_normalized.csv")
    df_transformed.to_csv(normalized_data_path, index=False)
    logger.info(f"Normalized dataset saved to: {normalized_data_path}")

    # Save the fitted transformer model for downstream branches
    transformer_path = os.path.join(output_dir, "power_transformer_yeo_johnson.joblib")
    joblib.dump(pt, transformer_path)
    logger.info(f"Fitted PowerTransformer saved to: {transformer_path}")

    print("\n" + "=" * 70)
    print("SKEWNESS & KURTOSIS BEFORE VS AFTER NORMALIZATION:")
    print("=" * 70)
    print(comparison_df.to_string())
    print("=" * 70 + "\n")

    return comparison_df


if __name__ == "__main__":
    run_skewness_correction_and_verification()
