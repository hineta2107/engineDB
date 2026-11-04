"""
EngineFaultDB - Experiment: Sequential + Parallel Hybrid Ensemble Architecture
Architecture:
Phase 1 (Sequential Tree Boosting):
  - HistGradientBoostingClassifier trains sequentially (each tree corrects residual errors of prior trees)
  - Outputs out-of-fold probabilistic predictions / residual decision signals
Phase 2 (Parallel Tree Bagging):
  - RandomForestClassifier trains in parallel across diverse bootstrap subsamples
Phase 3 (Hybrid Integration):
  - Option A: Soft-Voting Ensemble (Weighted probability blending: Boosting + Bagging)
  - Option B: Two-Stage Stacking Classifier (Meta-Learner trained on sequential + parallel predictions)

Evaluates on Locked Test Set (8,400 samples) and compares against single models.
Generates High-Resolution Confusion Matrix and Performance Comparison Charts (300 DPI).
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
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
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
logger = logging.getLogger("HybridEnsemble")

DATA_PATH = "outputs/common_foundation/prepared_feature_base.csv"
OUTPUT_DIR = "outputs/branch_b_hybrid_ensemble"
TARGET_COL = "Fault"
CLASS_NAMES = ["Normal (0)", "Fault Type 1", "Fault Type 2", "Fault Type 3"]


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str):
    plt.figure(figsize=(8.5, 7), dpi=300)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_perc[i, j]:.1f}%)"

    sns.heatmap(
        cm, annot=annot, fmt="", cmap="YlGnBu", cbar=True,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title(title, fontsize=13, weight="bold", pad=12)
    plt.xlabel("Predicted Label (Dự đoán)", fontsize=11, weight="bold")
    plt.ylabel("True Ground Truth Label (Thực tế)", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved confusion matrix: {save_path}")


def run_hybrid_ensemble_experiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("=" * 70)
    logger.info("RUNNING HYBRID TREE ENSEMBLE: SEQUENTIAL (BOOSTING) + PARALLEL (BAGGING)")
    logger.info("=" * 70)

    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c != TARGET_COL]

    X = df[feature_cols]
    y = df[TARGET_COL]

    # 2. Stratified Split (Train 70%, Val 15%, Locked Test 15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    val_relative = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=42
    )

    # 3. Z-score scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================================================
    # PHASE 1 & 2: DEFINE BASE MODELS
    # ==========================================================================
    logger.info("1/3 Initializing Base Learners...")
    # Model 1: Sequential Trees (Gradient Boosting - each tree corrects prior errors)
    sequential_model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        random_state=42
    )

    # Model 2: Parallel Trees (Random Forest - 300 trees grown independently in parallel)
    parallel_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=22,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    # ==========================================================================
    # PHASE 3: HYBRID ENSEMBLE STRATEGIES
    # ==========================================================================
    # Approach A: Soft Voting (Weighted blending of probabilities)
    logger.info("2/3 Training Hybrid Soft-Voting Ensemble (Sequential + Parallel)...")
    voting_ensemble = VotingClassifier(
        estimators=[
            ("sequential_boosting", sequential_model),
            ("parallel_rf", parallel_model)
        ],
        voting="soft",
        weights=[1.0, 1.2],  # Slightly higher weight to the best parallel RF
        n_jobs=-1
    )
    voting_ensemble.fit(X_train_scaled, y_train)

    # Approach B: Stacking Classifier (Meta-Learner combining predictions)
    logger.info("3/3 Training Hybrid Stacking Classifier (Meta-Learner)...")
    stacking_ensemble = StackingClassifier(
        estimators=[
            ("sequential_boosting", sequential_model),
            ("parallel_rf", parallel_model)
        ],
        final_estimator=LogisticRegression(max_iter=500, random_state=42),
        cv=3,
        n_jobs=-1
    )
    stacking_ensemble.fit(X_train_scaled, y_train)

    # ==========================================================================
    # EVALUATE ON LOCKED TEST SET (8,400 SAMPLES)
    # ==========================================================================
    models_to_evaluate = {
        "Sequential_Boosting_Alone": sequential_model.fit(X_train_scaled, y_train),
        "Parallel_RF_Alone": parallel_model.fit(X_train_scaled, y_train),
        "Hybrid_Soft_Voting": voting_ensemble,
        "Hybrid_Stacking_Meta": stacking_ensemble
    }

    eval_records = []
    cm_dict = {}

    for name, m in models_to_evaluate.items():
        preds = m.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro")
        cm = confusion_matrix(y_test, preds)
        cm_dict[name] = cm

        eval_records.append({
            "Architecture": name,
            "Test_Accuracy": f"{acc*100:.2f}%",
            "Test_F1_Macro": round(float(f1), 4),
            "Test_Precision": round(float(p), 4),
            "Test_Recall": round(float(r), 4),
            "Fault0_Accuracy": f"{cm[0, 0] / cm[0].sum() * 100:.1f}%",
            "Fault1_Accuracy": f"{cm[1, 1] / cm[1].sum() * 100:.1f}%",
            "Fault2_Accuracy": f"{cm[2, 2] / cm[2].sum() * 100:.1f}%",
            "Fault3_Accuracy": f"{cm[3, 3] / cm[3].sum() * 100:.1f}%"
        })

    eval_df = pd.DataFrame(eval_records)
    eval_csv_path = os.path.join(OUTPUT_DIR, "hybrid_ensemble_comparison.csv")
    eval_df.to_csv(eval_csv_path, index=False)

    # Plot Confusion Matrix for Hybrid Soft-Voting
    plot_confusion_matrix(
        cm_dict["Hybrid_Soft_Voting"],
        title="Hybrid Ensemble (Sequential Boosting + Parallel RF) - Confusion Matrix\nLocked Test Set (8,400 samples)",
        save_path=os.path.join(OUTPUT_DIR, "plot1_hybrid_voting_confusion_matrix.png")
    )

    # Plot Bar Chart comparison
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9.5, 5.2), dpi=300)
    plot_df = eval_df[["Architecture", "Test_F1_Macro"]].copy()
    plot_df["Architecture"] = plot_df["Architecture"].str.replace("_", " ")
    bars = plt.bar(plot_df["Architecture"], plot_df["Test_F1_Macro"], color=["#3498db", "#2ecc71", "#e67e22", "#9b59b6"], width=0.55)
    plt.title("Macro F1-Score Comparison: Single Trees vs Hybrid Ensembles", fontsize=13, weight="bold", pad=12)
    plt.ylabel("Macro F1-Score on Locked Test Set", fontsize=11, weight="bold")
    plt.ylim(0.70, 0.78)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.001, f"{h:.4f}", ha='center', va='bottom', fontsize=10, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot2_hybrid_f1_comparison.png"))
    plt.close()

    # Save best hybrid model
    joblib.dump(voting_ensemble, os.path.join(OUTPUT_DIR, "hybrid_soft_voting_model.joblib"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "hybrid_scaler.joblib"))

    print("\n" + "=" * 95)
    print("HYBRID ENSEMBLE EXPERIMENT RESULTS (SEQUENTIAL + PARALLEL TREES):")
    print("=" * 95)
    print(eval_df.to_string(index=False))
    print("=" * 95 + "\n")

    return eval_df


if __name__ == "__main__":
    run_hybrid_ensemble_experiment()

