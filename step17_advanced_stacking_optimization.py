"""
EngineFaultDB - Step 17: Advanced Stacking Meta-Learning Optimization
Explores 4 diverse Meta-Learning architectures to resolve the Fault 2 vs Fault 3 boundary:
Architecture 1 (Linear Meta-Learner):
  - Base: Sequential (HistGradientBoosting) + Parallel (RandomForest)
  - Meta: LogisticRegression (L2 regularized, C=1.0)
Architecture 2 (Non-linear Kernel Meta-Learner):
  - Base: Sequential (HistGradientBoosting) + Parallel (RandomForest) + Multiclass MLP
  - Meta: Multi-Layer Perceptron MLP (Dense 32, ReLU)
Architecture 3 (Gradient Boosted Meta-Learner):
  - Base: Sequential (HistGradientBoosting) + Parallel (RandomForest)
  - Meta: ExtraTrees / GradientBoosting Meta-Learner
Architecture 4 (Feature-Augmented Stacking / passthrough=True):
  - Base: Sequential (HistGradientBoosting) + Parallel (RandomForest) + KNN
  - Meta: LogisticRegression with original features passed through directly alongside base predictions

Evaluates all 4 configurations on Locked Test Set (8,400 samples).
Generates side-by-side Confusion Matrices and exports performance metrics.
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
    StackingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
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
logger = logging.getLogger("AdvancedStacking")

DATA_PATH = "outputs/common_foundation/prepared_feature_base.csv"
OUTPUT_DIR = "outputs/branch_b_advanced_stacking"
TARGET_COL = "Fault"
CLASS_NAMES = ["Normal (0)", "Fault Type 1", "Fault Type 2", "Fault Type 3"]


def plot_confusion_matrices(cm_dict: dict, save_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(18, 15), dpi=300)
    axes = axes.flatten()

    for idx, (name, cm) in enumerate(cm_dict.items()):
        cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]:,}\n({cm_perc[i, j]:.1f}%)"

        sns.heatmap(
            cm, annot=annot, fmt="", cmap="Blues", cbar=True,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[idx], annot_kws={"size": 10, "weight": "bold"}
        )
        axes[idx].set_title(name, fontsize=12, weight="bold", pad=10)
        axes[idx].set_xlabel("Predicted Label (Dự đoán)", fontsize=10, weight="bold")
        axes[idx].set_ylabel("True Label (Thực tế)", fontsize=10, weight="bold")

    fig.suptitle("Comparison of 4 Advanced Stacking Architectures on Locked Test Set", fontsize=15, weight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved side-by-side confusion matrices: {save_path}")


def run_advanced_stacking_experiments():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("=" * 70)
    logger.info("RUNNING ADVANCED STACKING META-LEARNING EXPERIMENTS")
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

    # 3. Z-Score Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. Define Base Learners
    base_boosting = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, random_state=42)
    base_rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)
    base_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    base_knn = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)

    # ==========================================================================
    # DEFINE 4 STACKING ARCHITECTURES
    # ==========================================================================
    architectures = {
        "Stacking_1_LinearMeta": StackingClassifier(
            estimators=[("boosting", base_boosting), ("rf", base_rf)],
            final_estimator=LogisticRegression(C=1.0, max_iter=500, random_state=42),
            cv=3,
            n_jobs=-1
        ),
        "Stacking_2_MultiLearner_MLP": StackingClassifier(
            estimators=[("boosting", base_boosting), ("rf", base_rf), ("mlp", base_mlp)],
            final_estimator=MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=42),
            cv=3,
            n_jobs=-1
        ),
        "Stacking_3_TreeMeta_ExtraTrees": StackingClassifier(
            estimators=[("boosting", base_boosting), ("rf", base_rf)],
            final_estimator=ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            cv=3,
            n_jobs=-1
        ),
        "Stacking_4_FeaturePassthrough": StackingClassifier(
            estimators=[("boosting", base_boosting), ("rf", base_rf), ("knn", base_knn)],
            final_estimator=LogisticRegression(C=0.5, max_iter=500, random_state=42),
            passthrough=True,  # Giữ lại toàn bộ 14 features gốc cho Meta-Learner cùng xác suất
            cv=3,
            n_jobs=-1
        )
    }

    results = []
    cm_dict = {}

    for name, stack_model in architectures.items():
        logger.info(f"Training {name} with Out-Of-Fold Cross-Validation...")
        stack_model.fit(X_train_scaled, y_train)

        # Locked Test Evaluation
        preds = stack_model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro")
        cm = confusion_matrix(y_test, preds)
        cm_dict[name] = cm

        results.append({
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
        logger.info(f"{name} -> Accuracy: {acc*100:.2f}%, F1: {f1:.4f}, Fault2: {cm[2,2]/cm[2].sum()*100:.1f}%, Fault3: {cm[3,3]/cm[3].sum()*100:.1f}%")

    # Save summary table
    summary_df = pd.DataFrame(results).sort_values("Test_F1_Macro", ascending=False)
    csv_path = os.path.join(OUTPUT_DIR, "advanced_stacking_comparison.csv")
    summary_df.to_csv(csv_path, index=False)

    # Plot Confusion Matrices
    plot_confusion_matrices(cm_dict, os.path.join(OUTPUT_DIR, "plot1_4_stacking_confusion_matrices.png"))

    print("\n" + "=" * 100)
    print("COMPARISON OF 4 STACKING META-LEARNER ARCHITECTURES ON LOCKED TEST SET (8,400 SAMPLES):")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100 + "\n")

    return summary_df


if __name__ == "__main__":
    run_advanced_stacking_experiments()

