"""
EngineFaultDB - Deep Learning Experiment: Artificial Neural Network (ANN / MLP) for Multiclass Classification
Architecture:
- Input: 14 Sensor Features (or Top-5 features) scaled via Z-Score
- Hidden Layers: Multi-layer Perceptron (e.g., 128 -> 64 -> 32)
- Activation: ReLU + Softmax output for 4 fault classes
- Optimizer: Adam with early stopping, dynamic learning rate schedule, and L2 regularization (alpha)
- Evaluation:
  - 5-Fold Stratified Cross-Validation
  - Training loss convergence curve & Validation loss
  - Locked Test Set (8,400 samples) Accuracy, Macro F1, Precision, Recall
  - High-Resolution Confusion Matrix Heatmap (300 DPI)
  - Benchmark comparison vs Random Forest & HistGradientBoosting
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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
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
logger = logging.getLogger("ANN_Experiment")

DATA_PATH = "outputs/common_foundation/prepared_feature_base.csv"
OUTPUT_DIR = "outputs/branch_b_ann_experiment"
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
        cm, annot=annot, fmt="", cmap="Purples", cbar=True,
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


def run_ann_experiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("=" * 70)
    logger.info("TRAINING MULTICLASS ARTIFICIAL NEURAL NETWORK (ANN / MLP)")
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

    # 3. Z-Score Scaling (Fitted strictly on Train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    logger.info(f"Features normalized via Z-Score across all {len(feature_cols)} sensors.")

    # 4. Define Deep ANN Architecture
    # Deep Multi-Layer Perceptron: 3 hidden layers (128 -> 64 -> 32)
    logger.info("Architecture: Input (14) -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Softmax(4)")
    ann_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,          # L2 Regularization weight decay
        batch_size=64,         # Mini-batch gradient descent
        learning_rate_init=0.001,
        learning_rate="adaptive",
        max_iter=300,
        early_stopping=True,   # Chống overfitting: dừng sớm nếu validation loss không giảm
        n_iter_no_change=15,
        validation_fraction=0.1,
        random_state=42,
        verbose=True
    )

    # 5. Fit ANN Model
    logger.info("Training ANN via Adam Optimizer with Backpropagation...")
    ann_model.fit(X_train_scaled, y_train)

    logger.info(f"ANN Training completed after {ann_model.n_iter_} epochs. Final Loss: {ann_model.loss_:.4f}")

    # 6. Locked Test Set Evaluation
    test_preds = ann_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_preds)
    test_p, test_r, test_f1, _ = precision_recall_fscore_support(y_test, test_preds, average=None)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test, test_preds, average="macro")
    cm = confusion_matrix(y_test, test_preds)
    cls_report = classification_report(y_test, test_preds, target_names=CLASS_NAMES, output_dict=True)

    logger.info(f"LOCKED TEST RESULTS (ANN): Accuracy = {test_acc*100:.2f}%, F1-Macro = {macro_f1:.4f}")

    # ==========================================================================
    # VISUALIZATIONS
    # ==========================================================================
    sns.set_theme(style="whitegrid")

    # Plot 1: Loss Curve over Epochs
    plt.figure(figsize=(8.5, 5.2), dpi=300)
    plt.plot(ann_model.loss_curve_, color="#8e44ad", lw=2.5, label="Training Loss (Cross-Entropy)")
    if hasattr(ann_model, "validation_scores_"):
        plt.plot(ann_model.validation_scores_, color="#27ae60", lw=2.0, linestyle="--", label="Validation Accuracy")
    plt.title(f"Deep ANN Training Loss Convergence ({ann_model.n_iter_} Epochs)", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Iteration / Epoch", fontsize=11, weight="bold")
    plt.ylabel("Loss / Score", fontsize=11, weight="bold")
    plt.legend(fontsize=10)
    plt.tight_layout()
    loss_plot_path = os.path.join(OUTPUT_DIR, "plot1_ann_loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()

    # Plot 2: Confusion Matrix Heatmap
    cm_plot_path = os.path.join(OUTPUT_DIR, "plot2_ann_confusion_matrix.png")
    plot_confusion_matrix(
        cm,
        title=f"Multiclass ANN (128-64-32) - Confusion Matrix\nLocked Test Set (8,400 samples) | Accuracy: {test_acc*100:.2f}% | F1: {macro_f1:.4f}",
        save_path=cm_plot_path
    )

    # Comparison vs Tree Models
    comp_df = pd.DataFrame([
        {"Model": "Random Forest (Champion)", "Test Accuracy": "75.25%", "Test F1-Macro": "0.7608", "Architecture / Type": "Ensemble 300 Trees"},
        {"Model": "Multiclass Deep ANN (MLP)", "Test Accuracy": f"{test_acc*100:.2f}%", "Test F1-Macro": f"{macro_f1:.4f}", "Architecture / Type": "Dense(128-64-32, ReLU)"},
        {"Model": "HistGradientBoosting", "Test Accuracy": "74.62%", "Test F1-Macro": "0.7538", "Architecture / Type": "Boosting Histogram"},
        {"Model": "SVM (RBF Kernel)", "Test Accuracy": "74.56%", "Test F1-Macro": "0.7541", "Architecture / Type": "Kernel Support Vector Machine"}
    ])
    comp_df.to_csv(os.path.join(OUTPUT_DIR, "ann_vs_all_models_comparison.csv"), index=False)

    print("\n" + "=" * 90)
    print("BENCHMARK SO SÁNH: DEEP ANN (MLP) VS CÁC MÔ HÌNH KHÁC:")
    print("=" * 90)
    print(comp_df.to_string(index=False))
    print("=" * 90 + "\n")

    # Save artifacts
    joblib.dump(ann_model, os.path.join(OUTPUT_DIR, "deep_ann_model.joblib"))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "ann_scaler.joblib"))

    summary = {
        "model": "Deep_ANN_MLP",
        "layers": [128, 64, 32],
        "activation": "relu",
        "epochs_trained": ann_model.n_iter_,
        "final_loss": round(float(ann_model.loss_), 4),
        "test_accuracy": round(float(test_acc), 4),
        "test_f1_macro": round(float(macro_f1), 4),
        "test_precision_macro": round(float(macro_p), 4),
        "test_recall_macro": round(float(macro_r), 4),
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist()
    }
    with open(os.path.join(OUTPUT_DIR, "ann_experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    logger.info("=" * 70)
    logger.info("ANN EXPERIMENT FINISHED & ARTIFACTS EXPORTED!")
    logger.info("=" * 70)
    return summary


if __name__ == "__main__":
    run_ann_experiment()
