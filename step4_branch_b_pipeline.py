"""
Branch B: Supervised Fault Classification Pipeline (Top-5 Features)
Implements all steps from Branch B diagram:
- B1: Target Definition & Stratified Train/Val/Locked Test Split
- B2: Train-only Preprocessing (Leakage-Safe Scaling)
- B3: Multi-Model Training & Stratified Cross-Validation (LogisticRegression, RandomForest, HistGradientBoosting, KNN, MLP)
- B4: Model Comparison & Selection (Accuracy, Macro F1, Precision, Recall)
- B5: Hyperparameter Tuning for the Best Model
- B6: Locked Test Set Evaluation (Opened once)
- B7: Production Artifacts Export (.joblib model, scaler, metadata JSON)
"""

import os
import sys
import json
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BranchBPipeline")

DATA_PATH = "outputs/datasets/data_top5_features.csv"
OUTPUT_DIR = "outputs/branch_b_top5"
TARGET_COL = "Fault"
TOP_5_FEATURES = ["CO", "Force", "Consumption L/H", "Power", "HC"]


class BranchBClassificationPipeline:
    def __init__(self, data_path: str = DATA_PATH, output_dir: str = OUTPUT_DIR):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.df = None
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.scaler = None
        self.X_train_scaled, self.X_val_scaled, self.X_test_scaled = None, None, None

        self.candidate_models = {}
        self.benchmark_df = None
        self.best_model_name = None
        self.best_model = None
        self.final_test_results = {}

    # --------------------------------------------------------------------------
    # B1: TARGET DEFINITION & STRATIFIED SPLIT
    # --------------------------------------------------------------------------
    def step_b1_stratified_split(self, test_size: float = 0.15, val_size: float = 0.15):
        logger.info("=" * 65)
        logger.info("BRANCH B - STEP B1: STRATIFIED SPLIT (70% Train, 15% Val, 15% Test)")
        logger.info("=" * 65)

        self.df = pd.read_csv(self.data_path)
        logger.info(f"Loaded dataset: {len(self.df)} records with features: {TOP_5_FEATURES}")

        X = self.df[TOP_5_FEATURES]
        y = self.df[TARGET_COL]

        # First split off the Locked Test set (15%)
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # Split remaining 85% into Train (70% of total) and Val (15% of total)
        val_relative = val_size / (1.0 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=42
        )

        logger.info(f"Split breakdown -> Train: {len(self.X_train)}, Val: {len(self.X_val)}, Locked Test: {len(self.X_test)}")
        logger.info(f"Train class balance: {self.y_train.value_counts().to_dict()}")

    # --------------------------------------------------------------------------
    # B2: TRAIN-ONLY PREPROCESSING (LEAKAGE-SAFE SCALING)
    # --------------------------------------------------------------------------
    def step_b2_train_only_preprocessing(self):
        logger.info("=" * 65)
        logger.info("BRANCH B - STEP B2: TRAIN-ONLY SCALING (LEAKAGE-SAFE)")
        logger.info("=" * 65)

        self.scaler = StandardScaler()
        # Fit ONLY on training data
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        # Transform validation and test data
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        logger.info("StandardScaler fitted strictly on training partition. Zero leakage to Val/Test.")

    # --------------------------------------------------------------------------
    # B3 & B4: MODEL TRAINING, CROSS-VALIDATION & BENCHMARK COMPARISON
    # --------------------------------------------------------------------------
    def step_b3_b4_model_benchmark(self) -> pd.DataFrame:
        logger.info("=" * 65)
        logger.info("BRANCH B - STEP B3 & B4: MULTI-MODEL 5-FOLD CV & BENCHMARK")
        logger.info("=" * 65)

        self.candidate_models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
            "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            "MLP_NeuralNet": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = []

        for name, clf in self.candidate_models.items():
            logger.info(f"Running 5-fold Stratified CV on {name}...")
            cv_scores = cross_val_score(clf, self.X_train_scaled, self.y_train, cv=cv, scoring="f1_macro", n_jobs=-1)

            # Fit on full train set and evaluate on validation set
            clf.fit(self.X_train_scaled, self.y_train)
            val_preds = clf.predict(self.X_val_scaled)
            val_acc = accuracy_score(self.y_val, val_preds)
            val_p, val_r, val_f1, _ = precision_recall_fscore_support(self.y_val, val_preds, average="macro")

            results.append({
                "Model": name,
                "CV_F1_Macro_Mean": round(cv_scores.mean(), 4),
                "CV_F1_Macro_Std": round(cv_scores.std(), 4),
                "Val_Accuracy": round(val_acc, 4),
                "Val_F1_Macro": round(val_f1, 4),
                "Val_Precision_Macro": round(val_p, 4),
                "Val_Recall_Macro": round(val_r, 4)
            })

        self.benchmark_df = pd.DataFrame(results).sort_values("Val_F1_Macro", ascending=False)
        benchmark_path = os.path.join(self.output_dir, "step_b4_model_benchmark_comparison.csv")
        self.benchmark_df.to_csv(benchmark_path, index=False)
        logger.info(f"Saved benchmark results to: {benchmark_path}")

        print("\n" + "=" * 85)
        print("MODEL BENCHMARK COMPARISON ON TOP-5 FEATURES:")
        print("=" * 85)
        print(self.benchmark_df.to_string(index=False))
        print("=" * 85 + "\n")

        self.best_model_name = self.benchmark_df.iloc[0]["Model"]
        self.best_model = self.candidate_models[self.best_model_name]
        logger.info(f">>> Top performing candidate model: {self.best_model_name} <<<")
        return self.benchmark_df

    # --------------------------------------------------------------------------
    # B5: HYPERPARAMETER TUNING FOR THE CHAMPION MODEL
    # --------------------------------------------------------------------------
    def step_b5_tune_best_model(self):
        logger.info("=" * 65)
        logger.info(f"BRANCH B - STEP B5: HYPERPARAMETER TUNING FOR {self.best_model_name}")
        logger.info("=" * 65)

        if self.best_model_name == "KNN":
            param_grid = {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            }
            grid = GridSearchCV(KNeighborsClassifier(n_jobs=-1), param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
        elif self.best_model_name == "RandomForest":
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [10, 15, 20],
                "min_samples_split": [2, 5]
            }
            grid = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid, cv=3, scoring="f1_macro", n_jobs=-1)
        else:
            param_grid = {
                "max_iter": [100, 150],
                "learning_rate": [0.05, 0.1]
            }
            grid = GridSearchCV(HistGradientBoostingClassifier(random_state=42), param_grid, cv=3, scoring="f1_macro", n_jobs=-1)

        logger.info(f"Searching best hyperparams across grid: {param_grid}")
        grid.fit(self.X_train_scaled, self.y_train)
        logger.info(f"Best parameters found: {grid.best_params_} with CV F1-Macro: {grid.best_score_:.4f}")

        self.best_model = grid.best_estimator_
        # Save tuning summary
        with open(os.path.join(self.output_dir, "step_b5_hyperparameter_tuning.json"), "w") as f:
            json.dump({"best_params": grid.best_params_, "best_cv_f1": round(grid.best_score_, 4)}, f, indent=4)

    # --------------------------------------------------------------------------
    # B6: LOCKED TEST SET EVALUATION (FINAL GENERALIZATION CHECK)
    # --------------------------------------------------------------------------
    def step_b6_locked_test_evaluation(self) -> Dict[str, Any]:
        logger.info("=" * 65)
        logger.info("BRANCH B - STEP B6: OPENING LOCKED TEST SET (8,400 UNSEEN SAMPLES)")
        logger.info("=" * 65)

        test_preds = self.best_model.predict(self.X_test_scaled)
        test_acc = accuracy_score(self.y_test, test_preds)
        test_p, test_r, test_f1, _ = precision_recall_fscore_support(self.y_test, test_preds, average="macro")
        cls_report = classification_report(self.y_test, test_preds, output_dict=True)
        cm = confusion_matrix(self.y_test, test_preds).tolist()

        self.final_test_results = {
            "model_name": self.best_model_name,
            "features_used": TOP_5_FEATURES,
            "test_sample_count": len(self.X_test),
            "test_accuracy": round(float(test_acc), 4),
            "test_f1_macro": round(float(test_f1), 4),
            "test_precision_macro": round(float(test_p), 4),
            "test_recall_macro": round(float(test_r), 4),
            "confusion_matrix": cm,
            "classification_report": cls_report
        }

        test_res_path = os.path.join(self.output_dir, "step_b6_locked_test_results.json")
        with open(test_res_path, "w") as f:
            json.dump(self.final_test_results, f, indent=4)

        logger.info(f"LOCKED TEST RESULTS -> Accuracy: {test_acc*100:.2f}%, F1-Macro: {test_f1:.4f}")
        logger.info(f"Confusion Matrix:\n{np.array(cm)}")
        return self.final_test_results

    # --------------------------------------------------------------------------
    # B7: PRODUCTION ARTIFACTS EXPORT
    # --------------------------------------------------------------------------
    def step_b7_export_production_pipeline(self):
        logger.info("=" * 65)
        logger.info("BRANCH B - STEP B7: EXPORTING PRODUCTION MODEL & SCALER")
        logger.info("=" * 65)

        model_path = os.path.join(self.output_dir, f"production_model_{self.best_model_name}.joblib")
        scaler_path = os.path.join(self.output_dir, "production_scaler.joblib")

        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)

        metadata = {
            "model_architecture": self.best_model_name,
            "features": TOP_5_FEATURES,
            "target": TARGET_COL,
            "test_metrics": {
                "accuracy": self.final_test_results["test_accuracy"],
                "f1_macro": self.final_test_results["test_f1_macro"]
            },
            "status": "Ready for deployment / Streamlit"
        }
        meta_path = os.path.join(self.output_dir, "production_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Saved Champion Model: {model_path}")
        logger.info(f"Saved Production Scaler: {scaler_path}")
        logger.info(f"Saved Production Metadata: {meta_path}")

    # --------------------------------------------------------------------------
    # EXECUTE FULL PIPELINE
    # --------------------------------------------------------------------------
    def run_full_branch_b(self):
        self.step_b1_stratified_split()
        self.step_b2_train_only_preprocessing()
        self.step_b3_b4_model_benchmark()
        self.step_b5_tune_best_model()
        self.step_b6_locked_test_evaluation()
        self.step_b7_export_production_pipeline()
        logger.info(">>> BRANCH B SUPERVISED PIPELINE FINISHED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    pipeline = BranchBClassificationPipeline()
    pipeline.run_full_branch_b()

