"""
EngineFaultDB - Reinforcement Learning (Q-Learning) for Automotive Engine Diagnostics
1. Environment Setup (EngineDiagnosticEnv):
   - State (S): Sensor observations [CO, Force, Consumption L/H, Power, HC] discretely binned.
   - Action (A): Diagnostic decision in {0: Normal, 1: Fault 1, 2: Fault 2, 3: Fault 3}.
   - Reward (R):
     * Correct Diagnosis: +1.0
     * Fatal Miss (Predicting Normal when vehicle is Faulty): -3.0 (Severe safety penalty)
     * Wrong Fault Class: -1.0
2. RL Agent (Q-Learning Algorithm):
   - Q-table Q(s, a) updated via Bellman Equation:
     Q(s, a) <- Q(s, a) + alpha * [Reward + gamma * max_a' Q(s', a') - Q(s, a)]
   - Epsilon-Greedy exploration decaying over training episodes.
3. Evaluation & Visualization (300 DPI):
   - Reward Convergence Curve over Episodes
   - Final Policy Confusion Matrix on Test Set
   - Comparison vs Supervised Learning
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
logger = logging.getLogger("ReinforcementLearning")

DATA_PATH = "outputs/common_foundation/prepared_feature_base.csv"
OUTPUT_DIR = "outputs/branch_b_reinforcement_learning"
TARGET_COL = "Fault"
TOP_5_FEATURES = ["CO", "Force", "Consumption L/H", "Power", "HC"]
CLASS_NAMES = ["Normal (0)", "Fault Type 1", "Fault Type 2", "Fault Type 3"]


class EngineDiagnosticEnv:
    """RL Environment representing the Engine Diagnostics Problem."""
    def __init__(self, X: np.ndarray, y: np.ndarray, n_bins: int = 5):
        self.X = X
        self.y = y
        self.n_samples = len(X)
        self.n_bins = n_bins
        self.n_actions = 4  # 4 classes

        # Discretize continuous sensor values into discrete state bins
        self.bin_edges = []
        for col_idx in range(X.shape[1]):
            # Use quantiles for uniform bin distribution
            edges = np.quantile(X[:, col_idx], np.linspace(0, 1, n_bins + 1))
            edges[0] -= 1e-5
            edges[-1] += 1e-5
            self.bin_edges.append(edges)

    def get_state(self, sample_idx: int) -> tuple:
        """Encodes continuous feature vector into a discrete state tuple."""
        row = self.X[sample_idx]
        state = []
        for col_idx, edges in enumerate(self.bin_edges):
            bin_idx = np.digitize(row[col_idx], edges) - 1
            bin_idx = max(0, min(self.n_bins - 1, bin_idx))
            state.append(bin_idx)
        return tuple(state)

    def step(self, sample_idx: int, action: int) -> float:
        """
        Executes an action and returns the reward.
        Domain-Specific Cost-Sensitive Reward:
        - True positive: +1.0
        - False negative (predicting normal when engine is broken): -3.0 (critical penalty)
        - Confusion between faults: -1.0
        """
        true_label = self.y[sample_idx]
        if action == true_label:
            return 1.0  # Reward for correct diagnosis
        elif action == 0 and true_label != 0:
            return -3.0 # Severe safety penalty: Missed Fault
        else:
            return -1.0 # Penalty for classifying wrong fault type


class QLearningDiagnosticAgent:
    def __init__(self, n_actions: int = 4, alpha: float = 0.15, gamma: float = 0.0, epsilon: float = 1.0, epsilon_decay: float = 0.9995, min_epsilon: float = 0.02):
        self.n_actions = n_actions
        self.alpha = alpha               # Learning rate
        self.gamma = gamma               # Discount factor (0.0 because each diagnosis is an episodic contextual bandit)
        self.epsilon = epsilon           # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = {}                # Q(s, a) mapping

    def get_q_values(self, state: tuple) -> np.ndarray:
        if state not in self.q_table:
            # Initialize with small zeros
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def choose_action(self, state: tuple, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)  # Explore
        q_vals = self.get_q_values(state)
        return int(np.argmax(q_vals))                 # Exploit best action

    def update(self, state: tuple, action: int, reward: float):
        """Bellman equation update rule."""
        q_vals = self.get_q_values(state)
        # Contextual bandit update: Q(s, a) <- Q(s, a) + alpha * (reward - Q(s, a))
        q_vals[action] += self.alpha * (reward - q_vals[action])
        self.q_table[state] = q_vals

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay


def run_reinforcement_learning_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("=" * 70)
    logger.info("TRAINING REINFORCEMENT LEARNING AGENT (Q-LEARNING) FOR DIAGNOSTICS")
    logger.info("=" * 70)

    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    X = df[TOP_5_FEATURES].values
    y = df[TARGET_COL].values

    # 2. Stratified Split (Train 70%, Val 15%, Locked Test 15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    # 3. Z-Score normalization
    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. Initialize RL Environment & Agent
    env_train = EngineDiagnosticEnv(X_train_val_scaled, y_train_val, n_bins=5)
    agent = QLearningDiagnosticAgent(n_actions=4, alpha=0.2, epsilon=1.0, epsilon_decay=0.9998, min_epsilon=0.01)

    # 5. Training Loop across Episodes
    n_episodes = 60000
    batch_indices = np.random.RandomState(42).choice(len(X_train_val_scaled), n_episodes, replace=True)

    reward_history = []
    moving_avg_rewards = []
    window = 1000

    logger.info(f"Agent interacting with Environment for {n_episodes} steps...")
    for step, idx in enumerate(batch_indices):
        state = env_train.get_state(idx)
        action = agent.choose_action(state, training=True)
        reward = env_train.step(idx, action)

        agent.update(state, action, reward)
        agent.decay_epsilon()

        reward_history.append(reward)
        if step >= window:
            moving_avg_rewards.append(np.mean(reward_history[-window:]))
        else:
            moving_avg_rewards.append(np.mean(reward_history))

        if (step + 1) % 15000 == 0:
            logger.info(f"Episode {step+1}/{n_episodes} | Avg Reward (last 1k): {moving_avg_rewards[-1]:.3f} | Epsilon: {agent.epsilon:.3f}")

    logger.info(f"RL Training complete. Total unique states learned: {len(agent.q_table):,}")

    # 6. Evaluation on Locked Test Set
    logger.info("Evaluating Learned RL Policy on Locked Test Set (8,400 samples)...")
    env_test = EngineDiagnosticEnv(X_test_scaled, y_test, n_bins=5)
    env_test.bin_edges = env_train.bin_edges  # Use exact same state discretization

    test_preds = []
    for i in range(len(X_test_scaled)):
        state = env_test.get_state(i)
        action = agent.choose_action(state, training=False)  # Pure exploitation (no exploration)
        test_preds.append(action)

    test_preds = np.array(test_preds)
    test_acc = accuracy_score(y_test, test_preds)
    p, r, f1, _ = precision_recall_fscore_support(y_test, test_preds, average="macro")
    cm = confusion_matrix(y_test, test_preds)
    cls_report = classification_report(y_test, test_preds, target_names=CLASS_NAMES, output_dict=True)

    logger.info(f"RL POLICY TEST METRICS: Accuracy = {test_acc*100:.2f}%, Macro F1 = {f1:.4f}")

    # ==========================================================================
    # VISUALIZATIONS (300 DPI)
    # ==========================================================================
    sns.set_theme(style="whitegrid")

    # Plot 1: Cumulative Moving Average Reward (Learning Curve)
    plt.figure(figsize=(9, 5.2), dpi=300)
    plt.plot(moving_avg_rewards, color="#2980b9", lw=2.0, label="Moving Average Reward (Window=1,000)")
    plt.axhline(y=0.0, color="gray", linestyle="--", lw=1.0)
    plt.title("Reinforcement Learning: Agent Reward Convergence (Q-Learning)", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Training Step (Episode)", fontsize=11, weight="bold")
    plt.ylabel("Mean Reward (Phần thưởng tích lũy)", fontsize=11, weight="bold")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot1_rl_reward_learning_curve.png"))
    plt.close()

    # Plot 2: Confusion Matrix Heatmap of RL Policy
    plt.figure(figsize=(8.5, 7), dpi=300)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_perc[i, j]:.1f}%)"

    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Oranges", cbar=True,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        annot_kws={"size": 11, "weight": "bold"}
    )
    plt.title(f"Reinforcement Learning (Q-Learning) - Policy Confusion Matrix\nLocked Test Set (8,400 samples) | Accuracy: {test_acc*100:.2f}% | F1: {f1:.4f}", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Predicted Action (Hành động chẩn đoán)", fontsize=11, weight="bold")
    plt.ylabel("True Fault Label (Lỗi thực tế)", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plot2_rl_confusion_matrix.png"))
    plt.close()

    # Benchmark Table
    bench_df = pd.DataFrame([
        {"Method / Paradigm": "Supervised (Random Forest)", "Accuracy": "75.25%", "Macro_F1": "0.7608", "Learning_Type": "Label-guided minimization of Gini/Entropy"},
        {"Method / Paradigm": "Reinforcement Learning (Q-Learning)", "Accuracy": f"{test_acc*100:.2f}%", "Macro_F1": f"{f1:.4f}", "Learning_Type": "Cost-sensitive Reward Maximization (Penalty on fatal errors)"}
    ])
    bench_df.to_csv(os.path.join(OUTPUT_DIR, "rl_vs_supervised_comparison.csv"), index=False)

    print("\n" + "=" * 90)
    print("REINFORCEMENT LEARNING VS SUPERVISED BENCHMARK:")
    print("=" * 90)
    print(bench_df.to_string(index=False))
    print("=" * 90 + "\n")

    summary = {
        "paradigm": "Reinforcement Learning (Q-Learning)",
        "episodes": n_episodes,
        "learned_states_count": len(agent.q_table),
        "test_accuracy": round(float(test_acc), 4),
        "test_f1_macro": round(float(f1), 4),
        "test_precision": round(float(p), 4),
        "test_recall": round(float(r), 4),
        "confusion_matrix": cm.tolist()
    }
    with open(os.path.join(OUTPUT_DIR, "rl_experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    logger.info("=" * 70)
    logger.info("REINFORCEMENT LEARNING EXPERIMENT COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    return summary


if __name__ == "__main__":
    run_reinforcement_learning_pipeline()
