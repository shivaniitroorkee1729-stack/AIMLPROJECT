"""
Dataset Size Experiments Module.
Investigates the influence of training dataset size on classification performance.
Evaluates training with 50, 100, 200, and Full samples per class, keeping
validation and test datasets strictly fixed across all experiments.
Generates performance metrics and comparative scaling plots.
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    MODELS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    CLIP_EMBEDDINGS_CACHE,
    DATASET_SIZE_EXPERIMENT_PATH,
    NAVARASA_CLASSES,
)


def run_dataset_size_experiments():
    print("=" * 60)
    print(" DATASET SIZE EXPERIMENTS (50, 100, 200 vs FULL) ")
    print("=" * 60)

    if not CLIP_EMBEDDINGS_CACHE.exists():
        raise FileNotFoundError(f"CLIP embeddings cache not found at {CLIP_EMBEDDINGS_CACHE}")

    data = np.load(CLIP_EMBEDDINGS_CACHE)
    X_train_full = data["X_train"]
    y_train_full = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(f"Full Train: {X_train_full.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    target_sizes = [50, 100, 200, "Full"]
    experiment_results = []

    np.random.seed(42)

    for size in target_sizes:
        if size == "Full":
            X_sub = X_train_full
            y_sub = y_train_full
            effective_per_class = int(round(len(y_sub) / len(NAVARASA_CLASSES)))
            label_str = f"Full ({effective_per_class}/class)"
        else:
            selected_indices = []
            for cls_idx in range(len(NAVARASA_CLASSES)):
                cls_indices = np.where(y_train_full == cls_idx)[0]
                if len(cls_indices) < size:
                    print(f"Warning: Class {cls_idx} has only {len(cls_indices)} samples (< {size})")
                    sampled = cls_indices
                else:
                    sampled = np.random.choice(cls_indices, size=size, replace=False)
                selected_indices.extend(sampled)

            selected_indices = np.array(selected_indices)
            X_sub = X_train_full[selected_indices]
            y_sub = y_train_full[selected_indices]
            effective_per_class = size
            label_str = f"{size}/class ({len(y_sub)} total)"

        print(f"\nTraining on {label_str}...")

        # Train classifier
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
        clf.fit(X_sub, y_sub)

        # Evaluate on fixed test set
        test_preds = clf.predict(X_test)
        test_acc = accuracy_score(y_test, test_preds)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, test_preds, average="macro", zero_division=0
        )
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(
            y_test, test_preds, average="weighted", zero_division=0
        )

        print(f"  --> Test Accuracy: {test_acc * 100:.2f}% | Macro F1: {f1_macro * 100:.2f}%")

        experiment_results.append({
            "size_setting": str(size),
            "samples_per_class": effective_per_class,
            "total_training_samples": int(len(y_sub)),
            "test_accuracy": float(test_acc),
            "test_macro_f1": float(f1_macro),
            "test_weighted_f1": float(f1_w),
            "test_macro_precision": float(p_macro),
            "test_macro_recall": float(r_macro),
        })

    # Save results JSON
    with open(DATASET_SIZE_EXPERIMENT_PATH, "w") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\nSaved experiment results to {DATASET_SIZE_EXPERIMENT_PATH}")

    # Plot results
    plot_scaling_graph(experiment_results, PLOTS_DIR / "dataset_size_vs_performance.png")

    return experiment_results


def plot_scaling_graph(results, output_path):
    samples_per_class = [r["samples_per_class"] for r in results]
    accuracies = [r["test_accuracy"] * 100 for r in results]
    macro_f1s = [r["test_macro_f1"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ax.plot(samples_per_class, accuracies, label="Test Accuracy (%)",
            color="#0D47A1", marker="o", linewidth=2.5, markersize=8)
    ax.plot(samples_per_class, macro_f1s, label="Macro F1-Score (%)",
            color="#2E7D32", marker="s", linewidth=2.5, markersize=8, linestyle="--")

    for i, txt in enumerate(accuracies):
        ax.annotate(f"{txt:.2f}%", (samples_per_class[i], accuracies[i]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold", fontsize=9)

    ax.set_title("Training Samples per Class vs. Model Test Performance",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Training Samples per Class", fontsize=11, labelpad=10)
    ax.set_ylabel("Performance (%)", fontsize=11, labelpad=10)
    ax.set_xticks(samples_per_class)
    ax.set_xticklabels([f"{s}" for s in samples_per_class])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)
    ax.set_ylim(min(macro_f1s) - 5, max(accuracies) + 5)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved dataset size scaling plot to {output_path}")


if __name__ == "__main__":
    run_dataset_size_experiments()
