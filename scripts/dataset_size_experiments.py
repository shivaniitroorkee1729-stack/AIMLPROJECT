"""
Dataset Size Experiments Module on Facial Crops.
Investigates the influence of training dataset size on classification performance.
Evaluates training with 50, 100, 200, and Full samples per class, keeping
validation and test datasets strictly fixed across all experiments.
Evaluates both Logistic Regression (Model A) and SVM (Model B).
Generates performance metrics and comparative scaling plots.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    MODELS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    DATASET_SIZE_EXPERIMENT_PATH,
    NAVARASA_CLASSES,
)


def run_dataset_size_experiments():
    print("=" * 60)
    print(" DATASET SIZE SCALING EXPERIMENTS ON FACE CROPS ")
    print("=" * 60)

    cache_path = MODELS_DIR / "clip_face_crops_embeddings.npz"
    if not cache_path.exists():
        cache_path = MODELS_DIR / "clip_embeddings_cache.npz"

    print(f"Loading embeddings from {cache_path}...")
    data = np.load(cache_path)
    X_train_full = data["X_train"]
    y_train_full = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(f"Full Dataset: Train={X_train_full.shape[0]} | Val={X_val.shape[0]} | Test={X_test.shape[0]}")

    target_sizes = [50, 100, 200, "Full"]
    results = {"logreg": [], "svm": []}

    np.random.seed(42)

    for size in target_sizes:
        if size == "Full":
            X_sub = X_train_full
            y_sub = y_train_full
            effective_per_class = int(round(len(y_sub) / len(NAVARASA_CLASSES)))
            label_str = f"Full ({effective_per_class}/class, {len(y_sub)} total)"
        else:
            selected_indices = []
            for cls_idx in range(len(NAVARASA_CLASSES)):
                cls_indices = np.where(y_train_full == cls_idx)[0]
                if len(cls_indices) < size:
                    sampled = cls_indices
                else:
                    sampled = np.random.choice(cls_indices, size=size, replace=False)
                selected_indices.extend(sampled)

            selected_indices = np.array(selected_indices)
            X_sub = X_train_full[selected_indices]
            y_sub = y_train_full[selected_indices]
            effective_per_class = size
            label_str = f"{size}/class ({len(y_sub)} total)"

        print(f"\n--- Evaluating Training Size: {label_str} ---")

        # 1. Model A: Logistic Regression (C=10.0)
        t0 = time.time()
        clf_lr = LogisticRegression(C=10.0, max_iter=1000, solver="lbfgs", random_state=42)
        clf_lr.fit(X_sub, y_sub)
        train_time_lr = time.time() - t0

        val_acc_lr = accuracy_score(y_val, clf_lr.predict(X_val))
        test_preds_lr = clf_lr.predict(X_test)
        test_acc_lr = accuracy_score(y_test, test_preds_lr)
        p_macro_lr, r_macro_lr, f1_macro_lr, _ = precision_recall_fscore_support(
            y_test, test_preds_lr, average="macro", zero_division=0
        )
        p_w_lr, r_w_lr, f1_w_lr, _ = precision_recall_fscore_support(
            y_test, test_preds_lr, average="weighted", zero_division=0
        )

        print(f"  [LogReg] Train Time: {train_time_lr:.2f}s | Val Acc: {val_acc_lr * 100:.2f}% | "
              f"Test Acc: {test_acc_lr * 100:.2f}% | Macro F1: {f1_macro_lr * 100:.2f}%")

        results["logreg"].append({
            "size_setting": str(size),
            "samples_per_class": effective_per_class,
            "total_training_samples": int(len(y_sub)),
            "train_time_seconds": round(train_time_lr, 3),
            "val_accuracy": float(val_acc_lr),
            "test_accuracy": float(test_acc_lr),
            "test_macro_f1": float(f1_macro_lr),
            "test_weighted_f1": float(f1_w_lr),
            "test_macro_precision": float(p_macro_lr),
            "test_macro_recall": float(r_macro_lr),
        })

        # 2. Model B: SVM (C=5.0, RBF kernel)
        t0 = time.time()
        clf_svm = SVC(C=5.0, kernel="rbf", probability=True, random_state=42)
        clf_svm.fit(X_sub, y_sub)
        train_time_svm = time.time() - t0

        val_acc_svm = accuracy_score(y_val, clf_svm.predict(X_val))
        test_preds_svm = clf_svm.predict(X_test)
        test_acc_svm = accuracy_score(y_test, test_preds_svm)
        p_macro_svm, r_macro_svm, f1_macro_svm, _ = precision_recall_fscore_support(
            y_test, test_preds_svm, average="macro", zero_division=0
        )
        p_w_svm, r_w_svm, f1_w_svm, _ = precision_recall_fscore_support(
            y_test, test_preds_svm, average="weighted", zero_division=0
        )

        print(f"  [SVM RBF] Train Time: {train_time_svm:.2f}s | Val Acc: {val_acc_svm * 100:.2f}% | "
              f"Test Acc: {test_acc_svm * 100:.2f}% | Macro F1: {f1_macro_svm * 100:.2f}%")

        results["svm"].append({
            "size_setting": str(size),
            "samples_per_class": effective_per_class,
            "total_training_samples": int(len(y_sub)),
            "train_time_seconds": round(train_time_svm, 3),
            "val_accuracy": float(val_acc_svm),
            "test_accuracy": float(test_acc_svm),
            "test_macro_f1": float(f1_macro_svm),
            "test_weighted_f1": float(f1_w_svm),
            "test_macro_precision": float(p_macro_svm),
            "test_macro_recall": float(r_macro_svm),
        })

    # Save results JSON
    with open(DATASET_SIZE_EXPERIMENT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved scaling experiment results to {DATASET_SIZE_EXPERIMENT_PATH}")

    # Plot comparative scaling graph
    plot_scaling_graph(results, PLOTS_DIR / "dataset_size_vs_performance.png")

    return results


def plot_scaling_graph(results, output_path):
    sizes = [r["samples_per_class"] for r in results["logreg"]]
    x_labels = [f"{r['samples_per_class']}/cls" if r["size_setting"] != "Full" else f"Full ({r['samples_per_class']})" for r in results["logreg"]]
    x_positions = np.arange(len(sizes))

    acc_lr = [r["test_accuracy"] * 100 for r in results["logreg"]]
    f1_lr = [r["test_macro_f1"] * 100 for r in results["logreg"]]

    acc_svm = [r["test_accuracy"] * 100 for r in results["svm"]]
    f1_svm = [r["test_macro_f1"] * 100 for r in results["svm"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Left plot: Accuracy comparison
    ax1.plot(x_positions, acc_svm, label="Model B: CLIP + SVM (RBF)",
             color="#2E7D32", marker="s", linewidth=2.5, markersize=8)
    ax1.plot(x_positions, acc_lr, label="Model A: CLIP + Logistic Regression",
             color="#1565C0", marker="o", linewidth=2.5, markersize=8)

    for i, (txt_svm, txt_lr) in enumerate(zip(acc_svm, acc_lr)):
        ax1.annotate(f"{txt_svm:.2f}%", (x_positions[i], txt_svm),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold", fontsize=9, color="#1B5E20")
        ax1.annotate(f"{txt_lr:.2f}%", (x_positions[i], txt_lr),
                     textcoords="offset points", xytext=(0, -14), ha="center", fontweight="bold", fontsize=9, color="#0D47A1")

    ax1.set_title("Test Accuracy vs. Training Set Size", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Training Set Size (per class)", fontsize=11)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, fontsize=10)
    ax1.set_ylim(min(min(acc_lr), min(acc_svm)) - 3, 102)

    # Right plot: Macro F1 comparison
    ax2.plot(x_positions, f1_svm, label="Model B: CLIP + SVM (RBF)",
             color="#2E7D32", marker="s", linewidth=2.5, markersize=8)
    ax2.plot(x_positions, f1_lr, label="Model A: CLIP + Logistic Regression",
             color="#1565C0", marker="o", linewidth=2.5, markersize=8)

    for i, (txt_svm, txt_lr) in enumerate(zip(f1_svm, f1_lr)):
        ax2.annotate(f"{txt_svm:.2f}%", (x_positions[i], txt_svm),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontweight="bold", fontsize=9, color="#1B5E20")
        ax2.annotate(f"{txt_lr:.2f}%", (x_positions[i], txt_lr),
                     textcoords="offset points", xytext=(0, -14), ha="center", fontweight="bold", fontsize=9, color="#0D47A1")

    ax2.set_title("Test Macro F1 vs. Training Set Size", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Training Set Size (per class)", fontsize=11)
    ax2.set_ylabel("Macro F1-Score (%)", fontsize=11)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(x_labels)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, fontsize=10)
    ax2.set_ylim(min(min(f1_lr), min(f1_svm)) - 3, 102)

    plt.suptitle("Dataset Size Scaling Experiments: Model A (LogReg) vs. Model B (SVM)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved dataset size scaling plot to {output_path}")


if __name__ == "__main__":
    run_dataset_size_experiments()
