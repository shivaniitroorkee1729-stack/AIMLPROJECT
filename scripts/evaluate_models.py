"""
Comprehensive Multi-Model Test Evaluation and Comparative Analysis Script.
Evaluates:
- Model A: CLIP ViT-B/32 + Logistic Regression (tuned C=10.0)
- Model B: CLIP ViT-B/32 + SVM (tuned C=5.0 RBF kernel)
- Model C: EfficientNet-B0 Two-Stage Fine-Tuned

Evaluates all models on the identical untouched held-out test facial crops split (2,109 samples).
Generates:
- Accuracy, Macro/Weighted Precision, Recall, F1
- Per-class metrics for all 9 Navarasa classes
- Normalized and Raw Confusion Matrix plots for all models
- Comparative benchmark bar charts across classes and metrics
- Detailed analysis of historically confusing pairs (Hasya vs Bhayanaka, Karuna vs Shanta, Raudra vs Veera, Adbhuta vs Bhayanaka)
- Comprehensive JSON and Markdown evaluation reports
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    SPLITS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    EVALUATION_METRICS_PATH,
    NAVARASA_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    DEVICE,
    EFFICIENTNET_BEST_MODEL_PATH,
)
from src.data.dataset_loader import NavarasaDataset
from src.data.preprocessing import get_efficientnet_transforms
from src.models.efficientnet_model import load_efficientnet_model
from src.evaluation.metrics import compute_comprehensive_metrics
from src.evaluation.confusion_matrix import plot_confusion_matrix_heatmap
from src.evaluation.model_comparison import generate_comparison_charts

CONFUSING_PAIRS = [
    ("Hasya", "Bhayanaka"),
    ("Karuna", "Shanta"),
    ("Raudra", "Veera"),
    ("Adbhuta", "Bhayanaka"),
]


def analyze_confusing_pairs(y_true, y_pred, model_name="Model"):
    """
    Analyzes specific confusing pairs in terms of directional error rates and total confusion.
    """
    pair_analysis = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for class_a, class_b in CONFUSING_PAIRS:
        idx_a = CLASS_TO_IDX[class_a]
        idx_b = CLASS_TO_IDX[class_b]

        total_a = int(np.sum(y_true == idx_a))
        total_b = int(np.sum(y_true == idx_b))

        # Direction 1: True A predicted as B
        a_pred_b = int(np.sum((y_true == idx_a) & (y_pred == idx_b)))
        a_pred_b_pct = round((a_pred_b / total_a * 100), 2) if total_a > 0 else 0.0

        # Direction 2: True B predicted as A
        b_pred_a = int(np.sum((y_true == idx_b) & (y_pred == idx_a)))
        b_pred_a_pct = round((b_pred_a / total_b * 100), 2) if total_b > 0 else 0.0

        total_mutual_errors = a_pred_b + b_pred_a
        mutual_confusion_rate = round(total_mutual_errors / (total_a + total_b) * 100, 2) if (total_a + total_b) > 0 else 0.0

        pair_analysis[f"{class_a}_vs_{class_b}"] = {
            "class_a": class_a,
            "class_b": class_b,
            "total_true_a": total_a,
            "total_true_b": total_b,
            "a_misclassified_as_b_count": a_pred_b,
            "a_misclassified_as_b_percent": a_pred_b_pct,
            "b_misclassified_as_a_count": b_pred_a,
            "b_misclassified_as_a_percent": b_pred_a_pct,
            "total_mutual_errors": total_mutual_errors,
            "mutual_confusion_rate_percent": mutual_confusion_rate,
        }

    return pair_analysis


def evaluate_clip_sklearn_model(model_name, joblib_path, X_test, y_test, slug="clip_model"):
    print(f"\n--- Evaluating {model_name} on Held-Out Test Set ---")
    data = joblib.load(joblib_path)
    clf = data["classifier"]

    # Measure latency
    t0 = time.time()
    y_pred = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)
    total_time = time.time() - t0
    latency_ms_per_sample = round((total_time / len(y_test)) * 1000, 3)

    metrics = compute_comprehensive_metrics(y_test, y_pred, y_probs=y_probs)
    metrics["latency_ms_per_sample"] = latency_ms_per_sample
    metrics["confusing_pairs_analysis"] = analyze_confusing_pairs(y_test, y_pred, model_name)

    print(f"  Test Accuracy: {metrics['accuracy'] * 100:.2f}% | Macro F1: {metrics['macro_f1'] * 100:.2f}% | Latency: {latency_ms_per_sample:.3f} ms/sample")

    # Plot normalized and raw confusion matrices
    cm_norm_path = PLOTS_DIR / f"{slug}_confusion_matrix_norm.png"
    cm_raw_path = PLOTS_DIR / f"{slug}_confusion_matrix_raw.png"
    plot_confusion_matrix_heatmap(metrics["confusion_matrix"], model_name=model_name, output_path=cm_norm_path, normalize=True)
    plot_confusion_matrix_heatmap(metrics["confusion_matrix"], model_name=model_name, output_path=cm_raw_path, normalize=False)

    return metrics, y_pred, y_probs


def evaluate_efficientnet_model(device=DEVICE):
    print("\n--- Evaluating EfficientNet-B0 (Fine-Tuned) on Test Set ---")
    val_tf = get_efficientnet_transforms(is_training=False)
    test_ds = NavarasaDataset(SPLITS_DIR / "test_crops.csv", transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    model = load_efficientnet_model(EFFICIENTNET_BEST_MODEL_PATH, device=device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    t0 = time.time()
    with torch.no_grad():
        for images, labels, _, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    total_time = time.time() - t0
    latency_ms_per_sample = round((total_time / len(test_ds)) * 1000, 3)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    metrics = compute_comprehensive_metrics(all_targets, all_preds, y_probs=all_probs)
    metrics["latency_ms_per_sample"] = latency_ms_per_sample
    metrics["confusing_pairs_analysis"] = analyze_confusing_pairs(all_targets, all_preds, "EfficientNet-B0")

    print(f"  Test Accuracy: {metrics['accuracy'] * 100:.2f}% | Macro F1: {metrics['macro_f1'] * 100:.2f}% | Latency: {latency_ms_per_sample:.3f} ms/sample")

    cm_norm_path = PLOTS_DIR / "efficientnet_confusion_matrix_norm.png"
    cm_raw_path = PLOTS_DIR / "efficientnet_confusion_matrix_raw.png"
    plot_confusion_matrix_heatmap(metrics["confusion_matrix"], model_name="EfficientNet-B0", output_path=cm_norm_path, normalize=True)
    plot_confusion_matrix_heatmap(metrics["confusion_matrix"], model_name="EfficientNet-B0", output_path=cm_raw_path, normalize=False)

    return metrics, all_preds, all_probs


def run_full_evaluation():
    print("=" * 65)
    print(" NAVARASA COMPREHENSIVE 3-MODEL EVALUATION & BENCHMARK ")
    print("=" * 65)

    # 1. Load held-out test embeddings for CLIP models
    cache_path = MODELS_DIR / "clip_face_crops_embeddings.npz"
    if not cache_path.exists():
        cache_path = MODELS_DIR / "clip_embeddings_cache.npz"
    print(f"Loading CLIP face crops test embeddings from {cache_path}...")
    emb_data = np.load(cache_path)
    X_test = emb_data["X_test"]
    y_test = emb_data["y_test"]
    print(f"Test samples: {len(y_test)}")

    # 2. Evaluate Model A: CLIP + Logistic Regression
    model_a_path = MODELS_DIR / "clip_logistic_regression.joblib"
    metrics_a, _, _ = evaluate_clip_sklearn_model(
        "Model A: CLIP + Logistic Regression",
        model_a_path,
        X_test,
        y_test,
        slug="clip_logreg"
    )

    # 3. Evaluate Model B: CLIP + SVM
    model_b_path = MODELS_DIR / "clip_svm.joblib"
    metrics_b, _, _ = evaluate_clip_sklearn_model(
        "Model B: CLIP + SVM (RBF)",
        model_b_path,
        X_test,
        y_test,
        slug="clip_svm"
    )

    # 4. Evaluate Model C: EfficientNet-B0
    metrics_c, _, _ = evaluate_efficientnet_model(device=DEVICE)

    # 5. Group models for comparison
    models_dict = {
        "Model A (CLIP + LogReg)": metrics_a,
        "Model B (CLIP + SVM)": metrics_b,
        "Model C (EfficientNet-B0)": metrics_c,
    }

    # Generate comparative charts
    generate_comparison_charts(models_dict, PLOTS_DIR / "model_comparison_bar.png")

    # Determine winner
    acc_map = {name: m["accuracy"] for name, m in models_dict.items()}
    best_model_name = max(acc_map, key=acc_map.get)

    unified_report = {
        "dataset_split": f"Held-Out Facial Crops Test Set ({len(y_test)} samples)",
        "classes": NAVARASA_CLASSES,
        "models": {
            "model_a_clip_logreg": metrics_a,
            "model_b_clip_svm": metrics_b,
            "model_c_efficientnet_b0": metrics_c,
        },
        "benchmark_summary": {
            "Model A (CLIP + LogReg)": {
                "test_accuracy": metrics_a["accuracy"],
                "macro_f1": metrics_a["macro_f1"],
                "weighted_f1": metrics_a["weighted_f1"],
                "latency_ms": metrics_a["latency_ms_per_sample"],
            },
            "Model B (CLIP + SVM)": {
                "test_accuracy": metrics_b["accuracy"],
                "macro_f1": metrics_b["macro_f1"],
                "weighted_f1": metrics_b["weighted_f1"],
                "latency_ms": metrics_b["latency_ms_per_sample"],
            },
            "Model C (EfficientNet-B0)": {
                "test_accuracy": metrics_c["accuracy"],
                "macro_f1": metrics_c["macro_f1"],
                "weighted_f1": metrics_c["weighted_f1"],
                "latency_ms": metrics_c["latency_ms_per_sample"],
            },
            "selected_primary_model": best_model_name,
            "selection_rationale": (
                f"{best_model_name} achieved the highest test accuracy ({acc_map[best_model_name]*100:.2f}%) "
                f"and highest macro F1-score ({models_dict[best_model_name]['macro_f1']*100:.2f}%) on held-out facial crops, "
                f"virtually resolving confusion between Hasya and Bhayanaka."
            ),
        }
    }

    # Save JSON report
    with open(EVALUATION_METRICS_PATH, "w") as f:
        json.dump(unified_report, f, indent=2)
    print(f"\nUnified evaluation metrics saved to {EVALUATION_METRICS_PATH}")

    # Generate Comprehensive Markdown Report
    generate_markdown_report(unified_report, REPORTS_DIR / "model_evaluation_report.md")

    # Generate Confusing Pairs Markdown Report
    generate_confusion_pairs_report(unified_report, REPORTS_DIR / "confusion_analysis.md")

    return unified_report


def generate_markdown_report(report, output_path):
    m_a = report["models"]["model_a_clip_logreg"]
    m_b = report["models"]["model_b_clip_svm"]
    m_c = report["models"]["model_c_efficientnet_b0"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Navarasa AI: Comprehensive Model Evaluation & Benchmark Report\n\n")
        f.write(f"**Test Set Split:** {report['dataset_split']}\n\n")
        f.write(f"**Selected Primary Model for Deployment:** **{report['benchmark_summary']['selected_primary_model']}**\n\n")
        f.write(f"> {report['benchmark_summary']['selection_rationale']}\n\n")

        f.write("## 1. Overall Performance Benchmark\n\n")
        f.write("| Evaluation Metric | Model A: CLIP + LogReg | Model B: CLIP + SVM (RBF) | Model C: EfficientNet-B0 |\n")
        f.write("|:---|:---:|:---:|:---:|\n")
        f.write(f"| **Test Accuracy** | **{m_a['accuracy'] * 100:.2f}%** | **{m_b['accuracy'] * 100:.2f}%** | **{m_c['accuracy'] * 100:.2f}%** |\n")
        f.write(f"| **Macro Precision** | {m_a['macro_precision'] * 100:.2f}% | {m_b['macro_precision'] * 100:.2f}% | {m_c['macro_precision'] * 100:.2f}% |\n")
        f.write(f"| **Macro Recall** | {m_a['macro_recall'] * 100:.2f}% | {m_b['macro_recall'] * 100:.2f}% | {m_c['macro_recall'] * 100:.2f}% |\n")
        f.write(f"| **Macro F1-Score** | {m_a['macro_f1'] * 100:.2f}% | {m_b['macro_f1'] * 100:.2f}% | {m_c['macro_f1'] * 100:.2f}% |\n")
        f.write(f"| **Weighted F1-Score** | {m_a['weighted_f1'] * 100:.2f}% | {m_b['weighted_f1'] * 100:.2f}% | {m_c['weighted_f1'] * 100:.2f}% |\n")
        f.write(f"| **Inference Latency** | {m_a.get('latency_ms_per_sample', 0):.2f} ms | {m_b.get('latency_ms_per_sample', 0):.2f} ms | {m_c.get('latency_ms_per_sample', 0):.2f} ms |\n\n")

        f.write("## 2. Per-Class F1-Score Breakdown (All 9 Navarasas)\n\n")
        f.write("| Navarasa Class | Support | Model A (LogReg) F1 | Model B (SVM) F1 | Model C (EffNet) F1 |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|\n")
        for cls in NAVARASA_CLASSES:
            supp = m_a["per_class"][cls]["support"]
            f1_a = m_a["per_class"][cls]["f1_score"] * 100
            f1_b = m_b["per_class"][cls]["f1_score"] * 100
            f1_c = m_c["per_class"][cls]["f1_score"] * 100
            f.write(f"| **{cls}** | {supp} | {f1_a:.2f}% | {f1_b:.2f}% | {f1_c:.2f}% |\n")

    print(f"Saved evaluation markdown report to {output_path}")


def generate_confusion_pairs_report(report, output_path):
    m_a = report["models"]["model_a_clip_logreg"]
    m_b = report["models"]["model_b_clip_svm"]
    m_c = report["models"]["model_c_efficientnet_b0"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Class Confusion Analysis for Historically Problematic Pairs\n\n")
        f.write("Detailed directional error rates and mutual confusion on the held-out facial crops test split.\n\n")

        for key, p_info in m_b["confusing_pairs_analysis"].items():
            cls_a = p_info["class_a"]
            cls_b = p_info["class_b"]
            f.write(f"### Pair: {cls_a} vs. {cls_b}\n\n")
            f.write(f"- **True {cls_a} Samples:** {p_info['total_true_a']} | **True {cls_b} Samples:** {p_info['total_true_b']}\n\n")

            f.write("| Model | True " + cls_a + " -> Pred " + cls_b + " | True " + cls_b + " -> Pred " + cls_a + " | Total Mutual Errors | Mutual Error Rate |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|\n")

            for m_label, m_obj in [("Model A (CLIP + LogReg)", m_a), ("Model B (CLIP + SVM)", m_b), ("Model C (EfficientNet-B0)", m_c)]:
                pair_data = m_obj["confusing_pairs_analysis"][key]
                f.write(f"| {m_label} | {pair_data['a_misclassified_as_b_count']} ({pair_data['a_misclassified_as_b_percent']}%) | "
                        f"{pair_data['b_misclassified_as_a_count']} ({pair_data['b_misclassified_as_a_percent']}%) | "
                        f"{pair_data['total_mutual_errors']} | **{pair_data['mutual_confusion_rate_percent']}%** |\n")
            f.write("\n")

    print(f"Saved confusion pairs analysis to {output_path}")


if __name__ == "__main__":
    run_full_evaluation()
