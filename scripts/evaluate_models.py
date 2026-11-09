"""
Comprehensive Model Evaluation Script.
Evaluates both CLIP + Logistic Regression and EfficientNet-B0 on the identical held-out test dataset.
Generates:
- Test Accuracy, Macro Precision/Recall/F1, Weighted F1, Per-Class metrics
- Normalized and raw Confusion Matrix plots
- Model comparison charts
- Comprehensive evaluation JSON report
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    SPLITS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    EVALUATION_METRICS_PATH,
    NAVARASA_CLASSES,
    DEVICE,
    CLIP_EMBEDDINGS_CACHE,
    CLIP_MODEL_PATH,
    EFFICIENTNET_BEST_MODEL_PATH,
)
from src.data.dataset_loader import get_dataloaders
from src.models.clip_model import CLIPClassifierPipeline
from src.models.efficientnet_model import load_efficientnet_model
from src.evaluation.metrics import compute_comprehensive_metrics
from src.evaluation.confusion_matrix import plot_confusion_matrix_heatmap
from src.evaluation.model_comparison import generate_comparison_charts


def evaluate_clip_on_test(device=DEVICE):
    print("\n--- Evaluating CLIP + Logistic Regression on Test Set ---")
    pipeline = CLIPClassifierPipeline(device=device)
    pipeline.load_model(CLIP_MODEL_PATH)

    if CLIP_EMBEDDINGS_CACHE.exists():
        data = np.load(CLIP_EMBEDDINGS_CACHE)
        X_test = data["X_test"]
        y_test = data["y_test"]
    else:
        raise FileNotFoundError(f"CLIP embeddings cache not found at {CLIP_EMBEDDINGS_CACHE}")

    y_pred = pipeline.classifier.predict(X_test)
    y_probs = pipeline.classifier.predict_proba(X_test)

    metrics = compute_comprehensive_metrics(y_test, y_pred, y_probs=y_probs)
    print(f"CLIP Test Accuracy: {metrics['accuracy'] * 100:.2f}% | Macro F1: {metrics['macro_f1'] * 100:.2f}%")

    # Plot confusion matrix
    cm_plot_path = PLOTS_DIR / "clip_confusion_matrix.png"
    plot_confusion_matrix_heatmap(
        metrics["confusion_matrix"],
        model_name="CLIP ViT-B/32 + Logistic Regression",
        output_path=cm_plot_path
    )
    return metrics


def evaluate_efficientnet_on_test(device=DEVICE):
    print("\n--- Evaluating EfficientNet-B0 on Test Set ---")
    model = load_efficientnet_model(EFFICIENTNET_BEST_MODEL_PATH, device=device)
    _, _, test_loader = get_dataloaders(batch_size=32)

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = compute_comprehensive_metrics(all_targets, all_preds, y_probs=np.array(all_probs))
    print(f"EfficientNet-B0 Test Accuracy: {metrics['accuracy'] * 100:.2f}% | Macro F1: {metrics['macro_f1'] * 100:.2f}%")

    # Plot confusion matrix
    cm_plot_path = PLOTS_DIR / "efficientnet_confusion_matrix.png"
    plot_confusion_matrix_heatmap(
        metrics["confusion_matrix"],
        model_name="EfficientNet-B0 (Fine-Tuned)",
        output_path=cm_plot_path
    )
    return metrics


def evaluate_all_models():
    print("=" * 60)
    print(" COMPREHENSIVE TEST EVALUATION & MODEL COMPARISON ")
    print("=" * 60)

    clip_metrics = evaluate_clip_on_test()
    effnet_metrics = evaluate_efficientnet_on_test()

    # Generate comparison chart
    generate_comparison_charts(clip_metrics, effnet_metrics)

    unified_report = {
        "dataset_split": "Held-Out Test Set (2,115 samples)",
        "classes": NAVARASA_CLASSES,
        "clip_logreg": clip_metrics,
        "efficientnet_b0": effnet_metrics,
        "comparison_summary": {
            "accuracy_difference": round((effnet_metrics["accuracy"] - clip_metrics["accuracy"]) * 100, 2),
            "macro_f1_difference": round((effnet_metrics["macro_f1"] - clip_metrics["macro_f1"]) * 100, 2),
            "winner": "EfficientNet-B0" if effnet_metrics["accuracy"] >= clip_metrics["accuracy"] else "CLIP + LogReg"
        }
    }

    with open(EVALUATION_METRICS_PATH, "w") as f:
        json.dump(unified_report, f, indent=2)
    print(f"\nUnified evaluation metrics saved to {EVALUATION_METRICS_PATH}")

    # Generate readable Markdown evaluation report
    md_path = REPORTS_DIR / "model_evaluation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Navarasa Models Evaluation & Comparison Report\n\n")
        f.write("Evaluation conducted on the strictly held-out test split of **2,115 unseen images**.\n\n")

        f.write("## Overall Performance Summary\n\n")
        f.write("| Metric | CLIP ViT-B/32 + LogReg | EfficientNet-B0 (Fine-Tuned) | Difference |\n")
        f.write("|--------|-------------------------|------------------------------|------------|\n")
        f.write(f"| **Test Accuracy** | {clip_metrics['accuracy'] * 100:.2f}% | {effnet_metrics['accuracy'] * 100:.2f}% | {unified_report['comparison_summary']['accuracy_difference']:+.2f}% |\n")
        f.write(f"| **Macro Precision** | {clip_metrics['macro_precision'] * 100:.2f}% | {effnet_metrics['macro_precision'] * 100:.2f}% | {(effnet_metrics['macro_precision'] - clip_metrics['macro_precision']) * 100:+.2f}% |\n")
        f.write(f"| **Macro Recall** | {clip_metrics['macro_recall'] * 100:.2f}% | {effnet_metrics['macro_recall'] * 100:.2f}% | {(effnet_metrics['macro_recall'] - clip_metrics['macro_recall']) * 100:+.2f}% |\n")
        f.write(f"| **Macro F1-Score** | {clip_metrics['macro_f1'] * 100:.2f}% | {effnet_metrics['macro_f1'] * 100:.2f}% | {unified_report['comparison_summary']['macro_f1_difference']:+.2f}% |\n")
        f.write(f"| **Weighted F1-Score** | {clip_metrics['weighted_f1'] * 100:.2f}% | {effnet_metrics['weighted_f1'] * 100:.2f}% | {(effnet_metrics['weighted_f1'] - clip_metrics['weighted_f1']) * 100:+.2f}% |\n\n")

        f.write("## Per-Class F1-Score Breakdown\n\n")
        f.write("| Navarasa Class | CLIP Precision | CLIP Recall | CLIP F1 | EffNet Precision | EffNet Recall | EffNet F1 |\n")
        f.write("|----------------|----------------|-------------|---------|------------------|---------------|-----------|\n")
        for cls in NAVARASA_CLASSES:
            cp = clip_metrics["per_class"][cls]["precision"] * 100
            cr = clip_metrics["per_class"][cls]["recall"] * 100
            cf = clip_metrics["per_class"][cls]["f1_score"] * 100
            ep = effnet_metrics["per_class"][cls]["precision"] * 100
            er = effnet_metrics["per_class"][cls]["recall"] * 100
            ef = effnet_metrics["per_class"][cls]["f1_score"] * 100
            f.write(f"| **{cls}** | {cp:.1f}% | {cr:.1f}% | {cf:.1f}% | {ep:.1f}% | {er:.1f}% | {ef:.1f}% |\n")

    print(f"Saved evaluation markdown report to {md_path}")
    return unified_report


if __name__ == "__main__":
    evaluate_all_models()
