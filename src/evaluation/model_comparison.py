"""
Model Comparison and Benchmark Analysis Module.
Compares Model A (CLIP + Logistic Regression), Model B (CLIP + SVM),
and Model C (EfficientNet-B0) across all test metrics,
generating comparison plots and analytical reports.
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import NAVARASA_CLASSES, PLOTS_DIR, REPORTS_DIR


def generate_comparison_charts(models_dict, output_path=None):
    """
    Renders grouped bar charts comparing all models per-class F1-scores
    and overall summary metrics.
    """
    classes = NAVARASA_CLASSES
    n_classes = len(classes)
    n_models = len(models_dict)
    model_names = list(models_dict.keys())

    # Color palette
    colors = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A", "#C2185B"][:n_models]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), dpi=300)

    # 1. Per-Class F1-Scores Grouped Bar Chart
    x = np.arange(n_classes)
    total_width = 0.8
    bar_width = total_width / n_models

    for m_idx, (m_name, metrics) in enumerate(models_dict.items()):
        f1_scores = [metrics["per_class"][c]["f1_score"] * 100 for c in classes]
        offset = (m_idx - (n_models - 1) / 2) * bar_width
        bars = ax1.bar(x + offset, f1_scores, bar_width, label=m_name,
                       color=colors[m_idx], edgecolor="#333", alpha=0.9)
        # Add labels
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

    ax1.set_title("Navarasa Classification: Per-Class F1-Score (%) Across Models",
                  fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Navarasa Expression Class", fontsize=11, labelpad=8)
    ax1.set_ylabel("F1-Score (%)", fontsize=11, labelpad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=20, ha="right", fontsize=10)
    ax1.set_ylim(0, 108)
    ax1.legend(frameon=True, fontsize=10, loc="lower right")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Overall Metrics Comparison Grouped Bar Chart
    summary_metrics_keys = [
        ("accuracy", "Accuracy"),
        ("macro_precision", "Macro Prec"),
        ("macro_recall", "Macro Rec"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
    ]
    x_sum = np.arange(len(summary_metrics_keys))
    bar_width_sum = total_width / n_models

    for m_idx, (m_name, metrics) in enumerate(models_dict.items()):
        values = [metrics[k] * 100 for k, _ in summary_metrics_keys]
        offset = (m_idx - (n_models - 1) / 2) * bar_width_sum
        bars = ax2.bar(x_sum + offset, values, bar_width_sum, label=m_name,
                       color=colors[m_idx], edgecolor="#333", alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.set_title("Navarasa Classification: Overall Benchmark Metrics Comparison",
                  fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Evaluation Metric", fontsize=11, labelpad=8)
    ax2.set_ylabel("Score (%)", fontsize=11, labelpad=8)
    ax2.set_xticks(x_sum)
    ax2.set_xticklabels([name for _, name in summary_metrics_keys], fontsize=10)
    min_val = min(min(metrics[k] * 100 for k, _ in summary_metrics_keys) for metrics in models_dict.values())
    ax2.set_ylim(max(0, min_val - 8), 108)
    ax2.legend(frameon=True, fontsize=10, loc="lower right")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if output_path is None:
        output_path = PLOTS_DIR / "model_comparison_bar.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved model comparison chart to {output_path}")
    return output_path
