"""
Model Comparison and Benchmark Analysis Module.
Compares CLIP + Logistic Regression vs EfficientNet-B0 across all test metrics,
generating comparison plots and analytical reports.
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import NAVARASA_CLASSES, PLOTS_DIR, REPORTS_DIR


def generate_comparison_charts(clip_metrics, effnet_metrics, output_path=None):
    """
    Renders grouped bar chart comparing CLIP and EfficientNet per-class F1-scores.
    """
    classes = NAVARASA_CLASSES
    x = np.arange(len(classes))
    width = 0.35

    clip_f1 = [clip_metrics["per_class"][c]["f1_score"] * 100 for c in classes]
    eff_f1 = [effnet_metrics["per_class"][c]["f1_score"] * 100 for c in classes]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    bars1 = ax.bar(x - width / 2, clip_f1, width, label="CLIP ViT-B/32 + LogReg", color="#1E88E5", edgecolor="#333")
    bars2 = ax.bar(x + width / 2, eff_f1, width, label="EfficientNet-B0 (Fine-Tuned)", color="#43A047", edgecolor="#333")

    ax.set_title("Navarasa Classification: Model Comparison Across All Nine Classes (Test F1-Score %)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Navarasa Class", fontsize=11, labelpad=10)
    ax.set_ylabel("F1-Score (%)", fontsize=11, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Add values on top of bars
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    if output_path is None:
        output_path = PLOTS_DIR / "model_comparison_bar.png"
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved model comparison chart to {output_path}")
    return output_path
