"""
Confusion Matrix Visualization and Heatmap Generation.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import NAVARASA_CLASSES, PLOTS_DIR


def plot_confusion_matrix_heatmap(
    cm,
    model_name="Model",
    output_path=None,
    normalize=True,
    class_names=NAVARASA_CLASSES
):
    """
    Renders and saves a heatmap of the confusion matrix.
    """
    cm = np.array(cm)
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_display = np.nan_to_num(cm_display)
        fmt = ".2f"
        title = f"{model_name} — Normalized Confusion Matrix"
    else:
        cm_display = cm
        fmt = "d"
        title = f"{model_name} — Confusion Matrix (Raw Counts)"

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
        linewidths=0.5,
        linecolor="#dddddd",
        ax=ax
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Navarasa", fontsize=11, labelpad=10)
    ax.set_ylabel("True Ground Truth Navarasa", fontsize=11, labelpad=10)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Saved confusion matrix heatmap to {output_path}")

    plt.close()
    return fig
