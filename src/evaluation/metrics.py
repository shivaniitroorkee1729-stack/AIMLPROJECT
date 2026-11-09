"""
Evaluation metrics computation module for Navarasa classification models.
Calculates Accuracy, Macro/Weighted Precision, Recall, F1-scores, and per-class metrics.
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import NAVARASA_CLASSES


def compute_comprehensive_metrics(y_true, y_pred, y_probs=None, class_names=NAVARASA_CLASSES):
    """
    Computes all standard multiclass evaluation metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred))

    # Macro metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Weighted metrics
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(class_names))), zero_division=0
    )

    per_class_dict = {}
    for idx, cname in enumerate(class_names):
        per_class_dict[cname] = {
            "precision": float(p_class[idx]),
            "recall": float(r_class[idx]),
            "f1_score": float(f1_class[idx]),
            "support": int(support_class[idx]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }
