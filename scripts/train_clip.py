"""
Training and evaluation script for Pipeline A: CLIP ViT-B/32 + Logistic Regression.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    SPLITS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    CLIP_MODEL_PATH,
    CLIP_EMBEDDINGS_CACHE,
    NAVARASA_CLASSES,
    DEVICE,
)
from src.data.dataset_loader import NavarasaDataset
from src.data.preprocessing import get_clip_transforms
from src.models.clip_model import CLIPClassifierPipeline
from torch.utils.data import DataLoader


def run_clip_pipeline(force_recompute=False):
    print("=" * 60)
    print(" PIPELINE A: CLIP ViT-B/32 + LOGISTIC REGRESSION ")
    print("=" * 60)

    pipeline = CLIPClassifierPipeline(device=DEVICE)

    # 1. Load or extract embeddings
    if CLIP_EMBEDDINGS_CACHE.exists() and not force_recompute:
        print(f"Found cached embeddings at {CLIP_EMBEDDINGS_CACHE}. Loading...")
        cached_data = np.load(CLIP_EMBEDDINGS_CACHE)
        X_train = cached_data["X_train"]
        y_train = cached_data["y_train"]
        X_val = cached_data["X_val"]
        y_val = cached_data["y_val"]
        X_test = cached_data["X_test"]
        y_test = cached_data["y_test"]
    else:
        print("Extracting CLIP embeddings for Train, Val, and Test splits...")
        clip_transform = get_clip_transforms()

        train_ds = NavarasaDataset(SPLITS_DIR / "train.csv", transform=clip_transform)
        val_ds = NavarasaDataset(SPLITS_DIR / "val.csv", transform=clip_transform)
        test_ds = NavarasaDataset(SPLITS_DIR / "test.csv", transform=clip_transform)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

        t0 = time.time()
        X_train, y_train = pipeline.extract_batch_embeddings(train_loader)
        X_val, y_val = pipeline.extract_batch_embeddings(val_loader)
        X_test, y_test = pipeline.extract_batch_embeddings(test_loader)
        elapsed = time.time() - t0
        print(f"Extraction completed in {elapsed:.1f} seconds.")

        # Cache embeddings to disk
        print(f"Caching extracted embeddings to {CLIP_EMBEDDINGS_CACHE}...")
        np.savez_compressed(
            CLIP_EMBEDDINGS_CACHE,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test
        )

    print(f"Dataset shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 2. Train and tune Logistic Regression
    best_clf, best_c, best_val_acc = pipeline.train_logistic_regression(
        X_train, y_train, X_val, y_val, tune_c=True
    )

    # 3. Save trained model
    pipeline.save_model(CLIP_MODEL_PATH)

    # 4. Final Evaluation on unseen held-out test set
    print("\n--- Evaluating on Held-Out Test Set (2,115 unseen images) ---")
    test_preds = best_clf.predict(X_test)
    test_probs = best_clf.predict_proba(X_test)
    test_acc = accuracy_score(y_test, test_preds)

    report_dict = classification_report(
        y_test, test_preds,
        target_names=NAVARASA_CLASSES,
        output_dict=True
    )
    report_text = classification_report(
        y_test, test_preds,
        target_names=NAVARASA_CLASSES
    )
    cm = confusion_matrix(y_test, test_preds)

    print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%\n")
    print(report_text)

    # 5. Save evaluation metrics JSON
    metrics_payload = {
        "model_name": "CLIP ViT-B/32 + Logistic Regression",
        "best_hyperparameter_C": best_c,
        "validation_accuracy": float(best_val_acc),
        "test_accuracy": float(test_acc),
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "per_class": {
            cls: {
                "precision": float(report_dict[cls]["precision"]),
                "recall": float(report_dict[cls]["recall"]),
                "f1_score": float(report_dict[cls]["f1-score"]),
                "support": int(report_dict[cls]["support"])
            }
            for cls in NAVARASA_CLASSES
        },
        "confusion_matrix": cm.tolist()
    }

    metrics_file = REPORTS_DIR / "clip_test_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved CLIP evaluation metrics to {metrics_file}")

    return metrics_payload


if __name__ == "__main__":
    run_clip_pipeline()
