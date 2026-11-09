"""
Phase 4: Training & Comparing Model A (CLIP + Logistic Regression) and Model B (CLIP + SVM)
Strictly trained and evaluated on the validated facial crops dataset.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    SPLITS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    CLIP_MODEL_PATH,
    CLIP_EMBEDDINGS_CACHE,
    NAVARASA_CLASSES,
    CLASS_TO_IDX,
    DEVICE,
)
from src.data.dataset_loader import NavarasaDataset
from src.data.preprocessing import get_clip_transforms
from src.models.clip_model import CLIPClassifierPipeline

CLIP_SVM_PATH = MODELS_DIR / "clip_svm.joblib"
CROPS_EMBEDDINGS_CACHE = MODELS_DIR / "clip_face_crops_embeddings.npz"


def extract_or_load_crop_embeddings(pipeline, force_recompute=False):
    if CROPS_EMBEDDINGS_CACHE.exists() and not force_recompute:
        print(f"Loading cached facial crop embeddings from {CROPS_EMBEDDINGS_CACHE}...")
        data = np.load(CROPS_EMBEDDINGS_CACHE)
        return (
            data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
            data["X_test"], data["y_test"]
        )

    print("Extracting CLIP embeddings for facial crops (Train, Val, Test)...")
    clip_transform = get_clip_transforms()

    train_ds = NavarasaDataset(SPLITS_DIR / "train_crops.csv", transform=clip_transform)
    val_ds = NavarasaDataset(SPLITS_DIR / "val_crops.csv", transform=clip_transform)
    test_ds = NavarasaDataset(SPLITS_DIR / "test_crops.csv", transform=clip_transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    t0 = time.time()
    X_train, y_train = pipeline.extract_batch_embeddings(train_loader)
    X_val, y_val = pipeline.extract_batch_embeddings(val_loader)
    X_test, y_test = pipeline.extract_batch_embeddings(test_loader)
    elapsed = time.time() - t0
    print(f"Embedding extraction finished in {elapsed:.1f} seconds.")

    print(f"Saving crop embeddings to {CROPS_EMBEDDINGS_CACHE}...")
    np.savez_compressed(
        CROPS_EMBEDDINGS_CACHE,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_logistic_regression(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 60)
    print(" MODEL A: HYPERPARAMETER TUNING FOR LOGISTIC REGRESSION ")
    print("=" * 60)

    candidate_c = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    candidate_weights = [None, "balanced"]
    best_clf = None
    best_score = 0.0
    best_params = {}

    for c_val in candidate_c:
        for cw in candidate_weights:
            clf = LogisticRegression(
                C=c_val,
                class_weight=cw,
                max_iter=1000,
                solver="lbfgs",
                random_state=42
            )
            clf.fit(X_train, y_train)
            val_preds = clf.predict(X_val)
            val_acc = accuracy_score(y_val, val_preds)
            val_f1 = f1_score(y_val, val_preds, average="macro")
            score = (val_acc + val_f1) / 2.0

            cw_str = "balanced" if cw == "balanced" else "none"
            print(f"  C={c_val:<5} | class_weight={cw_str:<8} -> Val Acc: {val_acc*100:.2f}%, Macro F1: {val_f1*100:.2f}%")

            if score > best_score:
                best_score = score
                best_clf = clf
                best_params = {"C": c_val, "class_weight": cw, "val_acc": val_acc, "val_f1": val_f1}

    print(f"\n[Optimal Model A] C={best_params['C']}, class_weight={best_params['class_weight']} -> Val Acc: {best_params['val_acc']*100:.2f}%, Macro F1: {best_params['val_f1']*100:.2f}%")
    return best_clf, best_params


def train_svm(X_train, y_train, X_val, y_val):
    print("\n" + "=" * 60)
    print(" MODEL B: HYPERPARAMETER TUNING FOR SUPPORT VECTOR MACHINE (SVM) ")
    print("=" * 60)

    # Search across linear and RBF kernels, C values, class weighting
    candidate_kernels = ["linear", "rbf"]
    candidate_c = [0.1, 0.5, 1.0, 2.0, 5.0]
    best_svm = None
    best_score = 0.0
    best_params = {}

    for kernel in candidate_kernels:
        for c_val in candidate_c:
            for cw in [None, "balanced"]:
                clf = SVC(
                    C=c_val,
                    kernel=kernel,
                    class_weight=cw,
                    probability=True,
                    random_state=42
                )
                clf.fit(X_train, y_train)
                val_preds = clf.predict(X_val)
                val_acc = accuracy_score(y_val, val_preds)
                val_f1 = f1_score(y_val, val_preds, average="macro")
                score = (val_acc + val_f1) / 2.0

                cw_str = "balanced" if cw == "balanced" else "none"
                print(f"  Kernel={kernel:<6} | C={c_val:<4} | class_weight={cw_str:<8} -> Val Acc: {val_acc*100:.2f}%, Macro F1: {val_f1*100:.2f}%")

                if score > best_score:
                    best_score = score
                    best_svm = clf
                    best_params = {"kernel": kernel, "C": c_val, "class_weight": cw, "val_acc": val_acc, "val_f1": val_f1}

    print(f"\n[Optimal Model B] Kernel={best_params['kernel']}, C={best_params['C']}, class_weight={best_params['class_weight']} -> Val Acc: {best_params['val_acc']*100:.2f}%, Macro F1: {best_params['val_f1']*100:.2f}%")
    return best_svm, best_params


def evaluate_classifier(clf, X_test, y_test, model_name):
    print(f"\n--- Evaluating {model_name} on Held-Out Test Facial Crops ({len(y_test)} images) ---")
    test_preds = clf.predict(X_test)
    test_probs = clf.predict_proba(X_test)
    test_acc = accuracy_score(y_test, test_preds)

    rep = classification_report(
        y_test, test_preds,
        target_names=NAVARASA_CLASSES,
        output_dict=True
    )
    rep_text = classification_report(
        y_test, test_preds,
        target_names=NAVARASA_CLASSES
    )
    cm = confusion_matrix(y_test, test_preds)
    print(f"\nTest Accuracy ({model_name}): {test_acc*100:.2f}%")
    print(rep_text)

    return {
        "model_name": model_name,
        "test_accuracy": float(test_acc),
        "macro_precision": float(rep["macro avg"]["precision"]),
        "macro_recall": float(rep["macro avg"]["recall"]),
        "macro_f1": float(rep["macro avg"]["f1-score"]),
        "weighted_f1": float(rep["weighted avg"]["f1-score"]),
        "per_class": {
            cls: {
                "precision": float(rep[cls]["precision"]),
                "recall": float(rep[cls]["recall"]),
                "f1_score": float(rep[cls]["f1-score"]),
                "support": int(rep[cls]["support"])
            }
            for cls in NAVARASA_CLASSES
        },
        "confusion_matrix": cm.tolist()
    }


def run_clip_training_and_comparison():
    pipeline = CLIPClassifierPipeline(device=DEVICE)

    X_train, y_train, X_val, y_val, X_test, y_test = extract_or_load_crop_embeddings(pipeline)
    print(f"\nEmbeddings shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 1. Train Model A: CLIP + Logistic Regression
    best_logreg, params_lr = train_logistic_regression(X_train, y_train, X_val, y_val)
    joblib.dump({
        "classifier": best_logreg,
        "classes": NAVARASA_CLASSES,
        "clip_model_name": "ViT-B-32",
        "clip_pretrained": "openai",
        "embedding_dim": 512,
        "hyperparameters": params_lr,
    }, CLIP_MODEL_PATH)
    print(f"Saved Model A (LogReg) to: {CLIP_MODEL_PATH}")

    # 2. Train Model B: CLIP + SVM
    best_svm, params_svm = train_svm(X_train, y_train, X_val, y_val)
    joblib.dump({
        "classifier": best_svm,
        "classes": NAVARASA_CLASSES,
        "clip_model_name": "ViT-B-32",
        "clip_pretrained": "openai",
        "embedding_dim": 512,
        "hyperparameters": params_svm,
    }, CLIP_SVM_PATH)
    print(f"Saved Model B (SVM) to: {CLIP_SVM_PATH}")

    # 3. Evaluate both on untouched held-out test crops
    eval_logreg = evaluate_classifier(best_logreg, X_test, y_test, "Model A: CLIP ViT-B/32 + Logistic Regression")
    eval_svm = evaluate_classifier(best_svm, X_test, y_test, "Model B: CLIP ViT-B/32 + SVM")

    # Save metrics
    metrics_summary = {
        "clip_logreg": eval_logreg,
        "clip_svm": eval_svm,
    }
    with open(REPORTS_DIR / "clip_models_evaluation.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n>>> MODEL A & MODEL B TRAINING & EVALUATION ON FACE CROPS COMPLETE! <<<")
    return eval_logreg, eval_svm


if __name__ == "__main__":
    run_clip_training_and_comparison()
