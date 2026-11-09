"""
Real-World Prediction Testing Script.
Tests all 3 trained models (Model A: CLIP+LogReg, Model B: CLIP+SVM, Model C: EfficientNet-B0)
on real held-out test set facial images across all nine Navarasa classes,
with particular focus on verifying the resolution of the Hasya vs Bhayanaka confusion.
Prints top-3 predictions and confidence scores and outputs a structured JSON report.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    SPLITS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    NAVARASA_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    DEVICE,
    EFFICIENTNET_BEST_MODEL_PATH,
)
from src.models.clip_model import CLIPClassifierPipeline
from src.models.efficientnet_model import load_efficientnet_model
from src.data.preprocessing import get_efficientnet_transforms


def run_real_prediction_tests():
    print("=" * 70)
    print(" PHASE 10: REAL-WORLD PREDICTION TESTING ACROSS ALL 9 CLASSES ")
    print("=" * 70)

    test_csv_path = SPLITS_DIR / "test_crops.csv"
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test crops file not found at {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)

    # 1. Load Model A (CLIP + LogReg)
    print("Loading Model A (CLIP + Logistic Regression)...")
    pipe_a = CLIPClassifierPipeline(device=DEVICE)
    pipe_a.load_clip_encoder()
    pipe_a.load_model(MODELS_DIR / "clip_logistic_regression.joblib")

    # 2. Load Model B (CLIP + SVM)
    print("Loading Model B (CLIP + SVM RBF)...")
    pipe_b = CLIPClassifierPipeline(device=DEVICE)
    pipe_b.clip_model = pipe_a.clip_model
    pipe_b.clip_preprocess = pipe_a.clip_preprocess
    pipe_b.load_model(MODELS_DIR / "clip_svm.joblib")

    # 3. Load Model C (EfficientNet-B0)
    print("Loading Model C (EfficientNet-B0)...")
    eff_model = load_efficientnet_model(EFFICIENTNET_BEST_MODEL_PATH, device=DEVICE)
    eff_model.eval()
    eff_tf = get_efficientnet_transforms(is_training=False)

    # Select representative samples: 2 samples per class (18 total)
    selected_samples = []
    np.random.seed(42)
    for c_idx in range(len(NAVARASA_CLASSES)):
        c_rows = test_df[test_df["class_idx"] == c_idx]
        chosen = c_rows.sample(n=min(2, len(c_rows)), random_state=42)
        for _, row in chosen.iterrows():
            selected_samples.append(row)

    print(f"\nEvaluating {len(selected_samples)} real test samples across all 9 classes...\n")

    test_results = []
    hasya_bhayanaka_correct = {"Model A": 0, "Model B": 0, "Model C": 0, "total": 0}

    for idx, row in enumerate(selected_samples, 1):
        crop_path = Path(row["crop_absolute_path"])
        ground_truth = str(row["class_name"])

        if not crop_path.exists():
            continue

        pil_img = Image.open(crop_path).convert("RGB")

        # Model A prediction
        pred_a, conf_a, probs_a, emb = pipe_a.predict(pil_img)
        sorted_a = sorted(probs_a.items(), key=lambda x: x[1], reverse=True)[:3]

        # Model B prediction (using cached embedding)
        emb_2d = emb.reshape(1, -1)
        probs_b_arr = pipe_b.classifier.predict_proba(emb_2d)[0]
        probs_b = {cls: float(probs_b_arr[i]) for i, cls in enumerate(NAVARASA_CLASSES)}
        sorted_b = sorted(probs_b.items(), key=lambda x: x[1], reverse=True)[:3]
        pred_b = sorted_b[0][0]
        conf_b = sorted_b[0][1]

        # Model C prediction
        eff_tensor = eff_tf(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits_c = eff_model(eff_tensor)
            probs_c_arr = F.softmax(logits_c, dim=1).cpu().numpy()[0]
        probs_c = {cls: float(probs_c_arr[i]) for i, cls in enumerate(NAVARASA_CLASSES)}
        sorted_c = sorted(probs_c.items(), key=lambda x: x[1], reverse=True)[:3]
        pred_c = sorted_c[0][0]
        conf_c = sorted_c[0][1]

        # Check Hasya vs Bhayanaka resolution
        if ground_truth in ["Hasya", "Bhayanaka"]:
            hasya_bhayanaka_correct["total"] += 1
            if pred_a == ground_truth:
                hasya_bhayanaka_correct["Model A"] += 1
            if pred_b == ground_truth:
                hasya_bhayanaka_correct["Model B"] += 1
            if pred_c == ground_truth:
                hasya_bhayanaka_correct["Model C"] += 1

        match_a = "[CORRECT]" if pred_a == ground_truth else f"[MISSED: {pred_a}]"
        match_b = "[CORRECT]" if pred_b == ground_truth else f"[MISSED: {pred_b}]"
        match_c = "[CORRECT]" if pred_c == ground_truth else f"[MISSED: {pred_c}]"

        print(f"Sample {idx:02d}: True Class: {ground_truth:<10} | File: {crop_path.name}")
        print(f"  Model A (CLIP+LogReg): {pred_a} ({conf_a*100:.1f}%) {match_a} | Top3: {[f'{k}:{v*100:.1f}%' for k,v in sorted_a]}")
        print(f"  Model B (CLIP+SVM):    {pred_b} ({conf_b*100:.1f}%) {match_b} | Top3: {[f'{k}:{v*100:.1f}%' for k,v in sorted_b]}")
        print(f"  Model C (EffNet-B0):   {pred_c} ({conf_c*100:.1f}%) {match_c} | Top3: {[f'{k}:{v*100:.1f}%' for k,v in sorted_c]}")
        print("-" * 70)

        test_results.append({
            "image_filename": crop_path.name,
            "ground_truth": ground_truth,
            "model_a": {
                "prediction": pred_a,
                "confidence": conf_a,
                "is_correct": pred_a == ground_truth,
                "top3": sorted_a,
            },
            "model_b": {
                "prediction": pred_b,
                "confidence": conf_b,
                "is_correct": pred_b == ground_truth,
                "top3": sorted_b,
            },
            "model_c": {
                "prediction": pred_c,
                "confidence": conf_c,
                "is_correct": pred_c == ground_truth,
                "top3": sorted_c,
            },
        })

    print("\n" + "=" * 70)
    print(" HASYA VS BHAYANAKA RESOLUTION SUMMARY ")
    print("=" * 70)
    tot = hasya_bhayanaka_correct["total"]
    for m in ["Model A", "Model B", "Model C"]:
        corr = hasya_bhayanaka_correct[m]
        print(f"  {m}: {corr}/{tot} correct ({corr/tot*100:.1f}%) on Hasya & Bhayanaka test samples")

    out_json = REPORTS_DIR / "real_world_test_results.json"
    with open(out_json, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nSaved real-world prediction test results to {out_json}")
    return test_results


if __name__ == "__main__":
    run_real_prediction_tests()
