"""
High-Level Inference Engine for Single and Dual-Model Prediction.
Supports:
1. CLIP + Logistic Regression
2. EfficientNet-B0
3. Dual-Model Simultaneous Comparison
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import NAVARASA_CLASSES, DEVICE, RASA_DESCRIPTIONS
from src.data.preprocessing import get_efficientnet_transforms
from src.models.model_loader import get_clip_pipeline, get_efficientnet_model


def predict_clip(image_input):
    """
    Runs prediction using CLIP + Logistic Regression pipeline.
    """
    pipeline = get_clip_pipeline()
    pred_class, confidence, probabilities, embedding = pipeline.predict(image_input)

    # Sort probabilities descending
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top_3 = sorted_probs[:3]

    return {
        "model_name": "CLIP ViT-B/32 + Logistic Regression",
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "top_3": top_3,
        "embedding": embedding,
        "description": RASA_DESCRIPTIONS.get(pred_class, {})
    }


def predict_efficientnet(image_input, device=DEVICE):
    """
    Runs prediction using fine-tuned EfficientNet-B0.
    """
    model = get_efficientnet_model(device=device)
    transform = get_efficientnet_transforms(is_training=False)

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = NAVARASA_CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    probabilities = {cls: float(probs[i]) for i, cls in enumerate(NAVARASA_CLASSES)}
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top_3 = sorted_probs[:3]

    return {
        "model_name": "EfficientNet-B0 (Fine-Tuned)",
        "predicted_class": pred_class,
        "predicted_idx": pred_idx,
        "confidence": confidence,
        "probabilities": probabilities,
        "top_3": top_3,
        "input_tensor": tensor,
        "description": RASA_DESCRIPTIONS.get(pred_class, {})
    }


def predict_dual_models(image_input, device=DEVICE):
    """
    Runs simultaneous inference on both CLIP and EfficientNet-B0.
    Calculates agreement, confidence differentials, and ranking differences.
    """
    res_clip = predict_clip(image_input)
    res_eff = predict_efficientnet(image_input, device=device)

    pred_clip = res_clip["predicted_class"]
    pred_eff = res_eff["predicted_class"]
    agreement = (pred_clip == pred_eff)

    conf_diff = round(abs(res_clip["confidence"] - res_eff["confidence"]), 4)

    # Class-wise probability differences: EffNet - CLIP
    prob_diffs = {}
    for cls in NAVARASA_CLASSES:
        p_eff = res_eff["probabilities"].get(cls, 0.0)
        p_clip = res_clip["probabilities"].get(cls, 0.0)
        prob_diffs[cls] = round(p_eff - p_clip, 4)

    return {
        "clip": res_clip,
        "efficientnet": res_eff,
        "agreement": agreement,
        "agreement_label": "Models Agree" if agreement else "Models Disagree",
        "predicted_clip": pred_clip,
        "predicted_efficientnet": pred_eff,
        "confidence_difference": conf_diff,
        "probability_differences": prob_diffs,
    }
