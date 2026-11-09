"""
High-Level Inference Engine for Single and Dual-Model Prediction.
Enforces Human Face Detection and Facial Crop Extraction before all ML operations:
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
from src.face_detection import validate_face_image, FaceValidationStatus, FaceValidationResult


def predict_clip(image_input, enforce_face_crop: bool = True):
    """
    Runs prediction using CLIP + Logistic Regression pipeline.
    Optionally enforces face validation and uses the extracted facial crop.
    """
    validation_res = None
    if enforce_face_crop:
        validation_res = validate_face_image(image_input)
        if not validation_res.is_valid:
            raise ValueError(validation_res.message)
        img_to_process = validation_res.face_crop
    else:
        img_to_process = image_input

    pipeline = get_clip_pipeline()
    pred_class, confidence, probabilities, embedding = pipeline.predict(img_to_process)

    # Sort probabilities descending
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top_3 = sorted_probs[:3]

    result = {
        "model_name": "CLIP ViT-B/32 + Logistic Regression",
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "top_3": top_3,
        "embedding": embedding,
        "description": RASA_DESCRIPTIONS.get(pred_class, {})
    }
    if validation_res is not None:
        result["face_validation"] = validation_res
        result["face_crop"] = validation_res.face_crop

    return result


def predict_efficientnet(image_input, device=DEVICE, enforce_face_crop: bool = True):
    """
    Runs prediction using fine-tuned EfficientNet-B0.
    Optionally enforces face validation and uses the extracted facial crop.
    """
    validation_res = None
    if enforce_face_crop:
        validation_res = validate_face_image(image_input)
        if not validation_res.is_valid:
            raise ValueError(validation_res.message)
        img_to_process = validation_res.face_crop
    else:
        img_to_process = image_input

    model = get_efficientnet_model(device=device)
    transform = get_efficientnet_transforms(is_training=False)

    if isinstance(img_to_process, (str, Path)):
        img = Image.open(img_to_process).convert("RGB")
    elif isinstance(img_to_process, Image.Image):
        img = img_to_process.convert("RGB")
    elif isinstance(img_to_process, np.ndarray):
        img = Image.fromarray(img_to_process).convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(img_to_process)}")

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

    result = {
        "model_name": "EfficientNet-B0 (Fine-Tuned)",
        "predicted_class": pred_class,
        "predicted_idx": pred_idx,
        "confidence": confidence,
        "probabilities": probabilities,
        "top_3": top_3,
        "input_tensor": tensor,
        "description": RASA_DESCRIPTIONS.get(pred_class, {})
    }
    if validation_res is not None:
        result["face_validation"] = validation_res
        result["face_crop"] = validation_res.face_crop

    return result


def predict_dual_models(image_input, device=DEVICE, enforce_face_crop: bool = True):
    """
    Runs simultaneous inference on both CLIP and EfficientNet-B0.
    Calculates agreement, confidence differentials, and ranking differences.
    Both models share the exact same validated facial crop.
    """
    validation_res = None
    if enforce_face_crop:
        validation_res = validate_face_image(image_input)
        if not validation_res.is_valid:
            raise ValueError(validation_res.message)
        target_input = validation_res.face_crop
    else:
        target_input = image_input

    res_clip = predict_clip(target_input, enforce_face_crop=False)
    res_eff = predict_efficientnet(target_input, device=device, enforce_face_crop=False)

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

    result = {
        "clip": res_clip,
        "efficientnet": res_eff,
        "agreement": agreement,
        "agreement_label": "Models Agree" if agreement else "Models Disagree",
        "predicted_clip": pred_clip,
        "predicted_efficientnet": pred_eff,
        "confidence_difference": conf_diff,
        "probability_differences": prob_diffs,
    }
    if validation_res is not None:
        result["face_validation"] = validation_res
        result["face_crop"] = validation_res.face_crop

    return result
