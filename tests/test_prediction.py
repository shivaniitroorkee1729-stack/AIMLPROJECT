"""
Integration tests for inference pipelines and dual-model comparison.
Uses authentic dataset face images to verify end-to-end inference.
"""

import sys
import glob
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import NAVARASA_CLASSES
from src.models.inference import predict_clip, predict_efficientnet, predict_dual_models


def get_face_img():
    samples = glob.glob("data/processed/*/*.jpg")
    if samples:
        return Image.open(samples[0]).convert("RGB")
    return Image.new("RGB", (224, 224), color=(180, 120, 80))


def test_clip_prediction():
    face_img = get_face_img()
    res = predict_clip(face_img, enforce_face_crop=True)

    assert "predicted_class" in res
    assert res["predicted_class"] in NAVARASA_CLASSES
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 9
    assert len(res["top_3"]) == 3
    print(f"CLIP test passed. Prediction: {res['predicted_class']} ({res['confidence'] * 100:.1f}%)")


def test_efficientnet_prediction():
    face_img = get_face_img()
    res = predict_efficientnet(face_img, enforce_face_crop=True)

    assert "predicted_class" in res
    assert res["predicted_class"] in NAVARASA_CLASSES
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 9
    assert len(res["top_3"]) == 3
    print(f"EfficientNet test passed. Prediction: {res['predicted_class']} ({res['confidence'] * 100:.1f}%)")


def test_dual_model_prediction():
    face_img = get_face_img()
    res = predict_dual_models(face_img, enforce_face_crop=True)

    assert "clip" in res
    assert "efficientnet" in res
    assert isinstance(res["agreement"], bool)
    assert len(res["probability_differences"]) == 9
    print(f"Dual-model test passed. Agreement: {res['agreement']}")


if __name__ == "__main__":
    test_clip_prediction()
    test_efficientnet_prediction()
    test_dual_model_prediction()
    print("All prediction tests passed!")
