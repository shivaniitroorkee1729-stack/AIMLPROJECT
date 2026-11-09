"""
Integration tests for inference pipelines and dual-model comparison.
"""

import sys
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import NAVARASA_CLASSES
from src.models.inference import predict_clip, predict_efficientnet, predict_dual_models


def test_clip_prediction():
    dummy_img = Image.new("RGB", (224, 224), color=(180, 120, 80))
    res = predict_clip(dummy_img)

    assert "predicted_class" in res
    assert res["predicted_class"] in NAVARASA_CLASSES
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 9
    assert len(res["top_3"]) == 3
    print(f"CLIP test passed. Prediction: {res['predicted_class']} ({res['confidence'] * 100:.1f}%)")


def test_efficientnet_prediction():
    dummy_img = Image.new("RGB", (224, 224), color=(100, 160, 200))
    res = predict_efficientnet(dummy_img)

    assert "predicted_class" in res
    assert res["predicted_class"] in NAVARASA_CLASSES
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 9
    assert len(res["top_3"]) == 3
    print(f"EfficientNet test passed. Prediction: {res['predicted_class']} ({res['confidence'] * 100:.1f}%)")


def test_dual_model_prediction():
    dummy_img = Image.new("RGB", (224, 224), color=(220, 150, 110))
    res = predict_dual_models(dummy_img)

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
