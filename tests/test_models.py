"""
Unit tests for model architectures and Grad-CAM integration.
"""

import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import NAVARASA_CLASSES
from src.models.efficientnet_model import NavarasaEfficientNet
from src.explainability.gradcam import GradCAM


def test_efficientnet_forward_and_head():
    model = NavarasaEfficientNet(num_classes=len(NAVARASA_CLASSES), dropout_p=0.4, pretrained=False)
    model.eval()

    dummy_batch = torch.randn(2, 3, 224, 224)
    logits = model(dummy_batch)

    assert logits.shape == (2, len(NAVARASA_CLASSES)), f"Expected shape (2, 9), got {logits.shape}"

    probs = model.predict_probabilities(dummy_batch)
    assert probs.shape == (2, len(NAVARASA_CLASSES))
    assert torch.allclose(torch.sum(probs, dim=1), torch.tensor([1.0, 1.0]), atol=1e-5)


def test_gradcam_computation():
    model = NavarasaEfficientNet(num_classes=len(NAVARASA_CLASSES), dropout_p=0.4, pretrained=False)
    model.eval()
    gradcam = GradCAM(model, device="cpu")

    dummy_input = torch.randn(1, 3, 224, 224)
    heatmap, pred_idx, conf = gradcam.generate_heatmap(dummy_input, target_class_idx=0)

    assert heatmap.shape == (224, 224), f"Expected heatmap (224, 224), got {heatmap.shape}"
    assert np_is_normalized(heatmap), "Heatmap must be normalized in range [0, 1]"


def np_is_normalized(arr):
    return (arr.min() >= 0.0) and (arr.max() <= 1.0)


if __name__ == "__main__":
    test_efficientnet_forward_and_head()
    test_gradcam_computation()
    print("All model and Grad-CAM tests passed!")
