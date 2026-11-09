"""
Occlusion-based Sensitivity Analysis.
Systematically occludes (masks or blurs) specified facial regions to measure
the model's prediction sensitivity to individual anatomical structures.
Supports both OpenCV and pure PIL fallbacks.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F

try:
    import cv2
    HAS_CV2 = True
except (ImportError, Exception):
    cv2 = None
    HAS_CV2 = False

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import NAVARASA_CLASSES, DEVICE
from src.data.preprocessing import get_efficientnet_transforms


def apply_mask_to_image(image, mask, mode="gaussian_blur"):
    """
    Applies occlusion to an image within the specified binary mask.
    Args:
        image: PIL Image or np.ndarray [H, W, 3] (RGB)
        mask: np.ndarray [H, W] binary mask
        mode: 'gaussian_blur', 'gray', or 'black'
    Returns:
        occluded_image: PIL Image
    """
    if isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
        img_np = np.array(pil_img)
    else:
        img_np = image.copy()
        pil_img = Image.fromarray(img_np)

    h, w = img_np.shape[:2]
    if mask.shape[:2] != (h, w):
        if HAS_CV2:
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = np.array(Image.fromarray(mask).resize((w, h), Image.Resampling.NEAREST))
    else:
        mask_resized = mask

    binary_mask = (mask_resized > 0)
    occluded_np = img_np.copy()

    if mode == "gaussian_blur":
        if HAS_CV2:
            blurred = cv2.GaussianBlur(img_np, (35, 35), 15)
        else:
            blurred_pil = pil_img.filter(ImageFilter.GaussianBlur(radius=15))
            blurred = np.array(blurred_pil)
        occluded_np[binary_mask] = blurred[binary_mask]
    elif mode == "gray":
        occluded_np[binary_mask] = [128, 128, 128]
    elif mode == "black":
        occluded_np[binary_mask] = [0, 0, 0]
    else:
        raise ValueError(f"Unknown occlusion mode: {mode}")

    return Image.fromarray(occluded_np)


class OcclusionSensitivityAnalyzer:
    """
    Executes systematic region occlusion sensitivity tests on EfficientNet-B0.
    """

    def __init__(self, model, device=DEVICE):
        self.model = model
        self.model.eval()
        self.device = device
        self.transform = get_efficientnet_transforms(is_training=False)
        self.classes = NAVARASA_CLASSES

    def run_sensitivity_analysis(self, original_image, region_masks, mode="gaussian_blur"):
        """
        Occludes each region in region_masks and tracks prediction shift.
        """
        import gc

        # 1. Base inference on original unmasked image
        orig_tensor = self.transform(original_image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            orig_logits = self.model(orig_tensor)
            orig_probs = F.softmax(orig_logits, dim=1)[0].cpu().numpy()

        del orig_tensor, orig_logits

        orig_class_idx = int(np.argmax(orig_probs))
        orig_class = self.classes[orig_class_idx]
        orig_conf = float(orig_probs[orig_class_idx])

        original_result = {
            "predicted_class": orig_class,
            "confidence": orig_conf,
            "probabilities": {cls: float(orig_probs[i]) for i, cls in enumerate(self.classes)}
        }

        # 2. Iterate each facial region sequentially
        region_results = {}
        occluded_images = {}
        max_drop = -1.0
        most_sensitive = None

        for region_name, mask in region_masks.items():
            occluded_img = apply_mask_to_image(original_image, mask, mode=mode)
            # Store memory-optimized thumbnail copy for display preview
            preview = occluded_img.copy()
            if preview.width > 256 or preview.height > 256:
                preview.thumbnail((256, 256))
            occluded_images[region_name] = preview

            occ_tensor = self.transform(occluded_img).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                occ_logits = self.model(occ_tensor)
                occ_probs = F.softmax(occ_logits, dim=1)[0].cpu().numpy()

            del occ_tensor, occ_logits, occluded_img

            occ_class_idx = int(np.argmax(occ_probs))
            occ_class = self.classes[occ_class_idx]
            occ_conf = float(occ_probs[occ_class_idx])

            prob_drop = float(orig_conf - occ_probs[orig_class_idx])
            prob_change_pct = float((prob_drop / orig_conf) * 100.0) if orig_conf > 0 else 0.0

            if prob_drop > max_drop:
                max_drop = prob_drop
                most_sensitive = region_name

            region_results[region_name] = {
                "region": region_name,
                "masked_predicted_class": occ_class,
                "masked_confidence": occ_conf,
                "target_class_original_prob": float(orig_probs[orig_class_idx]),
                "target_class_masked_prob": float(occ_probs[orig_class_idx]),
                "confidence_drop": round(prob_drop, 4),
                "confidence_drop_percentage": round(prob_change_pct, 2),
                "class_flipped": occ_class != orig_class,
                "probabilities": {cls: float(occ_probs[i]) for i, cls in enumerate(self.classes)}
            }

        return {
            "original": original_result,
            "regions": region_results,
            "occluded_images": occluded_images,
            "most_sensitive_region": most_sensitive,
            "max_confidence_drop": round(max_drop, 4)
        }
