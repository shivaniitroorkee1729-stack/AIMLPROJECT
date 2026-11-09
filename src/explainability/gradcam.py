"""
Explainable AI — Gradient-weighted Class Activation Mapping (Grad-CAM).
Calculates exact class-specific Grad-CAM heatmaps from feature activations
and gradients of the fine-tuned EfficientNet-B0 model.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import NAVARASA_CLASSES, DEVICE


class GradCAM:
    """
    Grad-CAM engine for NavarasaEfficientNet.
    """

    def __init__(self, model, device=DEVICE):
        self.model = model
        self.model.eval()
        self.device = device
        self.classes = NAVARASA_CLASSES

    def generate_heatmap(self, input_tensor, target_class_idx=None):
        """
        Generates class-specific Grad-CAM heatmap for a single image tensor [1, 3, 224, 224].

        Args:
            input_tensor: torch.Tensor of shape [1, 3, H, W]
            target_class_idx: int, target class index. If None, uses top predicted class.

        Returns:
            heatmap: 2D numpy array [H, W] normalized to [0, 1]
            predicted_idx: int
            confidence: float
        """
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        self.model.zero_grad()
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        confidence = float(probs[0, target_class_idx].item())

        # Target class score
        score = logits[0, target_class_idx]
        score.backward(retain_graph=True)

        # Gradients and activations from hooks
        gradients = self.model.gradients  # [1, C, H_feat, W_feat]
        activations = self.model.activations  # [1, C, H_feat, W_feat]

        if gradients is None or activations is None:
            raise RuntimeError("Gradients or activations not captured by Grad-CAM hooks.")

        # Global average pooling of gradients along spatial dimensions -> alpha weights [1, C, 1, 1]
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, H_feat, W_feat]

        # Apply ReLU to retain only positive influence
        cam = F.relu(cam)

        # Upsample to input tensor resolution
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False
        )

        cam_np = cam.squeeze().detach().cpu().numpy()

        # Normalize between 0 and 1
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max - cam_min > 1e-8:
            heatmap = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam_np)

        return heatmap, target_class_idx, confidence

    def overlay_heatmap(self, original_image, heatmap, alpha=0.55, colormap=cv2.COLORMAP_JET):
        """
        Overlays Grad-CAM heatmap onto the original RGB image.

        Args:
            original_image: PIL Image or np.ndarray [H, W, 3] (RGB)
            heatmap: 2D np.ndarray [H, W] (normalized to [0, 1])
            alpha: float, transparency factor
            colormap: OpenCV colormap constant

        Returns:
            overlay: np.ndarray [H, W, 3] in RGB format (uint8)
        """
        if isinstance(original_image, Image.Image):
            rgb_img = np.array(original_image.convert("RGB"))
        else:
            rgb_img = original_image.copy()

        h, w = rgb_img.shape[:2]

        # Resize heatmap to original image dimensions if different
        if heatmap.shape[:2] != (h, w):
            resized_cam = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            resized_cam = heatmap

        # Convert heatmap to uint8 colormap
        heatmap_uint8 = np.uint8(255 * resized_cam)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        # Convert OpenCV BGR to RGB
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Alpha blend overlay: image * (1 - alpha) + heatmap * alpha
        overlay = cv2.addWeighted(rgb_img, 1.0 - alpha, heatmap_colored, alpha, 0)
        return overlay
