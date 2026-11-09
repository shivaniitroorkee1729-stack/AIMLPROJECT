"""
Helper utilities for device handling, image processing, formatting, and I/O.
"""

import sys
import base64
import io
from pathlib import Path
from PIL import Image
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import DEVICE, NAVARASA_CLASSES, RASA_DESCRIPTIONS


def get_system_device_info():
    """Returns formatted hardware and acceleration info."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Standard Host)"
    return {
        "device": str(DEVICE),
        "cuda_available": cuda_available,
        "device_name": device_name,
        "torch_version": torch.__version__,
    }


def pil_to_base64(image, format="JPEG"):
    """Encodes a PIL image to a base64 string for HTML embedding."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def format_percentage(value, decimals=1):
    """Formats float (0.0 - 1.0 or 0 - 100) as percentage string."""
    if value <= 1.0:
        value = value * 100
    return f"{value:.{decimals}f}%"
