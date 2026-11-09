"""
Image preprocessing and data augmentation pipelines for Navarasa classification.
Provides training augmentations (conservative, culturally and semantically sound)
and deterministic validation/inference transforms.
"""

import torch
from torchvision import transforms
from PIL import Image

# EfficientNet standard normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CLIP standard normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def get_efficientnet_transforms(is_training=False, image_size=224):
    """
    Transforms for EfficientNet-B0.
    For training:
      - Resize & RandomResizedCrop with high scale to avoid losing expression context
      - Horizontal flip (facial expressions are symmetric across horizontal plane)
      - Mild rotation (-10 to +10 degrees)
      - Color jitter on brightness and contrast only (subtle, 0.1)
    For validation / test / inference:
      - Resize to 256 then CenterCrop to 224
      - ToTensor
      - Standard ImageNet Normalization
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_clip_transforms(image_size=224):
    """
    Standard deterministic preprocessing for CLIP ViT-B/32 vision encoder.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def load_and_preprocess_image(image_input, transform=None):
    """
    Utility to load an image from path, bytes, or PIL Image,
    convert to RGB, and apply transform.
    """
    if isinstance(image_input, (str, bytes)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    if transform is not None:
        tensor = transform(img)
        return img, tensor
    return img
