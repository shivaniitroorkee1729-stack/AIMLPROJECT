"""
Unit tests for preprocessing and image transforms.
"""

import sys
from pathlib import Path
import torch
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data.preprocessing import (
    get_efficientnet_transforms,
    get_clip_transforms,
    load_and_preprocess_image,
)


def test_efficientnet_transforms():
    train_tf = get_efficientnet_transforms(is_training=True, image_size=224)
    val_tf = get_efficientnet_transforms(is_training=False, image_size=224)

    dummy_img = Image.new("RGB", (300, 300), color=(128, 128, 128))

    tensor_train = train_tf(dummy_img)
    tensor_val = val_tf(dummy_img)

    assert tensor_train.shape == (3, 224, 224), f"Expected (3, 224, 224), got {tensor_train.shape}"
    assert tensor_val.shape == (3, 224, 224), f"Expected (3, 224, 224), got {tensor_val.shape}"
    assert isinstance(tensor_train, torch.Tensor)
    assert isinstance(tensor_val, torch.Tensor)


def test_clip_transforms():
    clip_tf = get_clip_transforms(image_size=224)
    dummy_img = Image.new("RGB", (400, 400), color=(200, 100, 50))
    tensor = clip_tf(dummy_img)

    assert tensor.shape == (3, 224, 224)
    assert isinstance(tensor, torch.Tensor)


def test_load_and_preprocess_image():
    dummy_img = Image.new("RGB", (250, 250), color=(50, 150, 250))
    img, tensor = load_and_preprocess_image(dummy_img, transform=get_efficientnet_transforms(is_training=False))

    assert isinstance(img, Image.Image)
    assert tensor.shape == (3, 224, 224)


if __name__ == "__main__":
    test_efficientnet_transforms()
    test_clip_transforms()
    test_load_and_preprocess_image()
    print("All preprocessing tests passed!")
