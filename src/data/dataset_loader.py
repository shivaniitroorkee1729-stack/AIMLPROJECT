"""
Dataset loader for Navarasa facial expression classification.
"""

import sys
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import BASE_DIR, SPLITS_DIR
from src.data.preprocessing import get_efficientnet_transforms, get_clip_transforms


class NavarasaDataset(Dataset):
    """
    PyTorch Dataset for Navarasa images indexed via split CSV files.
    """

    def __init__(self, csv_file, transform=None, base_dir=BASE_DIR):
        self.df = pd.read_csv(csv_file) if isinstance(csv_file, (str, Path)) else csv_file
        self.transform = transform
        self.base_dir = Path(base_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if "crop_filepath" in row:
            rel_path = row["crop_filepath"]
            full_path = self.base_dir / "data" / rel_path
            if not full_path.exists() and "crop_absolute_path" in row:
                full_path = Path(row["crop_absolute_path"])
        else:
            rel_path = row["filepath"]
            full_path = self.base_dir / rel_path
            if not full_path.exists() and "absolute_path" in row:
                full_path = Path(row["absolute_path"])

        try:
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {full_path}: {e}")

        if self.transform is not None:
            image = self.transform(image)

        label = int(row["class_idx"])
        return image, label, str(row["class_name"]), str(full_path)


def get_dataloaders(
    batch_size=32,
    num_workers=0,  # default to 0 for Windows compatibility
    train_transform=None,
    val_transform=None,
    test_transform=None,
    use_crops=True,
):
    """
    Constructs PyTorch DataLoaders for train, val, and test splits.
    """
    train_csv = SPLITS_DIR / ("train_crops.csv" if use_crops and (SPLITS_DIR / "train_crops.csv").exists() else "train.csv")
    val_csv = SPLITS_DIR / ("val_crops.csv" if use_crops and (SPLITS_DIR / "val_crops.csv").exists() else "val.csv")
    test_csv = SPLITS_DIR / ("test_crops.csv" if use_crops and (SPLITS_DIR / "test_crops.csv").exists() else "test.csv")

    if train_transform is None:
        train_transform = get_efficientnet_transforms(is_training=True)
    if val_transform is None:
        val_transform = get_efficientnet_transforms(is_training=False)
    if test_transform is None:
        test_transform = get_efficientnet_transforms(is_training=False)

    train_ds = NavarasaDataset(train_csv, transform=train_transform)
    val_ds = NavarasaDataset(val_csv, transform=val_transform)
    test_ds = NavarasaDataset(test_csv, transform=test_transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader
