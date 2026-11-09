"""
EfficientNet-B0 Two-Stage Training & Fine-Tuning Pipeline.
Stage 1: Feature Extraction & Classification Head Optimization (Frozen Backbone).
Stage 2: Top-Block Fine-Tuning (End-to-End Fine-Tuning with AdamW & Cosine Annealing).
Monitors Train/Val loss and accuracy, checkpoints best model, and plots curves.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    SPLITS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    EFFICIENTNET_BEST_MODEL_PATH,
    EFFICIENTNET_METRICS_PATH,
    NAVARASA_CLASSES,
    DEVICE,
)
from src.data.dataset_loader import get_dataloaders, NavarasaDataset
from src.data.preprocessing import get_efficientnet_transforms
from src.models.efficientnet_model import NavarasaEfficientNet


def extract_backbone_features(model, dataloader, device):
    """Extracts 1280-dim pooled features from frozen EfficientNet backbone."""
    model.eval()
    all_feats = []
    all_labels = []

    print(f"Extracting EfficientNet-B0 backbone features for {len(dataloader.dataset)} images...")
    with torch.no_grad():
        for b_idx, (imgs, lbls, _, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            feats = model.features(imgs)
            pooled = model.avgpool(feats)
            flattened = torch.flatten(pooled, 1)
            all_feats.append(flattened.cpu())
            all_labels.append(lbls)

    X = torch.cat(all_feats, dim=0)
    y = torch.cat(all_labels, dim=0)
    return X, y


def train_head_stage(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=64, lr=1e-3):
    """Stage 1: Trains classification head on extracted features."""
    print("\n>>> STAGE 1: Training Classification Head on Backbone Features <<<")
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.classifier.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "phases": []}
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for feats, labels in train_loader:
            feats = feats.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model.classifier(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * feats.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        tr_loss = running_loss / total
        tr_acc = correct / total

        # Validation
        model.classifier.eval()
        v_loss = 0.0
        v_corr = 0
        v_tot = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model.classifier(feats)
                loss = criterion(outputs, labels)
                v_loss += loss.item() * feats.size(0)
                _, preds = torch.max(outputs, 1)
                v_corr += (preds == labels).sum().item()
                v_tot += labels.size(0)

        va_loss = v_loss / v_tot
        va_acc = v_corr / v_tot

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["phases"].append("Stage 1 (Head)")

        print(f"Stage 1 - Epoch [{epoch:02d}/{epochs:02d}]: "
              f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc * 100:.2f}% | "
              f"Val Loss: {va_loss:.4f}, Val Acc: {va_acc * 100:.2f}%")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), EFFICIENTNET_BEST_MODEL_PATH)

    print(f"Stage 1 Best Validation Accuracy: {best_acc * 100:.2f}%")
    return history, best_acc


def finetune_stage(model, train_loader, val_loader, history, epochs=3, lr=1e-4):
    """Stage 2: Unfreezes top convolutional block for end-to-end fine-tuning."""
    print("\n>>> STAGE 2: Fine-Tuning Top Convolutional Block (features[-1]) <<<")
    model.unfreeze_top_layers(num_blocks=1)

    optimizer = AdamW([
        {"params": [p for p in model.features[-1].parameters() if p.requires_grad], "lr": lr},
        {"params": model.classifier.parameters(), "lr": lr * 3}
    ], weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_acc = max(history["val_acc"])

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for b_idx, (imgs, lbls, _, _) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            lbls = lbls.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

            if (b_idx + 1) % 40 == 0 or (b_idx + 1) == len(train_loader):
                print(f"  Batch {b_idx + 1}/{len(train_loader)} - Running Acc: {correct / total * 100:.2f}%")

        tr_loss = running_loss / total
        tr_acc = correct / total

        # Validation
        model.eval()
        v_loss = 0.0
        v_corr = 0
        v_tot = 0
        with torch.no_grad():
            for imgs, lbls, _, _ in val_loader:
                imgs = imgs.to(DEVICE)
                lbls = lbls.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
                v_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                v_corr += (preds == lbls).sum().item()
                v_tot += lbls.size(0)

        va_loss = v_loss / v_tot
        va_acc = v_corr / v_tot
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["phases"].append("Stage 2 (Fine-Tune)")

        print(f"Stage 2 - Epoch [{epoch}/{epochs}] ({elapsed:.1f}s): "
              f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc * 100:.2f}% | "
              f"Val Loss: {va_loss:.4f}, Val Acc: {va_acc * 100:.2f}%")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), EFFICIENTNET_BEST_MODEL_PATH)
            print(f"  --> Saved new best checkpoint (Val Acc: {va_acc * 100:.2f}%)")

    return history, best_acc


def run_efficientnet_pipeline():
    print("=" * 60)
    print(" PIPELINE B: EFFICIENTNET-B0 TWO-STAGE TRAINING & FINE-TUNING ")
    print("=" * 60)

    model = NavarasaEfficientNet(pretrained=True).to(DEVICE)
    model.freeze_backbone()

    train_tf_eval = get_efficientnet_transforms(is_training=False)
    val_tf = get_efficientnet_transforms(is_training=False)

    train_ds = NavarasaDataset(SPLITS_DIR / "train.csv", transform=train_tf_eval)
    val_ds = NavarasaDataset(SPLITS_DIR / "val.csv", transform=val_tf)

    train_loader_feats = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
    val_loader_feats = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # 1. Feature extraction for fast Stage 1
    t0 = time.time()
    X_train, y_train = extract_backbone_features(model, train_loader_feats, DEVICE)
    X_val, y_val = extract_backbone_features(model, val_loader_feats, DEVICE)
    print(f"Feature extraction done in {time.time() - t0:.1f}s. Shapes: Train={X_train.shape}, Val={X_val.shape}")

    # 2. Train head (Stage 1)
    history, best_acc = train_head_stage(model, X_train, y_train, X_val, y_val, epochs=12, batch_size=64, lr=1e-3)

    # 3. Fine-tune top block (Stage 2)
    # Using a stratified subset of train loader for fast fine-tuning on CPU (e.g. 150 per class)
    train_tf_aug = get_efficientnet_transforms(is_training=True)
    train_df = train_ds.df
    # Sample 120 per class for fine-tuning
    ft_indices = []
    np.random.seed(42)
    for c_idx in range(len(NAVARASA_CLASSES)):
        c_rows = train_df[train_df["class_idx"] == c_idx].index.values
        sampled = np.random.choice(c_rows, size=min(120, len(c_rows)), replace=False)
        ft_indices.extend(sampled)

    ft_df = train_df.iloc[ft_indices].reset_index(drop=True)
    ft_ds = NavarasaDataset(ft_df, transform=train_tf_aug)
    ft_loader = DataLoader(ft_ds, batch_size=32, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    history, final_best_acc = finetune_stage(model, ft_loader, val_loader, history, epochs=2, lr=8e-5)

    # Save training history JSON
    with open(EFFICIENTNET_METRICS_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {EFFICIENTNET_METRICS_PATH}")

    # Plot curves
    plot_training_curves(history, PLOTS_DIR / "efficientnet_training_curves.png")
    print(f"EfficientNet-B0 training completed! Best Validation Accuracy: {final_best_acc * 100:.2f}%")


def plot_training_curves(history, output_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1976D2", linewidth=2.2, marker="o")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#D32F2F", linewidth=2.2, marker="s")
    ax1.set_title("EfficientNet-B0: Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Accuracy", color="#388E3C", linewidth=2.2, marker="o")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Accuracy", color="#F57C00", linewidth=2.2, marker="s")
    ax2.set_title("EfficientNet-B0: Training & Validation Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved training curves plot to {output_path}")


if __name__ == "__main__":
    run_efficientnet_pipeline()
