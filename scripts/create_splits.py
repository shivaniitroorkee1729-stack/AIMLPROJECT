"""
Script to create clean, leakage-free, stratified train/val/test splits (70:15:15).
Ensures duplicate images are eliminated or grouped so no duplicate appears across splits.
"""

import sys
import hashlib
import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    PROCESSED_DATA_DIR,
    SPLITS_DIR,
    REPORTS_DIR,
    NAVARASA_CLASSES,
    CLASS_TO_IDX,
)


def create_splits(seed=42, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    print(f"Creating stratified splits ({train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}) with seed {seed}...")
    
    # 1. Gather all files and deduplicate by MD5 to prevent leakage
    hash_to_file = {}
    records = []
    duplicate_count = 0

    for cls in NAVARASA_CLASSES:
        cls_dir = PROCESSED_DATA_DIR / cls
        for img_path in sorted(cls_dir.glob("*.jpg")):
            with open(img_path, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
            
            if h in hash_to_file:
                duplicate_count += 1
                continue  # Skip duplicate to ensure 100% disjoint unique samples
            
            hash_to_file[h] = img_path
            # Store relative path for portability across environments/deployments
            rel_path = img_path.relative_to(PROCESSED_DATA_DIR.parent)
            records.append({
                "filepath": str(rel_path).replace("\\", "/"),
                "absolute_path": str(img_path).replace("\\", "/"),
                "class_name": cls,
                "class_idx": CLASS_TO_IDX[cls],
                "hash": h
            })

    df = pd.DataFrame(records)
    print(f"Total unique images after deduplication: {len(df)} (filtered out {duplicate_count} exact duplicates)")

    # 2. Stratified split into train (70%) and temp (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=df["class_idx"]
    )

    # 3. Split temp into val (50% of 30% = 15%) and test (50% of 30% = 15%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["class_idx"]
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save to CSV
    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"
    test_csv = SPLITS_DIR / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Saved {len(train_df)} train samples to {train_csv}")
    print(f"Saved {len(val_df)} val samples to {val_csv}")
    print(f"Saved {len(test_df)} test samples to {test_csv}")

    # Generate split summary
    summary = {
        "total_unique_samples": len(df),
        "deduplicated_removed": duplicate_count,
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "train_percentage": round(len(train_df) / len(df) * 100, 2),
        "val_percentage": round(len(val_df) / len(df) * 100, 2),
        "test_percentage": round(len(test_df) / len(df) * 100, 2),
        "class_breakdown": {}
    }

    print("\nPer-Class Breakdown across splits:")
    print(f"{'Class':<12} | {'Train':<6} | {'Val':<6} | {'Test':<6} | {'Total':<6}")
    print("-" * 46)
    for cls in NAVARASA_CLASSES:
        idx = CLASS_TO_IDX[cls]
        tr = int((train_df["class_idx"] == idx).sum())
        va = int((val_df["class_idx"] == idx).sum())
        te = int((test_df["class_idx"] == idx).sum())
        tot = tr + va + te
        summary["class_breakdown"][cls] = {"train": tr, "val": va, "test": te, "total": tot}
        print(f"{cls:<12} | {tr:<6} | {va:<6} | {te:<6} | {tot:<6}")

    summary_path = SPLITS_DIR / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved split summary to {summary_path}")


if __name__ == "__main__":
    create_splits()
