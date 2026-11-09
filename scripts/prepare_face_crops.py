"""
Phase 3 Preprocessing: Standardized Face Crop Dataset Generation.
Processes all images in train, val, and test splits through HumanFaceDetector,
validates single human faces, extracts proportional crops (padding=0.22),
and saves normalized 224x224 crops to data/face_crops/ for end-to-end alignment.
"""

import os
import sys
import time
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    BASE_DIR,
    DATA_DIR,
    SPLITS_DIR,
    PROCESSED_DATA_DIR,
    NAVARASA_CLASSES,
    CLASS_TO_IDX,
)
from src.face_detection.face_validation import validate_face_image

FACE_CROPS_DIR = DATA_DIR / "face_crops"


def generate_face_crops():
    print("=" * 70)
    print("PHASE 3: STANDARDIZED FACIAL CROP DATASET GENERATION")
    print("=" * 70)

    FACE_CROPS_DIR.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    stats = {}

    for split in splits:
        csv_path = SPLITS_DIR / f"{split}.csv"
        df = pd.read_csv(csv_path)
        print(f"\nProcessing {split} split ({len(df)} images)...")

        split_out_dir = FACE_CROPS_DIR / split
        split_out_dir.mkdir(parents=True, exist_ok=True)
        for cls in NAVARASA_CLASSES:
            (split_out_dir / cls).mkdir(parents=True, exist_ok=True)

        valid_records = []
        rejected_records = []

        t0 = time.time()
        for idx, row in df.iterrows():
            rel_path = row["filepath"]
            full_path = BASE_DIR / rel_path
            if not full_path.exists() and "absolute_path" in row:
                full_path = Path(row["absolute_path"])

            cls_name = row["class_name"]

            try:
                img = Image.open(full_path).convert("RGB")
            except Exception as e:
                rejected_records.append({"path": str(full_path), "reason": f"unreadable: {e}"})
                continue

            res = validate_face_image(img, padding_ratio=0.22)

            if res.is_valid and res.face_crop is not None:
                # Resize crop to 224x224
                crop_resized = res.face_crop.resize((224, 224), Image.Resampling.LANCZOS)
                
                crop_rel = f"face_crops/{split}/{cls_name}/{full_path.name}"
                crop_save_path = DATA_DIR / crop_rel
                crop_resized.save(crop_save_path, quality=95)

                valid_records.append({
                    "crop_filepath": str(crop_rel).replace("\\", "/"),
                    "crop_absolute_path": str(crop_save_path).replace("\\", "/"),
                    "original_filepath": row["filepath"],
                    "class_name": cls_name,
                    "class_idx": row["class_idx"],
                    "detection_conf": round(res.confidence, 4),
                    "face_area_pct": round(res.face_area_pct, 2),
                    "crop_bbox": str(res.crop_bbox),
                })
            else:
                rejected_records.append({
                    "path": str(full_path),
                    "status": res.status.value,
                    "face_count": res.face_count,
                    "class_name": cls_name
                })

            if (idx + 1) % 500 == 0 or (idx + 1) == len(df):
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                print(f"  [{split.upper()}] Processed {idx + 1}/{len(df)} images ({rate:.1f} img/s) — Valid crops: {len(valid_records)}")

        # Save crops CSV
        crops_df = pd.DataFrame(valid_records)
        out_csv = SPLITS_DIR / f"{split}_crops.csv"
        crops_df.to_csv(out_csv, index=False)
        print(f"Saved {len(crops_df)} validated face crops to: {out_csv}")

        stats[split] = {
            "total_input": len(df),
            "valid_crops": len(crops_df),
            "rejected_count": len(rejected_records),
            "retention_rate": round(len(crops_df) / len(df) * 100, 2),
        }

    summary_file = SPLITS_DIR / "face_crops_summary.json"
    import json
    with open(summary_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nFace crops extraction summary saved to: {summary_file}")
    print("\n>>> PHASE 3 PREPROCESSING COMPLETE! ALL FACIAL CROPS READY FOR MODEL TRAINING. <<<")
    return stats


if __name__ == "__main__":
    generate_face_crops()
