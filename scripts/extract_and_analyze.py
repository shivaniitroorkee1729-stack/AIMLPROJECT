"""
Script to extract raw Navarasa dataset, perform comprehensive data analysis,
integrity checks, duplicate detection, and generate reports and visualization plots.
"""

import os
import sys
import io
import zipfile
import hashlib
import json
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import numpy as np

# Allow truncated images to be read if repairable
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    RAW_CLASS_MAPPING,
    NAVARASA_CLASSES,
    DATASET_ANALYSIS_PATH,
)


def extract_dataset():
    """Extract nested zip files into structured class folders."""
    outer_zip = RAW_DATA_DIR / "Navaraas_new.zip"
    if not outer_zip.exists():
        raise FileNotFoundError(f"Outer dataset zip not found at {outer_zip}")

    print(f"Opening outer archive: {outer_zip}")
    extracted_summary = defaultdict(int)

    with zipfile.ZipFile(outer_zip, "r") as outer:
        for inner_zip_name in outer.namelist():
            if not inner_zip_name.lower().endswith(".zip"):
                continue

            # Determine raw class key from archive name (e.g. 'veer-20261109...')
            raw_key = inner_zip_name.split("-")[0].lower().strip()
            standard_class = RAW_CLASS_MAPPING.get(raw_key)

            if not standard_class:
                print(f"Warning: Unknown raw key '{raw_key}' in {inner_zip_name}, skipping.")
                continue

            target_class_dir = PROCESSED_DATA_DIR / standard_class
            target_class_dir.mkdir(parents=True, exist_ok=True)

            print(f"Extracting {inner_zip_name} -> {standard_class}...")
            inner_bytes = outer.read(inner_zip_name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                for member in inner.infolist():
                    if member.is_dir():
                        continue
                    fname = Path(member.filename).name
                    if not fname:
                        continue
                    
                    # Target path
                    dest_path = target_class_dir / fname
                    
                    # Extract file content
                    data = inner.read(member.filename)
                    with open(dest_path, "wb") as f_out:
                        f_out.write(data)
                    extracted_summary[standard_class] += 1

    print("\nExtraction Summary:")
    for cls in NAVARASA_CLASSES:
        print(f"  {cls}: {extracted_summary.get(cls, 0)} files")
    return extracted_summary


def analyze_dataset():
    """Perform comprehensive data analysis on extracted dataset."""
    print("\nPerforming dataset analysis, integrity verification, and duplicate check...")
    
    total_images = 0
    class_counts = defaultdict(int)
    formats = Counter()
    dimensions = []
    corrupted_images = []
    file_hashes = defaultdict(list)
    aspect_ratios = []

    for cls in NAVARASA_CLASSES:
        cls_dir = PROCESSED_DATA_DIR / cls
        if not cls_dir.exists():
            continue

        for img_path in cls_dir.glob("*"):
            if img_path.is_dir():
                continue
            
            try:
                # Check readability & integrity
                with open(img_path, "rb") as f:
                    content = f.read()
                    file_hash = hashlib.md5(content).hexdigest()
                    file_hashes[file_hash].append(str(img_path))

                with Image.open(img_path) as img:
                    img.verify()

                # Re-open for format & size inspection
                with Image.open(img_path) as img:
                    formats[img.format or "Unknown"] += 1
                    w, h = img.size
                    dimensions.append({"width": w, "height": h, "class": cls})
                    aspect_ratios.append(round(w / h, 2) if h > 0 else 1.0)
                    total_images += 1
                    class_counts[cls] += 1

            except Exception as e:
                corrupted_images.append({
                    "path": str(img_path),
                    "class": cls,
                    "error": str(e)
                })

    # Find duplicates
    duplicates = {h: paths for h, paths in file_hashes.items() if len(paths) > 1}
    num_duplicate_sets = len(duplicates)
    num_redundant_files = sum(len(paths) - 1 for paths in duplicates.values())

    widths = [d["width"] for d in dimensions]
    heights = [d["height"] for d in dimensions]

    analysis_results = {
        "total_valid_images": total_images,
        "class_counts": dict(class_counts),
        "class_distribution_percentages": {
            cls: round((class_counts[cls] / total_images) * 100, 2) if total_images else 0
            for cls in NAVARASA_CLASSES
        },
        "formats": dict(formats),
        "corrupted_images_count": len(corrupted_images),
        "corrupted_images": corrupted_images,
        "duplicate_sets_count": num_duplicate_sets,
        "redundant_files_count": num_redundant_files,
        "dimension_statistics": {
            "min_width": int(np.min(widths)) if widths else 0,
            "max_width": int(np.max(widths)) if widths else 0,
            "mean_width": float(np.mean(widths)) if widths else 0,
            "median_width": float(np.median(widths)) if widths else 0,
            "min_height": int(np.min(heights)) if heights else 0,
            "max_height": int(np.max(heights)) if heights else 0,
            "mean_height": float(np.mean(heights)) if heights else 0,
            "median_height": float(np.median(heights)) if heights else 0,
            "mean_aspect_ratio": float(np.mean(aspect_ratios)) if aspect_ratios else 0,
        }
    }

    # Save JSON report
    with open(DATASET_ANALYSIS_PATH, "w") as f:
        json.dump(analysis_results, f, indent=2)
    print(f"Saved analysis JSON to {DATASET_ANALYSIS_PATH}")

    # Generate Markdown report
    md_path = REPORTS_DIR / "dataset_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Navarasa Facial Expression Dataset Analysis Report\n\n")
        f.write(f"- **Total Valid Images:** {total_images}\n")
        f.write(f"- **Total Classes:** {len(NAVARASA_CLASSES)}\n")
        f.write(f"- **Corrupted Images:** {len(corrupted_images)}\n")
        f.write(f"- **Duplicate Image Sets:** {num_duplicate_sets} ({num_redundant_files} redundant files)\n\n")

        f.write("## 1. Class Distribution\n\n")
        f.write("| Index | Navarasa Class | Image Count | Percentage (%) |\n")
        f.write("|-------|----------------|-------------|----------------|\n")
        for i, cls in enumerate(NAVARASA_CLASSES, 1):
            cnt = class_counts.get(cls, 0)
            pct = analysis_results["class_distribution_percentages"].get(cls, 0.0)
            f.write(f"| {i} | **{cls}** | {cnt} | {pct}% |\n")

        f.write("\n## 2. Image Dimensions & Formats\n\n")
        f.write(f"- **Formats:** {dict(formats)}\n")
        stats = analysis_results["dimension_statistics"]
        f.write(f"- **Resolution Range:** {stats['min_width']}x{stats['min_height']} to {stats['max_width']}x{stats['max_height']}\n")
        f.write(f"- **Median Resolution:** {int(stats['median_width'])}x{int(stats['median_height'])}\n")
        f.write(f"- **Mean Aspect Ratio:** {stats['mean_aspect_ratio']:.2f}\n")

        f.write("\n## 3. Data Integrity & Quality Observations\n\n")
        if corrupted_images:
            f.write(f"- Found {len(corrupted_images)} unreadable images:\n")
            for ci in corrupted_images[:10]:
                f.write(f"  * `{ci['path']}`: {ci['error']}\n")
        else:
            f.write("- **Zero unreadable/corrupted files:** All extracted images passed PIL integrity validation.\n")
        f.write(f"- **Duplicate Analysis:** {num_duplicate_sets} sets of identical files detected. The stratified dataset split handles duplicates to prevent cross-split leakage.\n")

    print(f"Saved analysis Markdown report to {md_path}")
    return analysis_results


def generate_visualizations(analysis_results):
    """Generate professional visual distribution charts and sample grids."""
    print("\nGenerating visual distribution charts...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Class Distribution Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    classes = NAVARASA_CLASSES
    counts = [analysis_results["class_counts"].get(c, 0) for c in classes]
    colors = [
        "#E57373", "#81C784", "#64B5F6", "#E53935",
        "#FFB74D", "#4A148C", "#00897B", "#FFD54F", "#A1887F"
    ]
    bars = ax.bar(classes, counts, color=colors, edgecolor="#333333", linewidth=1.2, width=0.6)
    ax.set_title("Navarasa Dataset — Class Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Navarasa Class", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Images", fontsize=12, labelpad=10)
    plt.xticks(rotation=25, ha="right", fontsize=10)

    # Add count labels on bars
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    dist_plot_path = PLOTS_DIR / "class_distribution.png"
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"Saved class distribution plot to {dist_plot_path}")

    # 2. Sample Grid for all 9 classes (3x3 grid)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), dpi=300)
    axes = axes.flatten()

    for idx, cls in enumerate(NAVARASA_CLASSES):
        cls_dir = PROCESSED_DATA_DIR / cls
        img_files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.png"))
        ax = axes[idx]
        if img_files:
            # Pick a representative sample (middle index)
            sample_img_path = img_files[len(img_files) // 2]
            try:
                img = Image.open(sample_img_path).convert("RGB")
                ax.imshow(img)
                ax.set_title(f"{cls}\n({len(img_files)} images)", fontsize=11, fontweight="bold", pad=8)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, "No images", ha="center", va="center")
        ax.axis("off")

    plt.suptitle("Representative Sample Facial Expressions Across Nine Navarasas",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    grid_plot_path = PLOTS_DIR / "sample_grid_all_classes.png"
    plt.savefig(grid_plot_path, dpi=300)
    plt.close()
    print(f"Saved sample grid plot to {grid_plot_path}")


if __name__ == "__main__":
    extract_dataset()
    results = analyze_dataset()
    generate_visualizations(results)
    print("\nDataset extraction and analysis completed successfully!")
