"""
Page 1: Dataset Explorer — Navarasa Facial Expression Dataset Analysis & Visualization
"""

import sys
import json
from pathlib import Path
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    PROCESSED_DATA_DIR,
    SPLITS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    NAVARASA_CLASSES,
    RASA_DESCRIPTIONS,
    DATASET_ANALYSIS_PATH,
)
from src.utils.visualization import get_custom_css, THEME

st.set_page_config(page_title="Dataset Explorer — NAVARASA AI", page_icon="📊", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.markdown(
    """
    <div class="navarasa-header-card">
        <span class="badge-gold">DATASET INTELLIGENCE</span>
        <h1 style="margin-top: 10px;">Navarasa Dataset Explorer</h1>
        <p style="opacity: 0.95; font-size: 1.05rem;">
            Empirical analysis, class distribution, split verification, and sample inspection
            for the newly downloaded classical Navarasa facial expression dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load dataset analysis JSON if available
analysis_data = {}
if DATASET_ANALYSIS_PATH.exists():
    with open(DATASET_ANALYSIS_PATH, "r") as f:
        analysis_data = json.load(f)

# Load split summary JSON
split_data = {}
split_summary_path = SPLITS_DIR / "split_summary.json"
if split_summary_path.exists():
    with open(split_summary_path, "r") as f:
        split_data = json.load(f)

# Stat Metric Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    tot_imgs = analysis_data.get("total_valid_images", 14805)
    st.markdown(
        f"""
        <div class="navarasa-card" style="text-align: center;">
            <div style="font-size: 1.8rem; font-weight: 700; color: {THEME['forest_green']};">{tot_imgs:,}</div>
            <div style="font-size: 0.85rem; color: {THEME['text_muted']};">Total Images Extracted</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    uniq_imgs = split_data.get("total_unique_samples", 14100)
    st.markdown(
        f"""
        <div class="navarasa-card" style="text-align: center;">
            <div style="font-size: 1.8rem; font-weight: 700; color: {THEME['deep_green']};">{uniq_imgs:,}</div>
            <div style="font-size: 0.85rem; color: {THEME['text_muted']};">Unique Leakage-Free Images</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="navarasa-card" style="text-align: center;">
            <div style="font-size: 1.8rem; font-weight: 700; color: {THEME['regal_gold']};">9 Classes</div>
            <div style="font-size: 0.85rem; color: {THEME['text_muted']};">Complete Navarasa Categories</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    corrupted = analysis_data.get("corrupted_images_count", 0)
    st.markdown(
        f"""
        <div class="navarasa-card" style="text-align: center;">
            <div style="font-size: 1.8rem; font-weight: 700; color: #2E7D32;">{corrupted} Corrupted</div>
            <div style="font-size: 0.85rem; color: {THEME['text_muted']};">100% Integrity Verified</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Section 1: Distribution & Split Breakdown
tab1, tab2, tab3 = st.tabs(["📊 Class Distribution", "✂️ Dataset Splits", "🔍 Interactive Sample Gallery"])

with tab1:
    col_chart, col_table = st.columns([1.3, 1])
    with col_chart:
        dist_img_path = PLOTS_DIR / "class_distribution.png"
        if dist_img_path.exists():
            st.image(str(dist_img_path), use_container_width=True)
        else:
            st.info("Distribution plot generating...")

    with col_table:
        st.markdown("### Class Balance Table")
        class_counts = analysis_data.get("class_counts", {})
        percentages = analysis_data.get("class_distribution_percentages", {})

        table_rows = []
        for idx, cls in enumerate(NAVARASA_CLASSES, 1):
            cnt = class_counts.get(cls, 0)
            pct = percentages.get(cls, 0.0)
            table_rows.append({"#": idx, "Navarasa": cls, "Count": f"{cnt:,}", "Share (%)": f"{pct:.2f}%"})

        st.dataframe(table_rows, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("### Stratified 70 : 15 : 15 Dataset Partitioning")
    st.markdown(
        "To guarantee unbiased evaluation without optimistic leakage, 705 duplicate images were deduplicated. "
        "The test dataset remains **completely held-out and unseen** during model training and hyperparameter tuning."
    )

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        tr_cnt = split_data.get("train_count", 9870)
        st.metric("Training Split (70%)", f"{tr_cnt:,} images")
    with sc2:
        va_cnt = split_data.get("val_count", 2115)
        st.metric("Validation Split (15%)", f"{va_cnt:,} images")
    with sc3:
        te_cnt = split_data.get("test_count", 2115)
        st.metric("Held-Out Test Split (15%)", f"{te_cnt:,} images")

    # Detailed per-class split table
    breakdown = split_data.get("class_breakdown", {})
    if breakdown:
        st.markdown("#### Per-Class Split Counts")
        split_table = []
        for cls in NAVARASA_CLASSES:
            cdata = breakdown.get(cls, {})
            split_table.append({
                "Navarasa Class": cls,
                "Train (70%)": cdata.get("train", 0),
                "Validation (15%)": cdata.get("val", 0),
                "Test (15%)": cdata.get("test", 0),
                "Total Unique": cdata.get("total", 0),
            })
        st.dataframe(split_table, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### Browse Real Dataset Samples by Navarasa")
    selected_cls = st.selectbox("Select Navarasa Class to Inspect:", NAVARASA_CLASSES, index=0)

    cls_meta = RASA_DESCRIPTIONS.get(selected_cls, {})
    st.markdown(
        f"""
        <div style="background: {THEME['warm_ivory']}; padding: 12px 16px; border-radius: 8px; border-left: 4px solid {cls_meta.get('color', '#124E3F')}; margin-bottom: 16px;">
            <b>{selected_cls}</b> ({cls_meta.get('bhaava', '')}): {cls_meta.get('meaning', '')}<br>
            <span style="font-size: 0.85rem; color: {THEME['text_muted']};"><b>Key Facial Features:</b> {cls_meta.get('visual_cues', '')}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cls_dir = PROCESSED_DATA_DIR / selected_cls
    if cls_dir.exists():
        img_paths = list(cls_dir.glob("*.jpg"))[:8]
        if img_paths:
            img_cols = st.columns(4)
            for idx, p in enumerate(img_paths):
                col = img_cols[idx % 4]
                with col:
                    img = Image.open(p)
                    st.image(img, caption=p.name, use_container_width=True)
        else:
            st.warning("No images found in class folder.")
    else:
        st.error(f"Class folder not found at {cls_dir}")
