"""
NAVARASA AI — Explainable Facial Expression Intelligence
Streamlit Main Application Entry Point
"""

import sys
from pathlib import Path
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent))
from config.settings import NAVARASA_CLASSES, RASA_DESCRIPTIONS, PLOTS_DIR
from src.utils.visualization import get_custom_css, THEME
from src.utils.helpers import get_system_device_info

st.set_page_config(
    page_title="NAVARASA AI — Explainable Facial Expression Intelligence",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom Indian classical theme CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎭 NAVARASA AI")
    st.markdown(
        "<p style='color: #D8E2DC; font-size: 0.9rem;'>"
        "Explainable Artificial Intelligence for Indian Classical Facial Expressions</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    dev_info = get_system_device_info()
    st.markdown(
        f"""
        <div style='background: rgba(255,255,255,0.08); padding: 12px; border-radius: 8px; font-size: 0.85rem;'>
            <b>Compute Engine:</b> {dev_info['device']}<br>
            <b>Hardware:</b> {dev_info['device_name']}<br>
            <b>PyTorch:</b> {dev_info['torch_version']}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size: 0.8rem; color: #A0AEC0; text-align: center;'>"
        "Designed for Indian Classical Aesthetics & Modern Explainable AI Research"
        "</div>",
        unsafe_allow_html=True,
    )

# Header Banner
st.markdown(
    f"""
    <div class="navarasa-header-card">
        <span class="badge-gold">RESEARCH & DEPLOYMENT PLATFORM</span>
        <h1 style="margin-top: 12px; font-size: 2.3rem;">NAVARASA AI</h1>
        <p style="font-size: 1.15rem; opacity: 0.95; max-width: 900px; line-height: 1.6;">
            A dual-pipeline deep learning and explainable artificial intelligence system
            trained to identify the nine primary aesthetic sentiments (Navarasas) of Bharata Muni's 
            <em>Natya Shastra</em>, coupled with exact Grad-CAM visual grounding, 6-region facial energy 
            segmentation, FACS Facial Action Unit analysis, and counterfactual reasoning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Overview Columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""
        <div class="navarasa-card" style="text-align: center;">
            <div style="font-size: 2rem;">🎭</div>
            <h3 style="margin: 8px 0;">9 Navarasas</h3>
            <p style="font-size: 0.85rem; color: {THEME['text_muted']};">Classical Indian Rasa Aesthetic Categories</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="navarasa-card" style="text-align: center;">
            <div style="font-size: 2rem;">🧠</div>
            <h3 style="margin: 8px 0;">Dual Pipelines</h3>
            <p style="font-size: 0.85rem; color: {THEME['text_muted']};">CLIP ViT-B/32 & EfficientNet-B0</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""
        <div class="navarasa-card" style="text-align: center;">
            <div style="font-size: 2rem;">🔍</div>
            <h3 style="margin: 8px 0;">Grad-CAM & XAI</h3>
            <p style="font-size: 0.85rem; color: {THEME['text_muted']};">Gradient-weighted Visual Grounding</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"""
        <div class="navarasa-card" style="text-align: center;">
            <div style="font-size: 2rem;">⚡</div>
            <h3 style="margin: 8px 0;">FACS & Mesh</h3>
            <p style="font-size: 0.85rem; color: {THEME['text_muted']};">Facial Action Units & 6-Region Masks</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# The Nine Navarasas Showcase
st.markdown("## The Nine Primary Navarasas")
st.markdown(
    "In Sanskrit aesthetics (*Rasa Siddhanta*), each expression represents a universal emotional state (*Sthayi Bhaava*) "
    "manifested through distinct facial musculature, gaze, and posture."
)

grid_cols = st.columns(3)
for idx, cls in enumerate(NAVARASA_CLASSES):
    meta = RASA_DESCRIPTIONS[cls]
    col = grid_cols[idx % 3]
    with col:
        st.markdown(
            f"""
            <div class="navarasa-card" style="border-top: 4px solid {meta['color']}; min-height: 230px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.25rem; font-weight: 700; color: {THEME['deep_green']};">
                        {idx + 1}. {cls}
                    </span>
                    <span style="font-size: 0.8rem; background: #EDF2F7; padding: 2px 8px; border-radius: 12px; font-weight: 600;">
                        {meta['bhaava'].split(' ')[0]}
                    </span>
                </div>
                <div style="font-size: 0.9rem; font-weight: 600; color: {THEME['forest_green']}; margin: 6px 0;">
                    {meta['bhaava']}
                </div>
                <p style="font-size: 0.85rem; color: {THEME['text_muted']}; margin-bottom: 8px; line-height: 1.4;">
                    {meta['meaning']}
                </p>
                <div style="font-size: 0.8rem; background: {THEME['warm_ivory']}; padding: 6px 10px; border-radius: 6px; border: 1px dashed {THEME['border_gold']};">
                    <b>Cues:</b> {meta['visual_cues']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# Machine Learning Pipelines Overview
st.markdown("## Dual Machine Learning Architecture")
pcol1, pcol2 = st.columns(2)

with pcol1:
    st.markdown(
        f"""
        <div class="navarasa-card">
            <span class="badge-green">PIPELINE A</span>
            <h3 style="margin-top: 8px;">CLIP ViT-B/32 + Logistic Regression</h3>
            <p style="font-size: 0.9rem; color: {THEME['text_muted']}; line-height: 1.5;">
                Leverages OpenAI's Vision Transformer (ViT-B/32) vision encoder pretrained on 400M image-text pairs.
                The visual backbone extracts high-level semantic representations (512-dimensional embeddings),
                which are classified into nine Navarasa categories using an L2-regularized Logistic Regression classifier.
            </p>
            <div style="background: {THEME['warm_ivory']}; padding: 10px; border-radius: 8px; font-size: 0.85rem; font-family: monospace;">
                Image (224x224) ➔ Frozen ViT-B/32 ➔ 512-dim Embedding ➔ LogReg ➔ 9 Classes
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with pcol2:
    st.markdown(
        f"""
        <div class="navarasa-card">
            <span class="badge-gold">PIPELINE B (PRIMARY XAI)</span>
            <h3 style="margin-top: 8px;">EfficientNet-B0 (Two-Stage Fine-Tuned)</h3>
            <p style="font-size: 0.9rem; color: {THEME['text_muted']}; line-height: 1.5;">
                A convolutional neural network optimized via compound scaling. Trained using a two-stage strategy:
                Phase 1 trains the custom 9-class head with a frozen backbone, followed by Phase 2 fine-tuning top 
                residual inverted bottleneck blocks. Powers exact gradient-weighted activation mapping (Grad-CAM).
            </p>
            <div style="background: {THEME['warm_ivory']}; padding: 10px; border-radius: 8px; font-size: 0.85rem; font-family: monospace;">
                Image (224x224) ➔ EfficientNet-B0 ➔ Dropout (0.4) ➔ Linear Head ➔ Grad-CAM & 9 Classes
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Representative Dataset Sample Grid if available
grid_sample = PLOTS_DIR / "sample_grid_all_classes.png"
if grid_sample.exists():
    st.markdown("---")
    st.markdown("### Representative Expressions from New Dataset")
    st.image(str(grid_sample), use_container_width=True)
