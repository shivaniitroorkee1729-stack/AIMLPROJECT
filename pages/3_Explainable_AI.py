"""
Page 3: Explainable AI — Exact Grad-CAM Visual Grounding & Facial Region Energy Analysis
"""

import sys
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import NAVARASA_CLASSES, RASA_DESCRIPTIONS, PROCESSED_DATA_DIR, CLASS_TO_IDX
from src.utils.visualization import (
    get_custom_css,
    THEME,
    plot_region_importance_chart,
)
from src.models.model_loader import (
    get_efficientnet_model,
    get_gradcam_engine,
    get_facial_analyzer,
    get_natural_language_explainer,
    get_au_extractor,
)
from src.data.preprocessing import get_efficientnet_transforms

st.set_page_config(page_title="Explainable AI — NAVARASA AI", page_icon="🔍", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.markdown(
    """
    <div class="navarasa-header-card">
        <span class="badge-gold">VISUAL GROUNDING & EXPLAINABILITY</span>
        <h1 style="margin-top: 10px;">Grad-CAM & Facial Region Importance</h1>
        <p style="opacity: 0.95; font-size: 1.05rem;">
            Calculates exact gradient-weighted class activations from the final convolutional stage of EfficientNet-B0,
            overlaying neural attention on the facial anatomy and computing proportional energy across six anatomical regions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("### 🖼️ Sample Quick Select")
    quick_sample = st.selectbox(
        "Load an authentic test image:",
        ["None"] + NAVARASA_CLASSES,
        index=4  # Default to Veera
    )
    st.markdown("---")
    st.markdown("### 🎛️ Visualization Parameters")
    alpha_slider = st.slider("Heatmap Overlay Opacity (Alpha):", 0.1, 0.9, 0.55, 0.05)
    colormap_choice = st.selectbox("Colormap:", ["JET (Classic Blue-to-Red)", "INFERNO (High Contrast)"])

col_upload, col_ctrl = st.columns([1.2, 1])
input_image = None

with col_upload:
    uploaded = st.file_uploader("Upload Facial Image (JPG / PNG):", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        input_image = Image.open(uploaded).convert("RGB")
    elif quick_sample != "None":
        cls_dir = PROCESSED_DATA_DIR / quick_sample
        samples = list(cls_dir.glob("*.jpg"))
        if samples:
            input_image = Image.open(samples[len(samples) // 2]).convert("RGB")
            st.info(f"Loaded representative **{quick_sample}** sample image.")

with col_ctrl:
    st.markdown("### Target Class Explanation")
    target_class_choice = st.selectbox(
        "Generate Explanation for Navarasa:",
        ["Automatic (Top Predicted Rasa)"] + NAVARASA_CLASSES,
        index=0
    )

if input_image is not None:
    st.markdown("---")
    with st.spinner("Computing convolutional activations and backpropagating gradients..."):
        # Load engines
        eff_model = get_efficientnet_model()
        gradcam = get_gradcam_engine()
        facial_analyzer = get_facial_analyzer()
        au_extractor = get_au_extractor()
        explainer = get_natural_language_explainer()

        # Transform image
        transform = get_efficientnet_transforms(is_training=False)
        tensor = transform(input_image).unsqueeze(0)

        # Base prediction
        probs = eff_model.predict_probabilities(tensor)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_class = NAVARASA_CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        # Determine target class index for Grad-CAM
        if target_class_choice == "Automatic (Top Predicted Rasa)":
            cam_class_idx = pred_idx
            cam_class_name = pred_class
        else:
            cam_class_idx = CLASS_TO_IDX[target_class_choice]
            cam_class_name = target_class_choice

        # Generate exact Grad-CAM heatmap
        heatmap, _, _ = gradcam.generate_heatmap(tensor, target_class_idx=cam_class_idx)

        # Generate overlay
        cmap_name = "jet" if "JET" in colormap_choice else "inferno"
        overlay = gradcam.overlay_heatmap(input_image, heatmap, alpha=alpha_slider, colormap=cmap_name)

        # Extract landmarks and 6-region masks
        landmarks, det_result = facial_analyzer.extract_landmarks(input_image)
        masks = facial_analyzer.generate_region_masks(np.array(input_image).shape, landmarks)
        region_pcts, sorted_regions = facial_analyzer.calculate_region_importance(heatmap, masks)

        # Extract AUs
        aus, face_ok = au_extractor.extract_action_units(input_image)
        active_aus = [au for au in aus.values() if au["is_active"]]

    # Header Card for Prediction
    st.markdown(
        f"""
        <div class="navarasa-card" style="border-left: 5px solid {THEME['bright_gold']};">
            <span class="badge-gold">GRAD-CAM TARGET: {cam_class_name.upper()}</span>
            <h2 style="margin: 8px 0; color: {THEME['deep_green']};">
                Explaining Prediction for: {cam_class_name} ({probs[cam_class_idx] * 100:.1f}% probability)
            </h2>
            <p style="font-size: 0.9rem; color: {THEME['text_muted']};">
                Highest activation concentrates in the <b>{sorted_regions[0][0]}</b> ({sorted_regions[0][1]:.1f}%) 
                and <b>{sorted_regions[1][0]}</b> ({sorted_regions[1][1]:.1f}%) regions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Visual Triad: Original, Heatmap, Overlay
    col_orig, col_cam, col_over = st.columns(3)
    with col_orig:
        st.markdown("#### 1. Input Image")
        st.image(input_image, use_container_width=True)
    with col_cam:
        st.markdown(f"#### 2. Raw Grad-CAM ({cam_class_name})")
        import matplotlib.pyplot as plt
        cmap_fn = plt.colormaps[cmap_name] if hasattr(plt, "colormaps") else plt.cm.get_cmap(cmap_name)
        cam_vis_colored = (cmap_fn(heatmap)[:, :, :3] * 255).astype(np.uint8)
        st.image(cam_vis_colored, use_container_width=True)
    with col_over:
        st.markdown("#### 3. Overlay Grounding")
        st.image(overlay, use_container_width=True)

    st.markdown("---")

    # Section: Regional Importance Breakdown
    col_bar, col_text = st.columns([1.2, 1])
    with col_bar:
        st.markdown("### Facial-Region Grad-CAM Energy Distribution")
        fig_r = plot_region_importance_chart(region_pcts)
        st.pyplot(fig_r)

    with col_text:
        st.markdown("### Anatomical Importance Ranking")
        st.markdown(
            "Proportion of positive convolutional feature gradients captured within each anatomical mask:"
        )
        for r_name, pct in sorted_regions:
            st.write(f"**{r_name}**: {pct:.1f}%")
            st.progress(float(pct / 100.0))

        st.caption(
            "Note: Regional activation percentages reflect convolutional feature attention density "
            "and should not be interpreted as definitive causal proof."
        )

    st.markdown("---")

    # Section: Evidence-Based Natural Language Explanation
    st.markdown("### Structured Evidence-Based Explanation")
    prob_dict = {cls: float(probs[i]) for i, cls in enumerate(NAVARASA_CLASSES)}
    explanation_text = explainer.generate_explanation(
        predicted_class=cam_class_name,
        confidence=float(probs[cam_class_idx]),
        probabilities=prob_dict,
        top_regions=sorted_regions,
        active_action_units=active_aus,
    )
    st.markdown(explanation_text)
else:
    st.info("Please upload a facial image or pick an authentic sample from the sidebar.")
