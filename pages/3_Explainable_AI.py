"""
Page 3: Explainable AI — Exact Grad-CAM Visual Grounding & Facial Region Energy Analysis
Enforces Human Face Detection and Face Crop Restriction before computing heatmaps.
"""

import sys
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import NAVARASA_CLASSES, RASA_DESCRIPTIONS, PROCESSED_DATA_DIR, CLASS_TO_IDX
from src.utils.visualization import (
    get_custom_css,
    THEME,
    plot_region_importance_chart,
    render_pyplot_and_close,
)
from src.utils.memory import run_garbage_collection
from src.models.model_loader import (
    get_efficientnet_model,
    get_gradcam_engine,
    get_facial_analyzer,
    get_natural_language_explainer,
    get_au_extractor,
)
from src.data.preprocessing import get_efficientnet_transforms
from src.face_detection import render_face_validation_card

st.set_page_config(page_title="Explainable AI — NAVARASA AI", page_icon="🔍", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.markdown(
    """
    <div class="navarasa-header-card">
        <span class="badge-gold">VISUAL GROUNDING & EXPLAINABILITY</span>
        <h1 style="margin-top: 10px;">Grad-CAM & Facial Region Importance</h1>
        <p style="opacity: 0.95; font-size: 1.05rem;">
            Calculates exact gradient-weighted class activations from the final convolutional stage of EfficientNet-B0,
            strictly restricted to the validated facial crop, overlaying neural attention on the facial anatomy.
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
    uploaded = st.file_uploader("Upload Image (JPG / PNG):", type=["jpg", "jpeg", "png"])
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

# Strict Human Face Validation Layer
validation_res = render_face_validation_card(input_image)

if validation_res is None:
    st.info("Please upload an image or pick an authentic sample from the sidebar.")
elif not validation_res.is_valid:
    # All XAI processing is blocked by design
    pass
else:
    face_crop = validation_res.face_crop
    original_img = validation_res.original_image
    crop_bbox = validation_res.crop_bbox

    cache_key = f"xai_{validation_res.image_hash}_{target_class_choice}"
    if cache_key in st.session_state:
        cached = st.session_state[cache_key]
        probs = cached["probs"]
        pred_idx = cached["pred_idx"]
        pred_class = cached["pred_class"]
        cam_class_idx = cached["cam_class_idx"]
        cam_class_name = cached["cam_class_name"]
        heatmap = cached["heatmap"]
        masks = cached["masks"]
        region_pcts = cached["region_pcts"]
        sorted_regions = cached["sorted_regions"]
        active_aus = cached["active_aus"]
    else:
        with st.spinner("Computing convolutional activations strictly on facial crop..."):
            eff_model = get_efficientnet_model()
            gradcam = get_gradcam_engine()
            facial_analyzer = get_facial_analyzer()
            au_extractor = get_au_extractor()

            # Transform face crop to tensor
            transform = get_efficientnet_transforms(is_training=False)
            tensor = transform(face_crop).unsqueeze(0)

            # Base prediction on face crop
            probs = eff_model.predict_probabilities(tensor)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_class = NAVARASA_CLASSES[pred_idx]

            # Determine target class index for Grad-CAM
            if target_class_choice == "Automatic (Top Predicted Rasa)":
                cam_class_idx = pred_idx
                cam_class_name = pred_class
            else:
                cam_class_idx = CLASS_TO_IDX[target_class_choice]
                cam_class_name = target_class_choice

            # Generate exact Grad-CAM heatmap on the face crop
            heatmap, _, _ = gradcam.generate_heatmap(tensor, target_class_idx=cam_class_idx)

            # Extract landmarks and 6-region masks from the facial crop
            landmarks, _ = facial_analyzer.extract_landmarks(face_crop)
            masks = facial_analyzer.generate_region_masks(np.array(face_crop).shape, landmarks)
            region_pcts, sorted_regions = facial_analyzer.calculate_region_importance(heatmap, masks)

            # Extract AUs from the facial crop
            aus, _ = au_extractor.extract_action_units(face_crop)
            active_aus = [au for au in aus.values() if au["is_active"]]

            del tensor
            run_garbage_collection()

            # Maintain bounded session cache (max 4 entries)
            if "xai_cache_keys" not in st.session_state:
                st.session_state["xai_cache_keys"] = []
            if len(st.session_state["xai_cache_keys"]) >= 4:
                oldest = st.session_state["xai_cache_keys"].pop(0)
                st.session_state.pop(oldest, None)
            st.session_state["xai_cache_keys"].append(cache_key)

            st.session_state[cache_key] = {
                "probs": probs,
                "pred_idx": pred_idx,
                "pred_class": pred_class,
                "cam_class_idx": cam_class_idx,
                "cam_class_name": cam_class_name,
                "heatmap": heatmap,
                "masks": masks,
                "region_pcts": region_pcts,
                "sorted_regions": sorted_regions,
                "active_aus": active_aus,
            }

    gradcam = get_gradcam_engine()
    explainer = get_natural_language_explainer()

    # Generate overlay on the face crop
    cmap_name = "jet" if "JET" in colormap_choice else "inferno"
    crop_overlay = gradcam.overlay_heatmap(face_crop, heatmap, alpha=alpha_slider, colormap=cmap_name)

    # Generate overlay mapped strictly to the face bounding box in original image
    orig_overlay = gradcam.overlay_heatmap_on_original(
        original_img,
        heatmap,
        crop_bbox,
        alpha=alpha_slider,
        colormap=cmap_name,
    )

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

    # Visual Triad for Facial Crop: Crop, Raw Heatmap, Crop Overlay
    st.markdown("### 🎯 Facial Crop Grad-CAM Visual Grounding")
    st.caption("All gradient backpropagation operates strictly inside the facial boundary — no background, neck, or clothing artifacts.")
    c_crop, c_cam, c_over = st.columns(3)
    with c_crop:
        st.markdown("#### 1. Extracted Face Crop")
        st.image(face_crop, use_container_width=True)
    with c_cam:
        st.markdown(f"#### 2. Raw Facial Heatmap ({cam_class_name})")
        cmap_fn = plt.colormaps[cmap_name] if hasattr(plt, "colormaps") else plt.cm.get_cmap(cmap_name)
        cam_vis_colored = (cmap_fn(heatmap)[:, :, :3] * 255).astype(np.uint8)
        st.image(cam_vis_colored, use_container_width=True)
    with c_over:
        st.markdown("#### 3. Facial Crop Overlay")
        st.image(crop_overlay, use_container_width=True)

    # Full Context Localization Comparison
    st.markdown("---")
    st.markdown("### 📍 Bounding Box Localized Attention in Full Context")
    st.caption("Heatmap is projected cleanly back into the validated face bounding box. Notice surrounding background and clothing are untouched.")
    c_full_orig, c_full_over = st.columns(2)
    with c_full_orig:
        st.image(validation_res.annotated_image, caption="Original with Detected Face BBox", use_container_width=True)
    with c_full_over:
        st.image(orig_overlay, caption="Original with Face-Constrained Grad-CAM", use_container_width=True)

    st.markdown("---")

    # Section: Regional Importance Breakdown
    col_bar, col_text = st.columns([1.2, 1])
    with col_bar:
        st.markdown("### Facial-Region Grad-CAM Energy Distribution")
        fig_r = plot_region_importance_chart(region_pcts)
        render_pyplot_and_close(fig_r)

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
