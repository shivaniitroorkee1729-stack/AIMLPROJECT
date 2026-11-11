"""
Page 5: Occlusion Sensitivity Testing and Counterfactual Feature Analysis.
Enforces Human Face Detection and Face Crop Restriction before applying perturbations.
"""

import sys
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import NAVARASA_CLASSES, RASA_DESCRIPTIONS, PROCESSED_DATA_DIR
from src.utils.visualization import (
    get_custom_css,
    THEME,
    plot_occlusion_sensitivity_chart,
    render_pyplot_and_close,
)
from src.utils.memory import run_garbage_collection
from src.models.model_loader import (
    get_efficientnet_model,
    get_facial_analyzer,
    get_au_extractor,
    get_occlusion_analyzer,
    get_counterfactual_analyzer,
    get_natural_language_explainer,
    get_gradcam_engine,
)
from src.data.preprocessing import get_efficientnet_transforms
from src.face_detection import render_face_validation_card

st.set_page_config(page_title="Occlusion & Counterfactuals — NAVARASA AI", page_icon="🎭", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.markdown(
    """
    <div class="navarasa-header-card">
        <span class="badge-gold">PERTURBATION & CAUSAL SENSITIVITY</span>
        <h1 style="margin-top: 10px;">Occlusion Testing & Counterfactual Analysis</h1>
        <p style="opacity: 0.95; font-size: 1.05rem;">
            Test the model's empirical sensitivity to individual anatomical facial regions via systematic masking on the validated face crop,
            and inspect scientific feature-level counterfactual perturbations to steer predictions toward alternative Rasas.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🖼️ Sample Quick Select")
    quick_sample = st.selectbox(
        "Load an authentic test image:",
        ["None"] + NAVARASA_CLASSES,
        index=3  # Raudra
    )
    st.markdown("---")
    st.markdown("### 🎭 Occlusion Parameters")
    occlusion_mode = st.selectbox(
        "Masking Technique:",
        ["gaussian_blur", "gray", "black"],
        format_func=lambda x: "Gaussian Blur (Natural De-focus)" if x == "gaussian_blur" else ("Neutral Gray (128)" if x == "gray" else "Blackout (0)")
    )

col_up, col_param = st.columns([1.2, 1])
input_image = None

with col_up:
    uploaded = st.file_uploader("Upload Image (JPG / PNG):", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        input_image = Image.open(uploaded).convert("RGB")
    elif quick_sample != "None":
        cls_dir = PROCESSED_DATA_DIR / quick_sample
        samples = list(cls_dir.glob("*.jpg"))
        if samples:
            input_image = Image.open(samples[len(samples) // 2]).convert("RGB")
            st.info(f"Loaded sample image from **{quick_sample}** class.")

with col_param:
    st.markdown("### Counterfactual Goal")
    target_counterfactual = st.selectbox(
        "Target Alternative Navarasa:",
        [c for c in NAVARASA_CLASSES if c != quick_sample],
        index=0
    )

# Strict Human Face Validation Layer
validation_res = render_face_validation_card(input_image)

if validation_res is None:
    st.info("Please upload an image or pick an authentic sample from the sidebar.")
elif not validation_res.is_valid:
    pass
else:
    face_crop = validation_res.face_crop

    cache_key = f"occ_{validation_res.image_hash}_{occlusion_mode}"
    if cache_key in st.session_state:
        cached = st.session_state[cache_key]
        masks = cached["masks"]
        occ_results = cached["occ_results"]
        aus = cached["aus"]
    else:
        with st.spinner("Executing region occlusion tests on facial crop..."):
            facial_analyzer = get_facial_analyzer()
            occlusion_analyzer = get_occlusion_analyzer()
            au_extractor = get_au_extractor()

            # Landmarks and masks strictly on facial crop
            landmarks, _ = facial_analyzer.extract_landmarks(face_crop)
            masks = facial_analyzer.generate_region_masks(np.array(face_crop).shape, landmarks)

            # Run full sensitivity analysis on facial crop
            occ_results = occlusion_analyzer.run_sensitivity_analysis(
                face_crop, masks, mode=occlusion_mode
            )

            # Extract AUs from facial crop
            aus, face_ok = au_extractor.extract_action_units(face_crop)

            run_garbage_collection()

            # Maintain bounded session cache (max 4 entries)
            if "occ_cache_keys" not in st.session_state:
                st.session_state["occ_cache_keys"] = []
            if len(st.session_state["occ_cache_keys"]) >= 4:
                oldest = st.session_state["occ_cache_keys"].pop(0)
                st.session_state.pop(oldest, None)
            st.session_state["occ_cache_keys"].append(cache_key)

            st.session_state[cache_key] = {
                "masks": masks,
                "occ_results": occ_results,
                "aus": aus,
            }

    # Counterfactual analysis is lightweight mathematical comparison
    cf_analyzer = get_counterfactual_analyzer()
    orig_cls = occ_results["original"]["predicted_class"]
    orig_probs = occ_results["original"]["probabilities"]
    cf_results = cf_analyzer.analyze_counterfactual(
        current_class=orig_cls,
        target_class=target_counterfactual,
        current_aus=aus,
        current_probs=orig_probs
    )

    tab_occ, tab_cf = st.tabs(["🛡️ Regional Occlusion Sensitivity", "🔄 Counterfactual Feature Shifts"])

    with tab_occ:
        st.markdown(
            f"""
            <div class="navarasa-card">
                <h3>Original Unmasked Prediction: <span style="color: {THEME['deep_green']}">{orig_cls}</span> 
                ({occ_results['original']['confidence'] * 100:.1f}%)</h3>
                <p style="color: {THEME['text_muted']};">
                    <b>Most Sensitive Anatomical Region:</b> 
                    <span style="color: #D32F2F; font-weight: 700;">{occ_results['most_sensitive_region']}</span> 
                    (caused a drop of {occ_results['max_confidence_drop'] * 100:.1f}% when masked).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Sensitivity Bar Chart (Confidence Drop per Masked Region)")
        fig_occ = plot_occlusion_sensitivity_chart(occ_results["regions"])
        render_pyplot_and_close(fig_occ)

        st.markdown("---")
        st.markdown("### Interactive Region Mask Inspector")
        inspect_region = st.selectbox(
            "Select specific region to view masked image & detailed impact:",
            list(masks.keys())
        )

        reg_data = occ_results["regions"][inspect_region]
        occluded_preview = occ_results["occluded_images"][inspect_region]

        col_img1, col_img2, col_stat = st.columns([1, 1, 1.2])
        with col_img1:
            st.image(face_crop, caption="Facial Crop (Unmasked)", use_container_width=True)
        with col_img2:
            st.image(occluded_preview, caption=f"Masked {inspect_region}", use_container_width=True)
        with col_stat:
            st.markdown(f"#### Impact of Masking {inspect_region}")
            st.write(f"**Target Class Original Prob:** {reg_data['target_class_original_prob'] * 100:.1f}%")
            st.write(f"**Target Class Masked Prob:** {reg_data['target_class_masked_prob'] * 100:.1f}%")
            st.write(f"**Confidence Drop:** {reg_data['confidence_drop'] * 100:.1f}%")
            if reg_data["class_flipped"]:
                st.error(f"🚨 Prediction Flipped to: **{reg_data['masked_predicted_class']}** ({reg_data['masked_confidence'] * 100:.1f}%)")
            else:
                st.success("✅ Prediction stable (no class flip)")

    with tab_cf:
        target_prob_val = cf_results.get("target_class_current_prob", cf_results.get("target_probability", 0.0))
        shift_val = cf_results.get("required_shift", max(0.0, round(orig_probs.get(orig_cls, 0.0) - target_prob_val, 4)))
        st.markdown(
            f"""
            <div class="navarasa-card">
                <h3>Counterfactual Goal: Shift from <b>{orig_cls}</b> &rarr; <b>{target_counterfactual}</b></h3>
                <p style="color: {THEME['text_muted']};">
                    Baseline Target Probability: <b>{target_prob_val * 100:.2f}%</b> &bull;
                    Required Confidence Shift: <b>+{shift_val * 100:.2f}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Recommended Muscular Action Unit Adjustments")
        cf_recs = cf_results.get("recommendations", cf_results.get("feature_adjustments", []))

        if not cf_recs:
            st.info("No major muscular adjustments required — already close or minimal divergence.")
        else:
            for rec in cf_recs:
                au_id = rec.get("au_id", "")
                au_name = rec.get("name", "")
                au_muscle = rec.get("muscle", "")
                au_action = rec.get("action", "Adjust")
                au_shift = rec.get("target_intensity_shift", abs(rec.get("delta", 0.0)))
                au_vis = rec.get("visual_direction", rec.get("anatomical_action", ""))
                st.markdown(
                    f"""
                    <div style="background: #FFFFFF; border-left: 4px solid {THEME['deep_green']}; padding: 10px 14px; margin-bottom: 8px; border-radius: 6px;">
                        <b>{au_id} — {au_name} ({au_muscle}):</b><br/>
                        Action: <b>{au_action}</b> by ~<b>{au_shift * 100:.0f}%</b> intensity.<br/>
                        <span style="font-size: 0.85rem; color: {THEME['text_muted']};">{au_vis}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
