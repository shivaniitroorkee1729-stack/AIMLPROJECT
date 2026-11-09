"""
Page 5: Occlusion Sensitivity Testing and Counterfactual Feature Analysis
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
)
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

st.set_page_config(page_title="Occlusion & Counterfactuals — NAVARASA AI", page_icon="🎭", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.markdown(
    """
    <div class="navarasa-header-card">
        <span class="badge-gold">PERTURBATION & CAUSAL SENSITIVITY</span>
        <h1 style="margin-top: 10px;">Occlusion Testing & Counterfactual Analysis</h1>
        <p style="opacity: 0.95; font-size: 1.05rem;">
            Test the model's empirical sensitivity to individual anatomical facial regions via systematic masking,
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
    uploaded = st.file_uploader("Upload Facial Image (JPG / PNG):", type=["jpg", "jpeg", "png"])
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

if input_image is not None:
    st.markdown("---")
    with st.spinner("Executing region occlusion tests and counterfactual analysis..."):
        facial_analyzer = get_facial_analyzer()
        occlusion_analyzer = get_occlusion_analyzer()
        cf_analyzer = get_counterfactual_analyzer()
        au_extractor = get_au_extractor()
        eff_model = get_efficientnet_model()
        gradcam = get_gradcam_engine()
        explainer = get_natural_language_explainer()

        # Landmarks and masks
        landmarks, _ = facial_analyzer.extract_landmarks(input_image)
        masks = facial_analyzer.generate_region_masks(np.array(input_image).shape, landmarks)

        # Run full sensitivity analysis
        occ_results = occlusion_analyzer.run_sensitivity_analysis(
            input_image, masks, mode=occlusion_mode
        )

        # Extract AUs
        aus, face_ok = au_extractor.extract_action_units(input_image)

        # Counterfactual analysis
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
        st.pyplot(fig_occ)

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
            st.image(input_image, caption="Original Image", use_container_width=True)
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
                st.success(f"✅ Prediction Remains: **{reg_data['masked_predicted_class']}**")

    with tab_cf:
        st.markdown(
            f"""
            <div class="navarasa-card">
                <h3>Steering Prediction: {orig_cls} ➔ {target_counterfactual}</h3>
                <p style="color: {THEME['text_muted']}; font-size: 0.9rem;">
                    Identifies the minimal physiological Action Unit contractions and relaxations 
                    required to align the facial expression with the target aesthetic sentiment.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        shifts = cf_results.get("feature_adjustments", [])
        if shifts:
            st.markdown("#### Prescribed Action Unit Adjustments")
            cf_table = []
            for s in shifts:
                cf_table.append({
                    "Action Unit": f"{s['au_id']} ({s['name']})",
                    "Direction": s["direction"],
                    "Current Intensity": f"{s['current_intensity']:.2f}",
                    "Target Intensity": f"{s['suggested_intensity']:.2f}",
                    "Required Delta": f"{s['delta']:+.2f}",
                    "Muscular Rationale": s["anatomical_action"]
                })
            st.dataframe(cf_table, use_container_width=True, hide_index=True)
        else:
            st.info("No significant Action Unit adjustments needed for this pair.")

        st.markdown(
            f"""
            > [!NOTE]
            > **Scientific Rigor Disclaimer:** {cf_results.get('scientific_disclaimer', '')}
            """
        )
else:
    st.info("Please upload a facial image or pick an authentic sample from the sidebar.")
