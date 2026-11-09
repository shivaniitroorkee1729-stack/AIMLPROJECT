"""
Page 4: Facial Analysis — MediaPipe Face Mesh, 6 Anatomical Regions,
FACS Action Unit Extraction, and Top-Two Class Comparative Contrast.
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
    plot_action_units_chart,
)
from src.models.model_loader import (
    get_efficientnet_model,
    get_gradcam_engine,
    get_facial_analyzer,
    get_au_extractor,
    get_top_two_comparator,
)
from src.data.preprocessing import get_efficientnet_transforms

st.set_page_config(page_title="Facial Analysis — NAVARASA AI", page_icon="🧬", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.markdown(
    """
    <div class="navarasa-header-card">
        <span class="badge-gold">ANATOMICAL & MUSCULAR ANALYSIS</span>
        <h1 style="margin-top: 10px;">Facial Analysis & Top-Two Class Comparison</h1>
        <p style="opacity: 0.95; font-size: 1.05rem;">
            Dense 478-point facial mesh mapping, six-region anatomical segmentation,
            FACS Action Unit muscular measurement, and contrastive analysis between competing predictions.
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
        index=2  # Karuna
    )

col_up, col_info = st.columns([1.2, 1])
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

with col_info:
    st.markdown(
        f"""
        <div class="navarasa-card">
            <h4>FACS Action Unit System</h4>
            <p style="font-size: 0.85rem; color: {THEME['text_muted']};">
                The Facial Action Coding System (FACS) decomposes facial expressions into individual 
                anatomical muscle contractions called <b>Action Units (AUs)</b>. 
                Our system extracts measured blendshape coefficients directly from facial geometry.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if input_image is not None:
    st.markdown("---")
    with st.spinner("Analyzing facial geometry and muscular action units..."):
        facial_analyzer = get_facial_analyzer()
        au_extractor = get_au_extractor()
        eff_model = get_efficientnet_model()
        gradcam = get_gradcam_engine()
        comparator = get_top_two_comparator()

        # 1. Landmarks & Masks
        landmarks, det_result = facial_analyzer.extract_landmarks(input_image)
        masks = facial_analyzer.generate_region_masks(np.array(input_image).shape, landmarks)
        annotated_img = facial_analyzer.draw_landmarks_and_regions(input_image, landmarks, masks)

        # 2. Action Units
        aus, face_detected = au_extractor.extract_action_units(input_image)

        # 3. Model Inference & Grad-CAM for Top Two
        transform = get_efficientnet_transforms(is_training=False)
        tensor = transform(input_image).unsqueeze(0)
        probs = eff_model.predict_probabilities(tensor)[0].cpu().numpy()
        prob_dict = {cls: float(probs[i]) for i, cls in enumerate(NAVARASA_CLASSES)}

        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        top1_cls, top1_prob = sorted_probs[0]
        top2_cls, top2_prob = sorted_probs[1]

        # Grad-CAM for Top 1 and Top 2
        from config.settings import CLASS_TO_IDX
        cam1, _, _ = gradcam.generate_heatmap(tensor, target_class_idx=CLASS_TO_IDX[top1_cls])
        cam2, _, _ = gradcam.generate_heatmap(tensor, target_class_idx=CLASS_TO_IDX[top2_cls])

        reg_pcts1, _ = facial_analyzer.calculate_region_importance(cam1, masks)
        reg_pcts2, _ = facial_analyzer.calculate_region_importance(cam2, masks)

        top2_report = comparator.compare(prob_dict, reg_pcts1, reg_pcts2, aus)

    tab_mesh, tab_au, tab_top2 = st.tabs([
        "📍 Dense Mesh & 6 Regions",
        "⚡ FACS Action Unit Measurements",
        "⚔️ Top-Two Class Contrast"
    ])

    with tab_mesh:
        st.markdown("### Facial Landmark Mesh and 6 Anatomical Region Boundaries")
        col_m1, col_m2 = st.columns([1, 1.2])
        with col_m1:
            st.image(annotated_img, caption="Face Mesh & Region Contours", use_container_width=True)
        with col_m2:
            st.markdown("#### Anatomical Region Segmentations")
            st.markdown(
                f"""
                - 🔵 **Eyes:** Orbital aperture and eyelid boundaries
                - 🔴 **Eyebrows:** Corrugator and frontalis muscle paths
                - 🟡 **Nose:** Nasal bridge, dorsum, and alar lobules
                - 🟢 **Mouth:** Upper/lower vermilion borders and commissures
                - 🟠 **Cheeks:** Zygomatic region and buccal fat pads
                - 🟣 **Forehead:** Superior frontal cranium above supraorbital ridge
                """
            )
            st.info(f"Detected **{len(landmarks)}** 3D landmark coordinates.")

    with tab_au:
        st.markdown("### Measured Facial Action Units (FACS)")
        if not face_detected:
            st.warning("No face detected by landmarker; reporting zeroed baselines.")

        fig_au = plot_action_units_chart(aus)
        st.pyplot(fig_au)

        st.markdown("#### Action Unit Detail Table")
        au_table = []
        for au_id, meta in aus.items():
            au_table.append({
                "AU ID": au_id,
                "Name": meta["name"],
                "Muscle Group": meta["muscle"],
                "Intensity": f"{meta['intensity']:.2f}",
                "Status": "ACTIVE" if meta["is_active"] else "Inactive",
                "Indicative Rasas": ", ".join(meta["relevant_rasas"]),
            })
        st.dataframe(au_table, use_container_width=True, hide_index=True)

    with tab_top2:
        st.markdown("### Top-Two Competing Navarasa Predictions")
        st.markdown(
            f"""
            <div class="navarasa-card" style="border-left: 5px solid {THEME['deep_green']};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3>1st: {top1_cls} ({top1_prob * 100:.1f}%) &nbsp; vs &nbsp; 2nd: {top2_cls} ({top2_prob * 100:.1f}%)</h3>
                    <span class="badge-gold">{top2_report['margin_status']} (Margin: {top2_report['probability_margin'] * 100:.1f}%)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c_t1, c_t2 = st.columns(2)
        with c_t1:
            st.markdown(f"#### Primary: {top1_cls}")
            st.write(f"**Meaning:** {top2_report['top1_meaning']}")
            st.markdown("##### Regions Favoring This Prediction:")
            if top2_report["regions_favoring_top1"]:
                for r in top2_report["regions_favoring_top1"]:
                    diff = top2_report["region_differences"][r]
                    st.write(f"- **{r}**: +{diff:.1f}% higher attention")
            else:
                st.write("Diffused regional attention.")

            st.markdown("##### Corroborating Active AUs:")
            if top2_report["active_aus_top1"]:
                for au in top2_report["active_aus_top1"]:
                    st.write(f"- **{au['id']}** ({au['name']}): intensity {au['intensity']:.2f}")
            else:
                st.write("None active above threshold.")

        with c_t2:
            st.markdown(f"#### Alternative: {top2_cls}")
            st.write(f"**Meaning:** {top2_report['top2_meaning']}")
            st.markdown("##### Regions Favoring This Alternative:")
            if top2_report["regions_favoring_top2"]:
                for r in top2_report["regions_favoring_top2"]:
                    diff = abs(top2_report["region_differences"][r])
                    st.write(f"- **{r}**: +{diff:.1f}% higher attention when explaining {top2_cls}")
            else:
                st.write("No distinct regional advantage.")

            st.markdown("##### Corroborating Active AUs:")
            if top2_report["active_aus_top2"]:
                for au in top2_report["active_aus_top2"]:
                    st.write(f"- **{au['id']}** ({au['name']}): intensity {au['intensity']:.2f}")
            else:
                st.write("None active above threshold.")
else:
    st.info("Please upload a facial image or pick an authentic sample from the sidebar.")
