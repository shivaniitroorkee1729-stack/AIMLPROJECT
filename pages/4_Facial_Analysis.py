"""
Page 4: Facial Analysis — MediaPipe Face Mesh, 6 Anatomical Regions,
FACS Action Unit Extraction, and Top-Two Class Comparative Contrast.
Enforces Human Face Detection and Face Crop Restriction before processing.
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
    plot_action_units_chart,
    render_pyplot_and_close,
)
from src.utils.memory import run_garbage_collection
from src.models.model_loader import (
    get_efficientnet_model,
    get_gradcam_engine,
    get_facial_analyzer,
    get_au_extractor,
    get_top_two_comparator,
)
from src.data.preprocessing import get_efficientnet_transforms
from src.face_detection import render_face_validation_card

st.set_page_config(page_title="Facial Analysis — NAVARASA AI", page_icon="🧬", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.markdown(
    """
    <div class="navarasa-header-card">
        <span class="badge-gold">ANATOMICAL & MUSCULAR ANALYSIS</span>
        <h1 style="margin-top: 10px;">Facial Analysis & Top-Two Class Comparison</h1>
        <p style="opacity: 0.95; font-size: 1.05rem;">
            Dense 478-point facial mesh mapping, six-region anatomical segmentation,
            FACS Action Unit muscular measurement, and contrastive analysis strictly executed on the validated face crop.
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
    uploaded = st.file_uploader("Upload Image (JPG / PNG):", type=["jpg", "jpeg", "png"])
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
                Our system extracts measured blendshape coefficients directly from the validated facial crop.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Strict Human Face Validation Layer
validation_res = render_face_validation_card(input_image)

if validation_res is None:
    st.info("Please upload an image or pick an authentic sample from the sidebar.")
elif not validation_res.is_valid:
    pass
else:
    face_crop = validation_res.face_crop

    cache_key = f"fa_{validation_res.image_hash}"
    if cache_key in st.session_state:
        cached = st.session_state[cache_key]
        landmarks = cached["landmarks"]
        masks = cached["masks"]
        annotated_crop = cached["annotated_crop"]
        aus = cached["aus"]
        face_detected = cached["face_detected"]
        prob_dict = cached["prob_dict"]
        sorted_probs = cached["sorted_probs"]
        top1_cls = cached["top1_cls"]
        top1_prob = cached["top1_prob"]
        top2_cls = cached["top2_cls"]
        top2_prob = cached["top2_prob"]
        reg_pcts1 = cached["reg_pcts1"]
        reg_pcts2 = cached["reg_pcts2"]
        top2_report = cached["top2_report"]
    else:
        with st.spinner("Analyzing facial geometry and muscular action units on facial crop..."):
            facial_analyzer = get_facial_analyzer()
            au_extractor = get_au_extractor()
            eff_model = get_efficientnet_model()
            gradcam = get_gradcam_engine()
            comparator = get_top_two_comparator()

            # 1. Landmarks & Masks strictly on facial crop
            landmarks, det_result = facial_analyzer.extract_landmarks(face_crop)
            masks = facial_analyzer.generate_region_masks(np.array(face_crop).shape, landmarks)
            annotated_crop = facial_analyzer.draw_landmarks_and_regions(face_crop, landmarks, masks)

            # 2. Action Units strictly from facial crop
            aus, face_detected = au_extractor.extract_action_units(face_crop)

            # 3. Model Inference & Grad-CAM for Top Two on facial crop
            transform = get_efficientnet_transforms(is_training=False)
            tensor = transform(face_crop).unsqueeze(0)
            probs = eff_model.predict_probabilities(tensor)[0].cpu().numpy()
            prob_dict = {cls: float(probs[i]) for i, cls in enumerate(NAVARASA_CLASSES)}

            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            top1_cls, top1_prob = sorted_probs[0]
            top2_cls, top2_prob = sorted_probs[1]

            # Grad-CAM for Top 1 and Top 2
            cam1, _, _ = gradcam.generate_heatmap(tensor, target_class_idx=CLASS_TO_IDX[top1_cls])
            cam2, _, _ = gradcam.generate_heatmap(tensor, target_class_idx=CLASS_TO_IDX[top2_cls])

            reg_pcts1, _ = facial_analyzer.calculate_region_importance(cam1, masks)
            reg_pcts2, _ = facial_analyzer.calculate_region_importance(cam2, masks)

            top2_report = comparator.compare(prob_dict, reg_pcts1, reg_pcts2, aus)

            del tensor
            run_garbage_collection()

            # Maintain bounded session cache (max 4 entries)
            if "fa_cache_keys" not in st.session_state:
                st.session_state["fa_cache_keys"] = []
            if len(st.session_state["fa_cache_keys"]) >= 4:
                oldest = st.session_state["fa_cache_keys"].pop(0)
                st.session_state.pop(oldest, None)
            st.session_state["fa_cache_keys"].append(cache_key)

            st.session_state[cache_key] = {
                "landmarks": landmarks,
                "masks": masks,
                "annotated_crop": annotated_crop,
                "aus": aus,
                "face_detected": face_detected,
                "prob_dict": prob_dict,
                "sorted_probs": sorted_probs,
                "top1_cls": top1_cls,
                "top1_prob": top1_prob,
                "top2_cls": top2_cls,
                "top2_prob": top2_prob,
                "reg_pcts1": reg_pcts1,
                "reg_pcts2": reg_pcts2,
                "top2_report": top2_report,
            }

    tab_mesh, tab_au, tab_top2 = st.tabs([
        "📍 Dense Mesh & 6 Regions",
        "⚡ FACS Action Unit Measurements",
        "⚔️ Top-Two Class Contrast"
    ])

    with tab_mesh:
        st.markdown("### Facial Landmark Mesh and 6 Anatomical Region Boundaries")
        col_m1, col_m2 = st.columns([1, 1.2])
        with col_m1:
            st.image(annotated_crop, caption="Face Crop Mesh & Region Contours", use_container_width=True)
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
            st.info(f"Detected **{len(landmarks)}** 3D landmark coordinates inside validated facial crop.")

    with tab_au:
        st.markdown("### Measured Facial Action Units (FACS)")
        if not face_detected:
            st.warning("No face detected by landmarker; reporting zeroed baselines.")

        fig_au = plot_action_units_chart(aus)
        render_pyplot_and_close(fig_au)

        st.markdown("#### Action Unit Detail Table")
        au_table = []
        for au_id, meta in aus.items():
            au_table.append({
                "AU ID": au_id,
                "Name": meta["name"],
                "Muscle Group": meta["muscle"],
                "Intensity": f"{meta['intensity']:.2f}",
                "Status": "ACTIVE" if meta["is_active"] else "Inactive",
                "Associated Rasas": ", ".join(meta["relevant_rasas"]),
            })
        st.dataframe(au_table, use_container_width=True)

    with tab_top2:
        st.markdown("### Contrastive Analysis Between Top Two Competing Rasas")
        margin_val = top2_report.get("margin", top2_report.get("probability_margin", round(top1_prob - top2_prob, 4)))
        summary_val = top2_report.get(
            "summary",
            f"{top1_cls} leads over {top2_cls} with a {margin_val * 100:.1f}% probability margin ({top2_report.get('margin_status', 'Decisive Lead')})."
        )
        st.markdown(
            f"""
            <div class="navarasa-card">
                <h3>Top-Two Probability Margin: <b>{margin_val * 100:.1f}%</b></h3>
                <p>
                    Primary Candidate: <b>{top1_cls}</b> ({top1_prob * 100:.1f}%) &bull; 
                    Secondary Competitor: <b>{top2_cls}</b> ({top2_prob * 100:.1f}%)
                </p>
                <p style="font-size: 0.95rem; color: {THEME['text_dark']};">
                    {summary_val}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Anatomical Region Activation Differential (Primary vs Competitor)")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown(f"**{top1_cls} (Primary) Region Activation:**")
            for r, v in sorted(reg_pcts1.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {r}: {v:.1f}%")
        with col_c2:
            st.markdown(f"**{top2_cls} (Competitor) Region Activation:**")
            for r, v in sorted(reg_pcts2.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {r}: {v:.1f}%")
