"""
Page 2: Navarasa Prediction — Single and Dual-Model Classification Dashboard
"""

import sys
from pathlib import Path
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import NAVARASA_CLASSES, RASA_DESCRIPTIONS, PROCESSED_DATA_DIR
from src.utils.visualization import (
    get_custom_css,
    THEME,
    plot_probability_bars,
    plot_dual_probability_comparison_chart,
)
from src.models.inference import predict_clip, predict_efficientnet, predict_dual_models

st.set_page_config(page_title="Prediction — NAVARASA AI", page_icon="🎭", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.markdown(
    """
    <div class="navarasa-header-card">
        <span class="badge-gold">INFERENCE DASHBOARD</span>
        <h1 style="margin-top: 10px;">Navarasa Facial Expression Classification</h1>
        <p style="opacity: 0.95; font-size: 1.05rem;">
            Upload a facial image or pick a test image to classify the expression into the 
            nine Indian classical Navarasas using CLIP, EfficientNet-B0, or dual-model comparative mode.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar controls
with st.sidebar:
    st.markdown("### ⚙️ Inference Settings")
    model_choice = st.radio(
        "Select Pipeline Mode:",
        [
            "Dual-Model Comparison (Simultaneous)",
            "CLIP ViT-B/32 + Logistic Regression",
            "EfficientNet-B0 (Fine-Tuned)",
        ],
        index=0
    )
    st.markdown("---")
    st.markdown("### 🖼️ Sample Quick Select")
    quick_sample = st.selectbox(
        "Or load an authentic test sample:",
        ["None"] + NAVARASA_CLASSES,
        index=0
    )

# Upload and Image Selection Area
col_input, col_preview = st.columns([1, 1])

uploaded_file = None
input_image = None

with col_input:
    st.markdown("### 1. Upload Facial Image")
    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image:",
        type=["jpg", "jpeg", "png"]
    )

    if quick_sample != "None" and uploaded_file is None:
        cls_dir = PROCESSED_DATA_DIR / quick_sample
        samples = list(cls_dir.glob("*.jpg"))
        if samples:
            sample_path = samples[len(samples) // 3]  # Pick a clean sample
            input_image = Image.open(sample_path).convert("RGB")
            st.info(f"Loaded sample image from **{quick_sample}** class.")

    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")

with col_preview:
    st.markdown("### 2. Image Preview")
    if input_image is not None:
        st.image(input_image, caption="Uploaded Facial Image", use_container_width=True)
    else:
        st.markdown(
            f"""
            <div style="background: #FFFFFF; border: 2px dashed {THEME['border_gold']}; border-radius: 12px; height: 260px; display: flex; align-items: center; justify-content: center; color: {THEME['text_muted']};">
                Upload an image or pick a sample from the sidebar to preview
            </div>
            """,
            unsafe_allow_html=True,
        )

# Run Prediction Button
if input_image is not None:
    st.markdown("---")
    if st.button("🚀 Classify Navarasa Facial Expression", use_container_width=True):
        with st.spinner("Executing neural inference..."):
            if model_choice == "Dual-Model Comparison (Simultaneous)":
                res = predict_dual_models(input_image)
                clip_res = res["clip"]
                eff_res = res["efficientnet"]

                # Agreement Banner
                if res["agreement"]:
                    st.success(f"🤝 **Both Models Agree!** Consensus Prediction: **{res['predicted_clip']}**")
                else:
                    st.warning(
                        f"⚡ **Model Divergence:** EfficientNet-B0 predicts **{res['predicted_efficientnet']}** "
                        f"while CLIP predicts **{res['predicted_clip']}**."
                    )

                # Dual Column Display
                c_eff, c_clip = st.columns(2)

                with c_eff:
                    st.markdown(
                        f"""
                        <div class="navarasa-card" style="border-top: 4px solid {RASA_DESCRIPTIONS.get(eff_res['predicted_class'], {}).get('color', '#124E3F')};">
                            <span class="badge-gold">EFFICIENTNET-B0</span>
                            <h2 style="margin: 8px 0; color: {THEME['deep_green']};">{eff_res['predicted_class']}</h2>
                            <div style="font-size: 1.1rem; font-weight: 700; color: {THEME['forest_green']};">
                                Confidence: {eff_res['confidence'] * 100:.1f}%
                            </div>
                            <p style="font-size: 0.85rem; color: {THEME['text_muted']}; margin-top: 6px;">
                                {eff_res['description'].get('meaning', '')}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("#### Top 3 Predictions (EfficientNet)")
                    for cls_name, prob in eff_res["top_3"]:
                        st.write(f"**{cls_name}**: {prob * 100:.1f}%")
                        st.progress(float(prob))

                with c_clip:
                    st.markdown(
                        f"""
                        <div class="navarasa-card" style="border-top: 4px solid {RASA_DESCRIPTIONS.get(clip_res['predicted_class'], {}).get('color', '#1E88E5')};">
                            <span class="badge-green">CLIP ViT-B/32 + LOGREG</span>
                            <h2 style="margin: 8px 0; color: {THEME['deep_green']};">{clip_res['predicted_class']}</h2>
                            <div style="font-size: 1.1rem; font-weight: 700; color: #1E88E5;">
                                Confidence: {clip_res['confidence'] * 100:.1f}%
                            </div>
                            <p style="font-size: 0.85rem; color: {THEME['text_muted']}; margin-top: 6px;">
                                {clip_res['description'].get('meaning', '')}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("#### Top 3 Predictions (CLIP)")
                    for cls_name, prob in clip_res["top_3"]:
                        st.write(f"**{cls_name}**: {prob * 100:.1f}%")
                        st.progress(float(prob))

                st.markdown("---")
                st.markdown("### Dual-Model Probability Distribution Comparison")
                fig_dual = plot_dual_probability_comparison_chart(clip_res["probabilities"], eff_res["probabilities"])
                st.pyplot(fig_dual)

            elif model_choice == "EfficientNet-B0 (Fine-Tuned)":
                res = predict_efficientnet(input_image)
                meta = res["description"]

                st.markdown(
                    f"""
                    <div class="navarasa-card" style="border-top: 5px solid {meta.get('color', '#124E3F')};">
                        <span class="badge-gold">PRIMARY PREDICTION</span>
                        <h1 style="margin: 8px 0; font-size: 2.2rem; color: {THEME['deep_green']};">{res['predicted_class']}</h1>
                        <div style="font-size: 1.25rem; font-weight: 700; color: {THEME['forest_green']};">
                            Confidence: {res['confidence'] * 100:.1f}%
                        </div>
                        <p style="font-size: 0.95rem; color: {THEME['text_muted']}; margin: 8px 0;">
                            <b>Bhaava:</b> {meta.get('bhaava', '')} &bull; <b>Aesthetic:</b> {meta.get('meaning', '')}
                        </p>
                        <div style="font-size: 0.85rem; background: {THEME['warm_ivory']}; padding: 8px 12px; border-radius: 6px; border-left: 3px solid {THEME['bright_gold']};">
                            <b>Expressive Cues:</b> {meta.get('visual_cues', '')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_top3, col_bars = st.columns([1, 1.4])
                with col_top3:
                    st.markdown("### Top 3 Most Likely Rasas")
                    for cls_name, prob in res["top_3"]:
                        st.write(f"**{cls_name}**: {prob * 100:.1f}%")
                        st.progress(float(prob))
                with col_bars:
                    fig_bars = plot_probability_bars(res["probabilities"])
                    st.pyplot(fig_bars)

            else:  # CLIP + Logistic Regression
                res = predict_clip(input_image)
                meta = res["description"]

                st.markdown(
                    f"""
                    <div class="navarasa-card" style="border-top: 5px solid {meta.get('color', '#1E88E5')};">
                        <span class="badge-green">CLIP ViT-B/32 PREDICTION</span>
                        <h1 style="margin: 8px 0; font-size: 2.2rem; color: {THEME['deep_green']};">{res['predicted_class']}</h1>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #1E88E5;">
                            Confidence: {res['confidence'] * 100:.1f}%
                        </div>
                        <p style="font-size: 0.95rem; color: {THEME['text_muted']}; margin: 8px 0;">
                            <b>Bhaava:</b> {meta.get('bhaava', '')} &bull; <b>Aesthetic:</b> {meta.get('meaning', '')}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_top3, col_bars = st.columns([1, 1.4])
                with col_top3:
                    st.markdown("### Top 3 Predictions")
                    for cls_name, prob in res["top_3"]:
                        st.write(f"**{cls_name}**: {prob * 100:.1f}%")
                        st.progress(float(prob))
                with col_bars:
                    fig_bars = plot_probability_bars(res["probabilities"])
                    st.pyplot(fig_bars)
