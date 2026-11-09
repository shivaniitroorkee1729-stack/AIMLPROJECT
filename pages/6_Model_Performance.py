"""
Page 6: Model Performance — Benchmarks, Metrics, Confusion Matrices, and Dataset Size Scaling
"""

import sys
import json
from pathlib import Path
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    REPORTS_DIR,
    PLOTS_DIR,
    NAVARASA_CLASSES,
    EVALUATION_METRICS_PATH,
    DATASET_SIZE_EXPERIMENT_PATH,
)
from src.utils.visualization import get_custom_css, THEME

st.set_page_config(page_title="Model Performance — NAVARASA AI", page_icon="📈", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.markdown(
    """
    <div class="navarasa-header-card">
        <span class="badge-gold">EMPIRICAL BENCHMARKS & EVALUATION</span>
        <h1 style="margin-top: 10px;">Model Evaluation & Comparison</h1>
        <p style="opacity: 0.95; font-size: 1.05rem;">
            Rigorous evaluation on the held-out test split of 2,115 unseen images.
            Comparing CLIP ViT-B/32 + Logistic Regression against fine-tuned EfficientNet-B0.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load evaluation metrics if available
eval_data = {}
if EVALUATION_METRICS_PATH.exists():
    with open(EVALUATION_METRICS_PATH, "r") as f:
        eval_data = json.load(f)

# Load dataset size results if available
size_data = []
if DATASET_SIZE_EXPERIMENT_PATH.exists():
    with open(DATASET_SIZE_EXPERIMENT_PATH, "r") as f:
        size_data = json.load(f)

tab_overview, tab_perclass, tab_matrices, tab_curves, tab_scaling = st.tabs([
    "🏆 Model Comparison",
    "📊 Per-Class Breakdown",
    "🔲 Confusion Matrices",
    "📉 Training History Curves",
    "📏 Dataset Size Scaling Experiments"
])

with tab_overview:
    if eval_data:
        clip_m = eval_data.get("clip_logreg", {})
        eff_m = eval_data.get("efficientnet_b0", {})

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("EfficientNet-B0 Test Acc", f"{eff_m.get('accuracy', 0.0) * 100:.2f}%")
        with c2:
            st.metric("CLIP ViT-B/32 Test Acc", f"{clip_m.get('accuracy', 0.0) * 100:.2f}%")
        with c3:
            st.metric("EfficientNet Macro F1", f"{eff_m.get('macro_f1', 0.0) * 100:.2f}%")
        with c4:
            st.metric("CLIP Macro F1", f"{clip_m.get('macro_f1', 0.0) * 100:.2f}%")

        st.markdown("### Overall Performance Summary Table")
        diff_acc = (eff_m.get("accuracy", 0.0) - clip_m.get("accuracy", 0.0)) * 100
        diff_f1 = (eff_m.get("macro_f1", 0.0) - clip_m.get("macro_f1", 0.0)) * 100

        overview_table = [
            {
                "Metric": "Test Accuracy",
                "CLIP ViT-B/32 + LogReg": f"{clip_m.get('accuracy', 0.0) * 100:.2f}%",
                "EfficientNet-B0 (Fine-Tuned)": f"{eff_m.get('accuracy', 0.0) * 100:.2f}%",
                "Difference": f"{diff_acc:+.2f}%",
            },
            {
                "Metric": "Macro Precision",
                "CLIP ViT-B/32 + LogReg": f"{clip_m.get('macro_precision', 0.0) * 100:.2f}%",
                "EfficientNet-B0 (Fine-Tuned)": f"{eff_m.get('macro_precision', 0.0) * 100:.2f}%",
                "Difference": f"{(eff_m.get('macro_precision', 0.0) - clip_m.get('macro_precision', 0.0)) * 100:+.2f}%",
            },
            {
                "Metric": "Macro Recall",
                "CLIP ViT-B/32 + LogReg": f"{clip_m.get('macro_recall', 0.0) * 100:.2f}%",
                "EfficientNet-B0 (Fine-Tuned)": f"{eff_m.get('macro_recall', 0.0) * 100:.2f}%",
                "Difference": f"{(eff_m.get('macro_recall', 0.0) - clip_m.get('macro_recall', 0.0)) * 100:+.2f}%",
            },
            {
                "Metric": "Macro F1-Score",
                "CLIP ViT-B/32 + LogReg": f"{clip_m.get('macro_f1', 0.0) * 100:.2f}%",
                "EfficientNet-B0 (Fine-Tuned)": f"{eff_m.get('macro_f1', 0.0) * 100:.2f}%",
                "Difference": f"{diff_f1:+.2f}%",
            },
            {
                "Metric": "Weighted F1-Score",
                "CLIP ViT-B/32 + LogReg": f"{clip_m.get('weighted_f1', 0.0) * 100:.2f}%",
                "EfficientNet-B0 (Fine-Tuned)": f"{eff_m.get('weighted_f1', 0.0) * 100:.2f}%",
                "Difference": f"{(eff_m.get('weighted_f1', 0.0) - clip_m.get('weighted_f1', 0.0)) * 100:+.2f}%",
            },
        ]
        st.dataframe(overview_table, use_container_width=True, hide_index=True)

        # Comparison chart
        comp_plot = PLOTS_DIR / "model_comparison_bar.png"
        if comp_plot.exists():
            st.image(str(comp_plot), use_container_width=True)
    else:
        st.info("Evaluation metrics report will appear here once evaluation script finishes.")

with tab_perclass:
    st.markdown("### Per-Class Performance Breakdown Across All 9 Navarasas")
    if eval_data:
        clip_m = eval_data.get("clip_logreg", {})
        eff_m = eval_data.get("efficientnet_b0", {})

        class_rows = []
        for cls in NAVARASA_CLASSES:
            cp = clip_m.get("per_class", {}).get(cls, {})
            ep = eff_m.get("per_class", {}).get(cls, {})
            class_rows.append({
                "Navarasa Class": cls,
                "CLIP Precision": f"{cp.get('precision', 0.0) * 100:.1f}%",
                "CLIP Recall": f"{cp.get('recall', 0.0) * 100:.1f}%",
                "CLIP F1": f"{cp.get('f1_score', 0.0) * 100:.1f}%",
                "EffNet Precision": f"{ep.get('precision', 0.0) * 100:.1f}%",
                "EffNet Recall": f"{ep.get('recall', 0.0) * 100:.1f}%",
                "EffNet F1": f"{ep.get('f1_score', 0.0) * 100:.1f}%",
                "Support (Test)": cp.get("support", 0),
            })
        st.dataframe(class_rows, use_container_width=True, hide_index=True)

with tab_matrices:
    st.markdown("### Normalized Confusion Matrix Heatmaps")
    cm_c1, cm_c2 = st.columns(2)
    with cm_c1:
        st.markdown("#### CLIP ViT-B/32 + Logistic Regression")
        cm_clip_path = PLOTS_DIR / "clip_confusion_matrix.png"
        if cm_clip_path.exists():
            st.image(str(cm_clip_path), use_container_width=True)
        else:
            st.info("CLIP confusion matrix generating...")

    with cm_c2:
        st.markdown("#### EfficientNet-B0 (Fine-Tuned)")
        cm_eff_path = PLOTS_DIR / "efficientnet_confusion_matrix.png"
        if cm_eff_path.exists():
            st.image(str(cm_eff_path), use_container_width=True)
        else:
            st.info("EfficientNet confusion matrix generating...")

with tab_curves:
    st.markdown("### Training & Validation Dynamics")
    curves_path = PLOTS_DIR / "efficientnet_training_curves.png"
    if curves_path.exists():
        st.image(str(curves_path), use_container_width=True)
    else:
        st.info("Training curves will appear after EfficientNet training completes.")

with tab_scaling:
    st.markdown("### Influence of Training Dataset Size on Model Performance")
    st.markdown(
        "Investigates sample efficiency across **50, 100, 200, and Full samples per class**, "
        "maintaining validation and test sets completely fixed."
    )

    scale_plot = PLOTS_DIR / "dataset_size_vs_performance.png"
    if scale_plot.exists():
        st.image(str(scale_plot), use_container_width=True)

    if size_data:
        st.markdown("#### Quantitative Dataset Size Experiment Results")
        size_table = []
        for row in size_data:
            size_table.append({
                "Setting": f"{row['size_setting']} / class",
                "Total Train Samples": f"{row['total_training_samples']:,}",
                "Test Accuracy": f"{row['test_accuracy'] * 100:.2f}%",
                "Macro F1-Score": f"{row['test_macro_f1'] * 100:.2f}%",
                "Weighted F1-Score": f"{row['test_weighted_f1'] * 100:.2f}%",
            })
        st.dataframe(size_table, use_container_width=True, hide_index=True)
