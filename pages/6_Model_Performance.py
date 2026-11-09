"""
Page 6: Model Performance — Benchmarks, Metrics, Confusion Matrices, Confusing Pairs Analysis, and Dataset Size Scaling
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
        <h1 style="margin-top: 10px;">Model Evaluation & Benchmark Comparison</h1>
        <p style="opacity: 0.95; font-size: 1.05rem;">
            Rigorous evaluation on the held-out test split of validated facial crops.
            Comparing Model A (CLIP + LogReg), Model B (CLIP + SVM RBF), and Model C (Fine-Tuned EfficientNet-B0).
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
size_data = {}
if DATASET_SIZE_EXPERIMENT_PATH.exists():
    with open(DATASET_SIZE_EXPERIMENT_PATH, "r") as f:
        size_data = json.load(f)

tab_overview, tab_perclass, tab_matrices, tab_pairs, tab_curves, tab_scaling = st.tabs([
    "🏆 Model Comparison",
    "📊 Per-Class Breakdown",
    "🔲 Confusion Matrices",
    "🎯 Confusing Pairs Analysis",
    "📉 Training History Curves",
    "📏 Dataset Size Scaling Experiments"
])

with tab_overview:
    if eval_data:
        # Check if 3-model format or legacy 2-model format
        if "models" in eval_data:
            m_a = eval_data["models"].get("model_a_clip_logreg", {})
            m_b = eval_data["models"].get("model_b_clip_svm", {})
            m_c = eval_data["models"].get("model_c_efficientnet_b0", {})

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model B: CLIP + SVM (Primary)", f"{m_b.get('accuracy', 0.0) * 100:.2f}%", f"Macro F1: {m_b.get('macro_f1', 0.0) * 100:.2f}%")
            with col2:
                st.metric("Model A: CLIP + LogReg", f"{m_a.get('accuracy', 0.0) * 100:.2f}%", f"Macro F1: {m_a.get('macro_f1', 0.0) * 100:.2f}%")
            with col3:
                st.metric("Model C: EfficientNet-B0", f"{m_c.get('accuracy', 0.0) * 100:.2f}%", f"Macro F1: {m_c.get('macro_f1', 0.0) * 100:.2f}%")

            st.markdown("### Overall Multi-Model Benchmark Summary Table")
            overview_table = [
                {
                    "Metric": "Test Accuracy",
                    "Model A (CLIP + LogReg)": f"{m_a.get('accuracy', 0.0) * 100:.2f}%",
                    "Model B (CLIP + SVM)": f"{m_b.get('accuracy', 0.0) * 100:.2f}%",
                    "Model C (EfficientNet-B0)": f"{m_c.get('accuracy', 0.0) * 100:.2f}%",
                },
                {
                    "Metric": "Macro Precision",
                    "Model A (CLIP + LogReg)": f"{m_a.get('macro_precision', 0.0) * 100:.2f}%",
                    "Model B (CLIP + SVM)": f"{m_b.get('macro_precision', 0.0) * 100:.2f}%",
                    "Model C (EfficientNet-B0)": f"{m_c.get('macro_precision', 0.0) * 100:.2f}%",
                },
                {
                    "Metric": "Macro Recall",
                    "Model A (CLIP + LogReg)": f"{m_a.get('macro_recall', 0.0) * 100:.2f}%",
                    "Model B (CLIP + SVM)": f"{m_b.get('macro_recall', 0.0) * 100:.2f}%",
                    "Model C (EfficientNet-B0)": f"{m_c.get('macro_recall', 0.0) * 100:.2f}%",
                },
                {
                    "Metric": "Macro F1-Score",
                    "Model A (CLIP + LogReg)": f"{m_a.get('macro_f1', 0.0) * 100:.2f}%",
                    "Model B (CLIP + SVM)": f"{m_b.get('macro_f1', 0.0) * 100:.2f}%",
                    "Model C (EfficientNet-B0)": f"{m_c.get('macro_f1', 0.0) * 100:.2f}%",
                },
                {
                    "Metric": "Weighted F1-Score",
                    "Model A (CLIP + LogReg)": f"{m_a.get('weighted_f1', 0.0) * 100:.2f}%",
                    "Model B (CLIP + SVM)": f"{m_b.get('weighted_f1', 0.0) * 100:.2f}%",
                    "Model C (EfficientNet-B0)": f"{m_c.get('weighted_f1', 0.0) * 100:.2f}%",
                },
                {
                    "Metric": "Inference Latency",
                    "Model A (CLIP + LogReg)": f"{m_a.get('latency_ms_per_sample', 0):.2f} ms",
                    "Model B (CLIP + SVM)": f"{m_b.get('latency_ms_per_sample', 0):.2f} ms",
                    "Model C (EfficientNet-B0)": f"{m_c.get('latency_ms_per_sample', 0):.2f} ms",
                },
            ]
            st.dataframe(overview_table, use_container_width=True, hide_index=True)

            if "benchmark_summary" in eval_data and "selection_rationale" in eval_data["benchmark_summary"]:
                st.info(f"**Primary Model Selection:** {eval_data['benchmark_summary']['selected_primary_model']}\n\n"
                        f"{eval_data['benchmark_summary']['selection_rationale']}")
        else:
            clip_m = eval_data.get("clip_logreg", {})
            eff_m = eval_data.get("efficientnet_b0", {})
            c1, c2 = st.columns(2)
            with c1:
                st.metric("CLIP ViT-B/32 Test Acc", f"{clip_m.get('accuracy', 0.0) * 100:.2f}%")
            with c2:
                st.metric("EfficientNet-B0 Test Acc", f"{eff_m.get('accuracy', 0.0) * 100:.2f}%")

        comp_plot = PLOTS_DIR / "model_comparison_bar.png"
        if comp_plot.exists():
            st.image(str(comp_plot), use_container_width=True)
    else:
        st.info("Evaluation metrics report will appear here once evaluation script completes.")

with tab_perclass:
    st.markdown("### Per-Class Performance Breakdown Across All 9 Navarasas")
    if eval_data:
        if "models" in eval_data:
            m_a = eval_data["models"].get("model_a_clip_logreg", {})
            m_b = eval_data["models"].get("model_b_clip_svm", {})
            m_c = eval_data["models"].get("model_c_efficientnet_b0", {})

            class_rows = []
            for cls in NAVARASA_CLASSES:
                pa = m_a.get("per_class", {}).get(cls, {})
                pb = m_b.get("per_class", {}).get(cls, {})
                pc = m_c.get("per_class", {}).get(cls, {})
                class_rows.append({
                    "Navarasa Class": cls,
                    "Support": pa.get("support", 0),
                    "Model A (LogReg) F1": f"{pa.get('f1_score', 0.0) * 100:.1f}%",
                    "Model B (SVM) F1": f"{pb.get('f1_score', 0.0) * 100:.1f}%",
                    "Model C (EffNet) F1": f"{pc.get('f1_score', 0.0) * 100:.1f}%",
                    "Model B Precision": f"{pb.get('precision', 0.0) * 100:.1f}%",
                    "Model B Recall": f"{pb.get('recall', 0.0) * 100:.1f}%",
                })
            st.dataframe(class_rows, use_container_width=True, hide_index=True)
        else:
            clip_m = eval_data.get("clip_logreg", {})
            eff_m = eval_data.get("efficientnet_b0", {})
            class_rows = []
            for cls in NAVARASA_CLASSES:
                cp = clip_m.get("per_class", {}).get(cls, {})
                ep = eff_m.get("per_class", {}).get(cls, {})
                class_rows.append({
                    "Navarasa Class": cls,
                    "CLIP F1": f"{cp.get('f1_score', 0.0) * 100:.1f}%",
                    "EffNet F1": f"{ep.get('f1_score', 0.0) * 100:.1f}%",
                })
            st.dataframe(class_rows, use_container_width=True, hide_index=True)

with tab_matrices:
    st.markdown("### Normalized Confusion Matrix Heatmaps")
    cm_c1, cm_c2, cm_c3 = st.columns(3)
    with cm_c1:
        st.markdown("#### Model B: CLIP + SVM (RBF)")
        cm_b = PLOTS_DIR / "clip_svm_confusion_matrix_norm.png"
        if cm_b.exists():
            st.image(str(cm_b), use_container_width=True)
        else:
            st.info("Generating Model B confusion matrix...")

    with cm_c2:
        st.markdown("#### Model A: CLIP + LogReg")
        cm_a = PLOTS_DIR / "clip_logreg_confusion_matrix_norm.png"
        if not cm_a.exists():
            cm_a = PLOTS_DIR / "clip_confusion_matrix.png"
        if cm_a.exists():
            st.image(str(cm_a), use_container_width=True)
        else:
            st.info("Generating Model A confusion matrix...")

    with cm_c3:
        st.markdown("#### Model C: EfficientNet-B0")
        cm_c = PLOTS_DIR / "efficientnet_confusion_matrix_norm.png"
        if not cm_c.exists():
            cm_c = PLOTS_DIR / "efficientnet_confusion_matrix.png"
        if cm_c.exists():
            st.image(str(cm_c), use_container_width=True)
        else:
            st.info("Generating Model C confusion matrix...")

with tab_pairs:
    st.markdown("### Historically Problematic Class Pairs Deep-Dive")
    st.markdown(
        "Analysis of directional misclassifications and mutual error rates on held-out test crops."
    )
    if eval_data and "models" in eval_data:
        m_b = eval_data["models"].get("model_b_clip_svm", {})
        pairs_dict = m_b.get("confusing_pairs_analysis", {})
        if pairs_dict:
            for p_key, p_info in pairs_dict.items():
                cls_a = p_info["class_a"]
                cls_b = p_info["class_b"]
                st.markdown(f"#### 🔍 {cls_a} vs. {cls_b}")
                st.write(f"- True {cls_a} samples: **{p_info['total_true_a']}** | True {cls_b} samples: **{p_info['total_true_b']}**")
                st.write(f"- Direction 1 ({cls_a} predicted as {cls_b}): **{p_info['a_misclassified_as_b_count']}** ({p_info['a_misclassified_as_b_percent']}%)")
                st.write(f"- Direction 2 ({cls_b} predicted as {cls_a}): **{p_info['b_misclassified_as_a_count']}** ({p_info['b_misclassified_as_a_percent']}%)")
                st.write(f"- Total mutual error rate: **{p_info['mutual_confusion_rate_percent']}%**")
                st.markdown("---")
    else:
        st.info("Confusion pairs analysis will display after full evaluation completes.")

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
        "maintaining validation and test sets completely fixed on facial crops."
    )

    scale_plot = PLOTS_DIR / "dataset_size_vs_performance.png"
    if scale_plot.exists():
        st.image(str(scale_plot), use_container_width=True)

    if size_data and "svm" in size_data:
        st.markdown("#### Quantitative Dataset Size Experiment Results (Model B: SVM vs Model A: LogReg)")
        size_table = []
        for r_svm, r_lr in zip(size_data.get("svm", []), size_data.get("logreg", [])):
            size_table.append({
                "Setting": f"{r_svm['size_setting']} / class",
                "Total Train Samples": f"{r_svm['total_training_samples']:,}",
                "Model B (SVM) Acc": f"{r_svm['test_accuracy'] * 100:.2f}%",
                "Model B (SVM) F1": f"{r_svm['test_macro_f1'] * 100:.2f}%",
                "Model A (LogReg) Acc": f"{r_lr['test_accuracy'] * 100:.2f}%",
                "Model A (LogReg) F1": f"{r_lr['test_macro_f1'] * 100:.2f}%",
            })
        st.dataframe(size_table, use_container_width=True, hide_index=True)
    elif size_data and isinstance(size_data, list):
        size_table = []
        for row in size_data:
            size_table.append({
                "Setting": f"{row['size_setting']} / class",
                "Total Train Samples": f"{row['total_training_samples']:,}",
                "Test Accuracy": f"{row['test_accuracy'] * 100:.2f}%",
                "Macro F1-Score": f"{row['test_macro_f1'] * 100:.2f}%",
            })
        st.dataframe(size_table, use_container_width=True, hide_index=True)
