"""
Visualization and UI styling utilities for Navarasa AI.
Implements Indian Classical aesthetic (Deep Emerald, Warm Ivory, Regal Gold),
high-performance charts using Altair / Plotly / Matplotlib, and custom CSS injection.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import NAVARASA_CLASSES, RASA_DESCRIPTIONS

# Classical Indian Aesthetic Palette
THEME = {
    "deep_green": "#0B3026",
    "forest_green": "#124E3F",
    "soft_sage": "#D8E2DC",
    "regal_gold": "#C5A059",
    "bright_gold": "#D4AF37",
    "warm_ivory": "#FAF8F5",
    "card_bg": "#FFFFFF",
    "text_dark": "#1A202C",
    "text_muted": "#4A5568",
    "border_gold": "#E2D3B3",
}


def get_custom_css():
    """
    Returns custom CSS for Streamlit web app in Indian classical aesthetic.
    """
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Palette */
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, sans-serif;
        color: {THEME['text_dark']};
    }}

    .stApp {{
        background-color: {THEME['warm_ivory']};
    }}

    /* Classical Headers */
    h1, h2, h3 {{
        font-family: 'Cinzel', serif !important;
        color: {THEME['deep_green']} !important;
        letter-spacing: 0.5px;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {THEME['deep_green']} !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: #F7FAFC !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {THEME['bright_gold']} !important;
    }}

    /* Stat Cards & Metric Containers */
    .navarasa-card {{
        background: {THEME['card_bg']};
        border-radius: 12px;
        padding: 24px;
        border: 1px solid {THEME['border_gold']};
        box-shadow: 0 4px 15px rgba(11, 48, 38, 0.05);
        margin-bottom: 20px;
    }}

    .navarasa-header-card {{
        background: linear-gradient(135deg, {THEME['deep_green']} 0%, {THEME['forest_green']} 100%);
        border-radius: 14px;
        padding: 28px 32px;
        color: #FFFFFF !important;
        border-bottom: 3px solid {THEME['bright_gold']};
        box-shadow: 0 6px 20px rgba(11, 48, 38, 0.15);
        margin-bottom: 25px;
    }}
    .navarasa-header-card h1, .navarasa-header-card h2, .navarasa-header-card p {{
        color: #FFFFFF !important;
    }}

    .badge-gold {{
        background: linear-gradient(135deg, {THEME['regal_gold']} 0%, {THEME['bright_gold']} 100%);
        color: #1A202C !important;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        display: inline-block;
        letter-spacing: 0.5px;
    }}

    .badge-green {{
        background-color: {THEME['forest_green']};
        color: #FFFFFF !important;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        display: inline-block;
    }}

    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, {THEME['forest_green']} 0%, {THEME['deep_green']} 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: 1px solid {THEME['regal_gold']} !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        transition: all 0.25s ease-in-out !important;
        box-shadow: 0 2px 8px rgba(11, 48, 38, 0.2) !important;
    }}
    .stButton>button:hover {{
        background: linear-gradient(135deg, {THEME['regal_gold']} 0%, {THEME['bright_gold']} 100%) !important;
        color: {THEME['deep_green']} !important;
        border-color: {THEME['deep_green']} !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(197, 160, 89, 0.35) !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: {THEME['text_dark']};
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {THEME['card_bg']} !important;
        border-top: 3px solid {THEME['bright_gold']} !important;
        color: {THEME['deep_green']} !important;
        font-weight: 700 !important;
    }}
    </style>
    """


def plot_probability_bars(probabilities, title="Navarasa Class Probabilities (%)"):
    """Generates a horizontal bar chart of 9-class probabilities."""
    classes = list(probabilities.keys())
    scores = [probabilities[c] * 100 for c in classes]

    # Colors per rasa
    colors = [RASA_DESCRIPTIONS.get(c, {}).get("color", "#1976D2") for c in classes]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    y_pos = np.arange(len(classes))

    bars = ax.barh(y_pos, scores, color=colors, edgecolor="#222", height=0.6, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=10, fontweight="bold")
    ax.invert_yaxis()  # Top class at the top
    ax.set_xlabel("Probability (%)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_xlim(0, 105)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.1f}%",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_dual_probability_comparison_chart(clip_probs, eff_probs):
    """Generates side-by-side grouped horizontal bar chart for CLIP vs EfficientNet."""
    classes = NAVARASA_CLASSES
    y_pos = np.arange(len(classes))
    height = 0.35

    clip_vals = [clip_probs.get(c, 0.0) * 100 for c in classes]
    eff_vals = [eff_probs.get(c, 0.0) * 100 for c in classes]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    bars1 = ax.barh(y_pos - height / 2, clip_vals, height, label="CLIP ViT-B/32 + LogReg", color="#1E88E5", alpha=0.85)
    bars2 = ax.barh(y_pos + height / 2, eff_vals, height, label="EfficientNet-B0", color="#43A047", alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Probability (%)", fontsize=10)
    ax.set_title("Dual-Model Prediction Probability Comparison", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0, 105)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_region_importance_chart(region_pcts, title="Facial Region Grad-CAM Activation Energy (%)"):
    """Generates bar chart for facial region activation percentages."""
    regions = list(region_pcts.keys())
    scores = [region_pcts[r] for r in regions]

    colors = ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#FF7043", "#AB47BC"]

    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=200)
    bars = ax.bar(regions, scores, color=colors[:len(regions)], edgecolor="#333", width=0.55)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("Activation Energy (%)", fontsize=10)
    ax.set_ylim(0, max(scores, default=50) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.xticks(rotation=20, ha="right", fontsize=9, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_action_units_chart(action_units):
    """Generates bar chart of detected FACS Action Unit intensities."""
    au_ids = list(action_units.keys())
    intensities = [action_units[au]["intensity"] for au in au_ids]
    colors = ["#D32F2F" if action_units[au]["is_active"] else "#90A4AE" for au in au_ids]

    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    bars = ax.bar(au_ids, intensities, color=colors, edgecolor="#333", width=0.6)
    ax.set_title("Facial Action Unit (FACS) Intensities & Activation (Red = Active)", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("Intensity (0.0 - 1.0)", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.15, color="#D32F2F", linestyle="--", alpha=0.7, label="Activation Threshold (0.15)")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    return fig


def plot_occlusion_sensitivity_chart(region_results):
    """Generates bar chart showing confidence drop when each region is masked."""
    regions = list(region_results.keys())
    drops = [region_results[r]["confidence_drop"] * 100 for r in regions]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    bars = ax.bar(regions, drops, color="#E65100", edgecolor="#333", width=0.55)
    ax.set_title("Occlusion Sensitivity: Confidence Drop per Masked Region (%)", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("Confidence Drop (%)", fontsize=10)
    ax.set_ylim(0, max(drops, default=20) * 1.3)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.xticks(rotation=20, ha="right", fontsize=9, fontweight="bold")
    plt.tight_layout()
    return fig
