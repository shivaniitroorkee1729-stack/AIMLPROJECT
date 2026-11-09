"""
Navarasa AI Configuration and Global Settings
"""

import os
from pathlib import Path
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

# Ensure runtime directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, MODELS_DIR, REPORTS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Nine Canonical Navarasas in standard order
NAVARASA_CLASSES = [
    "Shringara",
    "Hasya",
    "Karuna",
    "Raudra",
    "Veera",
    "Bhayanaka",
    "Bibhatsa",
    "Adbhuta",
    "Shanta",
]

# Mapping from raw folder/archive names to official class names
RAW_CLASS_MAPPING = {
    "shringara": "Shringara",
    "hasya": "Hasya",
    "karuna": "Karuna",
    "raudra": "Raudra",
    "veer": "Veera",
    "veera": "Veera",
    "bhayanak": "Bhayanaka",
    "bhayanaka": "Bhayanaka",
    "bhibhatsa": "Bibhatsa",
    "bibhatsa": "Bibhatsa",
    "adhbuta": "Adbhuta",
    "adbhuta": "Adbhuta",
    "shanta": "Shanta",
}

# English sentiment and classical meaning descriptions
RASA_DESCRIPTIONS = {
    "Shringara": {
        "rasa": "Shringara",
        "bhaava": "Rati (Delight / Love)",
        "meaning": "Love, beauty, attractiveness, devotion, and romantic aesthetic.",
        "visual_cues": "Soft warm smile, relaxed open facial muscles, gentle expressive eye gaze.",
        "color": "#E57373",
        "deity": "Vishnu",
    },
    "Hasya": {
        "rasa": "Hasya",
        "bhaava": "Hasa (Mirth / Laughter)",
        "meaning": "Joy, laughter, humor, cheerful celebration, and comedic amusement.",
        "visual_cues": "Elevated lip corners, broad open teeth, raised cheek apple (AU6), crinkled outer eye corners.",
        "color": "#81C784",
        "deity": "Pramathas",
    },
    "Karuna": {
        "rasa": "Karuna",
        "bhaava": "Shoka (Sorrow / Grief)",
        "meaning": "Sorrow, compassion, empathy, tragedy, and sympathetic sadness.",
        "visual_cues": "Downturned lip corners, elevated inner eyebrows (AU1), subdued lowered eyelids, drooping facial tension.",
        "color": "#64B5F6",
        "deity": "Yama",
    },
    "Raudra": {
        "rasa": "Raudra",
        "bhaava": "Krodha (Anger / Wrath)",
        "meaning": "Anger, fury, rage, fierce indignation, and tempestuous power.",
        "visual_cues": "Lowered and strongly furrowed eyebrows (AU4), intense wide glaring eyes, pressed or clenched lips.",
        "color": "#E53935",
        "deity": "Rudra",
    },
    "Veera": {
        "rasa": "Veera",
        "bhaava": "Utsaha (Enthusiasm / Valor)",
        "meaning": "Heroism, courage, chivalry, majestic valor, and unwavering confidence.",
        "visual_cues": "Firm upright posture, set jaw, elevated high eyebrows, determined steady gaze, subtle poised tension.",
        "color": "#FFB74D",
        "deity": "Indra",
    },
    "Bhayanaka": {
        "rasa": "Bhayanaka",
        "bhaava": "Bhaya (Fear / Terror)",
        "meaning": "Fear, apprehension, terror, anxiety, and trembling dread.",
        "visual_cues": "Extremely widened eyelids exposing sclera (AU5), raised drawn brows (AU1+2), parted tense lips.",
        "color": "#4A148C",
        "deity": "Kala",
    },
    "Bibhatsa": {
        "rasa": "Bibhatsa",
        "bhaava": "Jugupsa (Disgust / Aversion)",
        "meaning": "Disgust, revulsion, abhorrence, aversion, and bitter contempt.",
        "visual_cues": "Strongly wrinkled nose bridge (AU9), raised upper lip showing canine/incisors (AU10), squinted eyes.",
        "color": "#00897B",
        "deity": "Mahakala",
    },
    "Adbhuta": {
        "rasa": "Adbhuta",
        "bhaava": "Vismaya (Wonder / Astonishment)",
        "meaning": "Wonder, surprise, astonishment, awe, and marveling curiosity.",
        "visual_cues": "Arched raised curved eyebrows, rounded open wide eyes (AU5), relaxed open drop jaw (AU26/27).",
        "color": "#FFD54F",
        "deity": "Brahma",
    },
    "Shanta": {
        "rasa": "Shanta",
        "bhaava": "Sama (Tranquility / Peace)",
        "meaning": "Peace, tranquility, calm, serenity, meditative detachment, and spiritual harmony.",
        "visual_cues": "Neutral balanced lips, smoothly relaxed forehead and brows, serene soft gaze, complete absence of facial strain.",
        "color": "#A1887F",
        "deity": "Vishnu (Narayana)",
    },
}

CLASS_TO_IDX = {name: idx for idx, name in enumerate(NAVARASA_CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(NAVARASA_CLASSES)}

# Device configuration (CUDA auto-detect, fallback to CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configurations
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
IMAGE_SIZE_EFFICIENTNET = 224
IMAGE_SIZE_CLIP = 224

# Model artifact paths
CLIP_MODEL_PATH = MODELS_DIR / "clip_logistic_regression.joblib"
CLIP_EMBEDDINGS_CACHE = MODELS_DIR / "clip_embeddings_cache.npz"
EFFICIENTNET_BEST_MODEL_PATH = MODELS_DIR / "efficientnet_b0_best.pt"
EFFICIENTNET_METRICS_PATH = REPORTS_DIR / "efficientnet_training_history.json"
EVALUATION_METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"
DATASET_SIZE_EXPERIMENT_PATH = REPORTS_DIR / "dataset_size_experiment_results.json"
DATASET_ANALYSIS_PATH = REPORTS_DIR / "dataset_analysis.json"
