"""
Model Loader and Centralized Caching Module for Navarasa AI.
Optimized for Streamlit Community Cloud (sub-1GB memory constraint):
- Wraps model resources with Streamlit's @st.cache_resource for persistent cross-rerun sharing
- Implements lazy loading: models load ONLY when their specific feature is requested
- Provides a shared MediaPipe FaceLandmarker singleton across face detection, 6-region analysis, and AUs
- Enforces garbage collection after loading heavy weights to discard transient buffers
"""

import sys
import gc
from pathlib import Path
from typing import Optional
import torch

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    st = None
    HAS_STREAMLIT = False

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    DEVICE,
    CLIP_MODEL_PATH,
    EFFICIENTNET_BEST_MODEL_PATH,
    MODELS_DIR,
)

# In-memory fallback dictionary for non-Streamlit environments (unit tests, CLI)
_FALLBACK_CACHE = {}


def _cache_resource_decorator(func):
    """Applies st.cache_resource if running in Streamlit, otherwise returns plain function."""
    if HAS_STREAMLIT and hasattr(st, "cache_resource"):
        try:
            return st.cache_resource(show_spinner=False)(func)
        except Exception:
            return func
    return func


# 1. EfficientNet-B0 Model (Cached Singleton, ~60 MB)
@_cache_resource_decorator
def _load_efficientnet_cached(device_str: str = "cpu"):
    from src.models.efficientnet_model import load_efficientnet_model
    dev = torch.device(device_str)
    model = load_efficientnet_model(EFFICIENTNET_BEST_MODEL_PATH, device=dev)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    gc.collect()
    return model


def get_efficientnet_model(device=None):
    dev_str = str(device or DEVICE)
    cache_key = f"eff_{dev_str}"
    if HAS_STREAMLIT and hasattr(st, "cache_resource"):
        return _load_efficientnet_cached(dev_str)
    if cache_key not in _FALLBACK_CACHE:
        _FALLBACK_CACHE[cache_key] = _load_efficientnet_cached(dev_str)
    return _FALLBACK_CACHE[cache_key]


# 2. CLIP Vision Pipeline (Cached Singleton, Lazy Loaded Only on Demand)
@_cache_resource_decorator
def _load_clip_cached(device_str: str = "cpu", model_type: str = "svm"):
    from src.models.clip_model import CLIPClassifierPipeline
    dev = torch.device(device_str)
    pipeline = CLIPClassifierPipeline(device=dev)
    pipeline.load_clip_encoder()
    if model_type == "svm":
        target_path = MODELS_DIR / "clip_svm.joblib"
    else:
        target_path = MODELS_DIR / "clip_logistic_regression.joblib"
    if not target_path.exists():
        target_path = CLIP_MODEL_PATH
    if target_path.exists():
        pipeline.load_model(target_path)
    gc.collect()
    return pipeline


def get_clip_pipeline(device=None, model_type="svm"):
    dev_str = str(device or DEVICE)
    cache_key = f"clip_{dev_str}_{model_type}"
    if HAS_STREAMLIT and hasattr(st, "cache_resource"):
        return _load_clip_cached(dev_str, model_type=model_type)
    if cache_key not in _FALLBACK_CACHE:
        _FALLBACK_CACHE[cache_key] = _load_clip_cached(dev_str, model_type=model_type)
    return _FALLBACK_CACHE[cache_key]


def get_clip_svm_pipeline(device=None):
    return get_clip_pipeline(device=device, model_type="svm")


def get_clip_logreg_pipeline(device=None):
    return get_clip_pipeline(device=device, model_type="logreg")


get_clip_model = get_clip_pipeline



# 3. Shared MediaPipe FaceLandmarker Singleton (~120 MB vs ~360 MB for 3 duplicates)
@_cache_resource_decorator
def _load_shared_landmarker_cached(model_path_str: str):
    from pathlib import Path
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError:
        return None

    path = Path(model_path_str)
    if not path.exists():
        return None

    try:
        base_options = python.BaseOptions(model_asset_path=str(path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=4,
            min_face_detection_confidence=0.20,
            min_face_presence_confidence=0.20,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        gc.collect()
        return landmarker
    except Exception as e:
        print(f"Warning: Could not create shared FaceLandmarker: {e}")
        return None


def get_shared_face_landmarker(model_path: Optional[Path] = None):
    p = str(model_path or (MODELS_DIR / "face_landmarker.task"))
    cache_key = f"mp_landmarker_{p}"
    if HAS_STREAMLIT and hasattr(st, "cache_resource"):
        return _load_shared_landmarker_cached(p)
    if cache_key not in _FALLBACK_CACHE:
        _FALLBACK_CACHE[cache_key] = _load_shared_landmarker_cached(p)
    return _FALLBACK_CACHE[cache_key]


# 4. Grad-CAM Engine (Reuses cached EfficientNet)
@_cache_resource_decorator
def _load_gradcam_cached(device_str: str = "cpu"):
    from src.explainability.gradcam import GradCAM
    model = get_efficientnet_model(device=device_str)
    return GradCAM(model, device=torch.device(device_str))


def get_gradcam_engine(device=None):
    dev_str = str(device or DEVICE)
    cache_key = f"gradcam_{dev_str}"
    if HAS_STREAMLIT and hasattr(st, "cache_resource"):
        return _load_gradcam_cached(dev_str)
    if cache_key not in _FALLBACK_CACHE:
        _FALLBACK_CACHE[cache_key] = _load_gradcam_cached(dev_str)
    return _FALLBACK_CACHE[cache_key]


# 5. Facial Region Analyzer (Reuses shared FaceLandmarker)
@_cache_resource_decorator
def _load_facial_analyzer_cached():
    from src.explainability.facial_regions import FacialRegionAnalyzer
    shared_lm = get_shared_face_landmarker()
    return FacialRegionAnalyzer(shared_detector=shared_lm)


def get_facial_analyzer():
    cache_key = "facial_analyzer"
    if HAS_STREAMLIT and hasattr(st, "cache_resource"):
        return _load_facial_analyzer_cached()
    if cache_key not in _FALLBACK_CACHE:
        _FALLBACK_CACHE[cache_key] = _load_facial_analyzer_cached()
    return _FALLBACK_CACHE[cache_key]


# 6. Action Unit Extractor (Reuses Facial Region Analyzer)
@_cache_resource_decorator
def _load_au_extractor_cached():
    from src.explainability.action_units import ActionUnitExtractor
    analyzer = get_facial_analyzer()
    return ActionUnitExtractor(analyzer)


def get_au_extractor():
    cache_key = "au_extractor"
    if HAS_STREAMLIT and hasattr(st, "cache_resource"):
        return _load_au_extractor_cached()
    if cache_key not in _FALLBACK_CACHE:
        _FALLBACK_CACHE[cache_key] = _load_au_extractor_cached()
    return _FALLBACK_CACHE[cache_key]


# 7. Occlusion Analyzer (Reuses cached EfficientNet)
@_cache_resource_decorator
def _load_occlusion_analyzer_cached(device_str: str = "cpu"):
    from src.explainability.occlusion import OcclusionSensitivityAnalyzer
    model = get_efficientnet_model(device=device_str)
    return OcclusionSensitivityAnalyzer(model, device=torch.device(device_str))


def get_occlusion_analyzer(device=None):
    dev_str = str(device or DEVICE)
    cache_key = f"occlusion_{dev_str}"
    if HAS_STREAMLIT and hasattr(st, "cache_resource"):
        return _load_occlusion_analyzer_cached(dev_str)
    if cache_key not in _FALLBACK_CACHE:
        _FALLBACK_CACHE[cache_key] = _load_occlusion_analyzer_cached(dev_str)
    return _FALLBACK_CACHE[cache_key]


# 8. Explainability Logic Helpers
def get_top_two_comparator():
    if "comparator" not in _FALLBACK_CACHE:
        from src.explainability.counterfactual import TopTwoComparator
        _FALLBACK_CACHE["comparator"] = TopTwoComparator()
    return _FALLBACK_CACHE["comparator"]


def get_counterfactual_analyzer():
    if "counterfactual" not in _FALLBACK_CACHE:
        from src.explainability.counterfactual import CounterfactualFeatureAnalyzer
        _FALLBACK_CACHE["counterfactual"] = CounterfactualFeatureAnalyzer()
    return _FALLBACK_CACHE["counterfactual"]


def get_natural_language_explainer():
    if "explainer" not in _FALLBACK_CACHE:
        from src.explainability.explanations import NaturalLanguageExplainer
        _FALLBACK_CACHE["explainer"] = NaturalLanguageExplainer()
    return _FALLBACK_CACHE["explainer"]
