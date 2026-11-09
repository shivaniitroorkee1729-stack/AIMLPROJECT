"""
Model Loader and Caching Module.
Provides centralized, cached loading for CLIP, EfficientNet-B0,
MediaPipe FaceLandmarker, and Explainability engines with lazy imports.
"""

import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    DEVICE,
    CLIP_MODEL_PATH,
    EFFICIENTNET_BEST_MODEL_PATH,
)

# Global singleton cache
_MODEL_CACHE = {}


def get_clip_pipeline(device=DEVICE):
    if "clip" not in _MODEL_CACHE:
        from src.models.clip_model import CLIPClassifierPipeline
        pipeline = CLIPClassifierPipeline(device=device)
        if CLIP_MODEL_PATH.exists():
            pipeline.load_model(CLIP_MODEL_PATH)
        _MODEL_CACHE["clip"] = pipeline
    return _MODEL_CACHE["clip"]


def get_efficientnet_model(device=DEVICE):
    if "efficientnet" not in _MODEL_CACHE:
        from src.models.efficientnet_model import load_efficientnet_model
        model = load_efficientnet_model(EFFICIENTNET_BEST_MODEL_PATH, device=device)
        _MODEL_CACHE["efficientnet"] = model
    return _MODEL_CACHE["efficientnet"]


def get_gradcam_engine(device=DEVICE):
    if "gradcam" not in _MODEL_CACHE:
        from src.explainability.gradcam import GradCAM
        eff_model = get_efficientnet_model(device=device)
        _MODEL_CACHE["gradcam"] = GradCAM(eff_model, device=device)
    return _MODEL_CACHE["gradcam"]


def get_facial_analyzer():
    if "facial_analyzer" not in _MODEL_CACHE:
        from src.explainability.facial_regions import FacialRegionAnalyzer
        _MODEL_CACHE["facial_analyzer"] = FacialRegionAnalyzer()
    return _MODEL_CACHE["facial_analyzer"]


def get_au_extractor():
    if "au_extractor" not in _MODEL_CACHE:
        from src.explainability.action_units import ActionUnitExtractor
        analyzer = get_facial_analyzer()
        _MODEL_CACHE["au_extractor"] = ActionUnitExtractor(analyzer)
    return _MODEL_CACHE["au_extractor"]


def get_occlusion_analyzer(device=DEVICE):
    if "occlusion" not in _MODEL_CACHE:
        from src.explainability.occlusion import OcclusionSensitivityAnalyzer
        eff_model = get_efficientnet_model(device=device)
        _MODEL_CACHE["occlusion"] = OcclusionSensitivityAnalyzer(eff_model, device=device)
    return _MODEL_CACHE["occlusion"]


def get_top_two_comparator():
    if "comparator" not in _MODEL_CACHE:
        from src.explainability.counterfactual import TopTwoComparator
        _MODEL_CACHE["comparator"] = TopTwoComparator()
    return _MODEL_CACHE["comparator"]


def get_counterfactual_analyzer():
    if "counterfactual" not in _MODEL_CACHE:
        from src.explainability.counterfactual import CounterfactualFeatureAnalyzer
        _MODEL_CACHE["counterfactual"] = CounterfactualFeatureAnalyzer()
    return _MODEL_CACHE["counterfactual"]


def get_natural_language_explainer():
    if "explainer" not in _MODEL_CACHE:
        from src.explainability.explanations import NaturalLanguageExplainer
        _MODEL_CACHE["explainer"] = NaturalLanguageExplainer()
    return _MODEL_CACHE["explainer"]
