"""
Unit tests for explainability components:
- Facial Region Analyzer
- Action Unit Extractor
- Top Two Comparator
- Counterfactual Analyzer
- Natural Language Explainer
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import NAVARASA_CLASSES
from src.explainability.facial_regions import FacialRegionAnalyzer
from src.explainability.action_units import ActionUnitExtractor
from src.explainability.counterfactual import TopTwoComparator, CounterfactualFeatureAnalyzer
from src.explainability.explanations import NaturalLanguageExplainer


def test_facial_regions_and_energy():
    analyzer = FacialRegionAnalyzer()
    dummy_img = Image.new("RGB", (224, 224), color=(100, 100, 100))

    landmarks, _ = analyzer.extract_landmarks(dummy_img)
    masks = analyzer.generate_region_masks((224, 224, 3), landmarks)

    assert len(masks) == 6, f"Expected 6 facial regions, got {len(masks)}"
    assert set(masks.keys()) == {"Eyes", "Eyebrows", "Nose", "Mouth", "Cheeks", "Forehead"}

    # Mock heatmap
    dummy_cam = np.ones((224, 224), dtype=np.float32)
    pcts, sorted_r = analyzer.calculate_region_importance(dummy_cam, masks)

    assert len(pcts) == 6
    total_pct = sum(pcts.values())
    assert abs(total_pct - 100.0) < 1.0, f"Region percentages should sum to ~100%, got {total_pct}"


def test_action_unit_extractor():
    extractor = ActionUnitExtractor()
    dummy_img = Image.new("RGB", (224, 224), color=(150, 150, 150))
    aus, face_ok = extractor.extract_action_units(dummy_img)

    assert len(aus) >= 10
    assert "AU01" in aus
    assert "AU12" in aus
    assert "intensity" in aus["AU01"]


def test_top_two_comparator():
    comparator = TopTwoComparator()
    dummy_probs = {c: 0.1 for c in NAVARASA_CLASSES}
    dummy_probs["Raudra"] = 0.45
    dummy_probs["Bibhatsa"] = 0.35

    r_c1 = {r: 16.6 for r in ["Eyes", "Eyebrows", "Nose", "Mouth", "Cheeks", "Forehead"]}
    r_c2 = {r: 16.6 for r in ["Eyes", "Eyebrows", "Nose", "Mouth", "Cheeks", "Forehead"]}
    r_c1["Eyebrows"] = 35.0
    r_c2["Nose"] = 40.0

    dummy_aus = {
        "AU04": {"id": "AU04", "name": "Brow Lowerer", "intensity": 0.8, "is_active": True, "relevant_rasas": ["Raudra"]}
    }

    report = comparator.compare(dummy_probs, r_c1, r_c2, dummy_aus)
    assert report["top1_class"] == "Raudra"
    assert report["top2_class"] == "Bibhatsa"
    assert report["probability_margin"] == 0.10


def test_counterfactual_analyzer():
    cf_analyzer = CounterfactualFeatureAnalyzer()
    dummy_aus = {
        "AU04": {"id": "AU04", "name": "Brow Lowerer", "intensity": 0.85, "is_active": True},
        "AU12": {"id": "AU12", "name": "Smile", "intensity": 0.05, "is_active": False},
    }
    dummy_probs = {"Raudra": 0.70, "Hasya": 0.10}

    report = cf_analyzer.analyze_counterfactual("Raudra", "Hasya", dummy_aus, dummy_probs)
    assert report["source_class"] == "Raudra"
    assert report["target_class"] == "Hasya"
    assert len(report["feature_adjustments"]) > 0


def test_natural_language_explainer():
    explainer = NaturalLanguageExplainer()
    probs = {c: 0.05 for c in NAVARASA_CLASSES}
    probs["Hasya"] = 0.60
    top_regions = [("Mouth", 45.0), ("Cheeks", 35.0)]
    active_aus = [{"id": "AU12", "name": "Lip Corner Puller", "intensity": 0.75}]

    text = explainer.generate_explanation(
        predicted_class="Hasya",
        confidence=0.60,
        probabilities=probs,
        top_regions=top_regions,
        active_action_units=active_aus
    )
    assert "Hasya" in text
    assert "Mouth" in text
    assert "AU12" in text


if __name__ == "__main__":
    test_facial_regions_and_energy()
    test_action_unit_extractor()
    test_top_two_comparator()
    test_counterfactual_analyzer()
    test_natural_language_explainer()
    print("All explainability tests passed!")
