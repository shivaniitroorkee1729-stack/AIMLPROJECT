"""
Facial Action Unit (AU) Extraction and Analysis.
Utilizes MediaPipe FaceLandmarker's 52 FACS-derived facial blendshape coefficients
to extract standardized Facial Action Units (AU1 to AU27).
Provides intensity measurements, thresholded activations, and comparative analysis.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.explainability.facial_regions import FacialRegionAnalyzer

# FACS Action Unit Definitions with corresponding Blendshape Mappings
AU_FACS_MAPPING = {
    "AU01": {
        "name": "Inner Brow Raiser",
        "muscle": "Frontalis, pars medialis",
        "blendshapes": ["browInnerUp"],
        "rasas": ["Karuna", "Bhayanaka", "Adbhuta"],
        "description": "Pulls inner portion of eyebrows upward, creating sadness or dread."
    },
    "AU02": {
        "name": "Outer Brow Raiser",
        "muscle": "Frontalis, pars lateralis",
        "blendshapes": ["browOuterUpLeft", "browOuterUpRight"],
        "rasas": ["Adbhuta", "Bhayanaka"],
        "description": "Arches and lifts outer brow arches, expressing astonishment or fright."
    },
    "AU04": {
        "name": "Brow Lowerer / Furrower",
        "muscle": "Corrugator supercilii / Depressor supercilii",
        "blendshapes": ["browDownLeft", "browDownRight"],
        "rasas": ["Raudra", "Bibhatsa"],
        "description": "Draws eyebrows together and downward in concentrated anger or contempt."
    },
    "AU05": {
        "name": "Upper Lid Raiser",
        "muscle": "Levator palpebrae superioris",
        "blendshapes": ["eyeWideLeft", "eyeWideRight"],
        "rasas": ["Bhayanaka", "Adbhuta", "Raudra"],
        "description": "Widens eye aperture, revealing white sclera above the iris."
    },
    "AU06": {
        "name": "Cheek Raiser",
        "muscle": "Orbicularis oculi, pars orbitalis",
        "blendshapes": ["cheekSquintLeft", "cheekSquintRight"],
        "rasas": ["Hasya"],
        "description": "Lifts cheeks upward, producing crows-feet wrinkles around eye corners."
    },
    "AU07": {
        "name": "Lid Tightener",
        "muscle": "Orbicularis oculi, pars palpebralis",
        "blendshapes": ["eyeSquintLeft", "eyeSquintRight"],
        "rasas": ["Raudra", "Bibhatsa"],
        "description": "Narrows eye aperture by tightening lower and upper eyelids."
    },
    "AU09": {
        "name": "Nose Wrinkler",
        "muscle": "Levator labii superioris alaeque nasi",
        "blendshapes": ["noseSneerLeft", "noseSneerRight"],
        "rasas": ["Bibhatsa"],
        "description": "Pulls skin upwards along nasal bridge, characteristic of aversion."
    },
    "AU10": {
        "name": "Upper Lip Raiser",
        "muscle": "Levator labii superioris",
        "blendshapes": ["mouthUpperUpLeft", "mouthUpperUpRight"],
        "rasas": ["Bibhatsa", "Raudra"],
        "description": "Elevates upper lip, exposing upper teeth in disgust."
    },
    "AU12": {
        "name": "Lip Corner Puller (Smile)",
        "muscle": "Zygomaticus major",
        "blendshapes": ["mouthSmileLeft", "mouthSmileRight"],
        "rasas": ["Hasya", "Shringara"],
        "description": "Pulls mouth corners obliquely upwards in joy or affection."
    },
    "AU14": {
        "name": "Dimpler",
        "muscle": "Buccinator",
        "blendshapes": ["mouthDimpleLeft", "mouthDimpleRight"],
        "rasas": ["Shringara", "Veera"],
        "description": "Tightens lip corners inward creating dimples, indicating poise."
    },
    "AU15": {
        "name": "Lip Corner Depressor",
        "muscle": "Depressor anguli oris",
        "blendshapes": ["mouthFrownLeft", "mouthFrownRight"],
        "rasas": ["Karuna"],
        "description": "Depresses lip corners downwards, signalling poignant grief."
    },
    "AU20": {
        "name": "Lip Stretcher",
        "muscle": "Risorius",
        "blendshapes": ["mouthStretchLeft", "mouthStretchRight"],
        "rasas": ["Bhayanaka"],
        "description": "Pulls lip corners horizontally back toward ears in alarm."
    },
    "AU25": {
        "name": "Lips Part",
        "muscle": "Depressor labii inferioris / Orbicularis oris",
        "blendshapes": ["mouthPucker"],
        "rasas": ["Adbhuta", "Bhayanaka"],
        "description": "Parts upper and lower lips, exposing mouth cavity."
    },
    "AU26": {
        "name": "Jaw Drop",
        "muscle": "Masseter / Temporalis relaxation",
        "blendshapes": ["jawOpen"],
        "rasas": ["Adbhuta", "Hasya"],
        "description": "Lowers mandible downwards, creating an open-mouthed expression."
    }
}


class ActionUnitExtractor:
    """
    Extracts FACS Action Units from image using MediaPipe blendshape outputs.
    """

    def __init__(self, region_analyzer=None):
        self.analyzer = region_analyzer or FacialRegionAnalyzer()

    def extract_action_units(self, image_input):
        """
        Extracts AU intensities (0.0 to 1.0) and activation state.
        Returns:
            au_results: dict of {au_id: {"name": str, "intensity": float, "is_active": bool, ...}}
            face_detected: bool
        """
        landmarks, detection_result = self.analyzer.extract_landmarks(image_input)

        if detection_result is None or not detection_result.face_blendshapes:
            # Face not detected
            fallback_aus = {}
            for au_id, meta in AU_FACS_MAPPING.items():
                fallback_aus[au_id] = {
                    "id": au_id,
                    "name": meta["name"],
                    "muscle": meta["muscle"],
                    "intensity": 0.0,
                    "is_active": False,
                    "relevant_rasas": meta["rasas"],
                    "description": meta["description"],
                }
            return fallback_aus, False

        # Extract blendshapes dictionary
        blendshapes = detection_result.face_blendshapes[0]
        bs_dict = {bs.category_name: bs.score for bs in blendshapes}

        au_results = {}
        for au_id, meta in AU_FACS_MAPPING.items():
            # Average scores across mapped blendshape channels (e.g. left + right)
            scores = [bs_dict.get(channel, 0.0) for channel in meta["blendshapes"]]
            intensity = float(np.mean(scores)) if scores else 0.0

            # Scale slightly for human readability (most blendshapes cap around 0.7 in natural face)
            normalized_intensity = min(1.0, round(intensity * 1.35, 3))
            is_active = normalized_intensity >= 0.15

            au_results[au_id] = {
                "id": au_id,
                "name": meta["name"],
                "muscle": meta["muscle"],
                "intensity": normalized_intensity,
                "is_active": is_active,
                "relevant_rasas": meta["rasas"],
                "description": meta["description"],
            }

        return au_results, True
