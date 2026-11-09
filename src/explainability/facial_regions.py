"""
Facial Region Importance Analysis.
Segments facial images into 6 canonical facial regions:
1. Eyes
2. Eyebrows
3. Nose
4. Mouth
5. Cheeks
6. Forehead
Calculates the proportional Grad-CAM activation energy across regions
to explain which facial features influenced the model's decision.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import MODELS_DIR

LANDMARKER_MODEL_PATH = MODELS_DIR / "face_landmarker.task"

# MediaPipe 468/478 Landmark Indices for 6 Facial Regions
LANDMARK_REGIONS = {
    "Eyebrows": [
        # Left eyebrow
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
        # Right eyebrow
        336, 296, 334, 293, 300, 276, 283, 282, 295, 285
    ],
    "Eyes": [
        # Left eye
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        # Right eye
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
    ],
    "Nose": [
        1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 19, 94, 218, 438, 274, 44, 278, 48
    ],
    "Mouth": [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
        14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311,
        312, 13, 82, 81, 42, 183, 78
    ],
    "Cheeks": [
        # Left cheek
        116, 123, 147, 213, 192, 214, 210, 211, 32, 208, 199, 201, 200, 18,
        # Right cheek
        345, 352, 376, 433, 416, 434, 430, 431, 262, 428, 421, 422
    ],
    "Forehead": [
        10, 338, 297, 332, 284, 251, 389, 356, 109, 67, 103, 54, 21, 162, 127
    ]
}

REGION_COLORS = {
    "Eyes": (66, 133, 244),       # Blue
    "Eyebrows": (234, 67, 53),     # Red
    "Nose": (251, 188, 5),        # Yellow
    "Mouth": (52, 168, 83),       # Green
    "Cheeks": (255, 112, 67),     # Orange
    "Forehead": (171, 71, 188),   # Purple
}


class FacialRegionAnalyzer:
    """
    Facial landmark extractor and region importance analyzer.
    """

    def __init__(self, model_path=LANDMARKER_MODEL_PATH):
        self.model_path = Path(model_path)
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        if not self.model_path.exists():
            print(f"Warning: Landmark model not found at {self.model_path}. Fallback mode active.")
            return

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def extract_landmarks(self, image_input):
        """
        Extracts 478 normalized facial landmarks from an image.
        Returns:
            landmarks_xy: list of (x_pixel, y_pixel)
            raw_result: FaceLandmarkerResult
        """
        if isinstance(image_input, (str, Path)):
            img_bgr = cv2.imread(str(image_input))
            rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            rgb_img = np.array(image_input.convert("RGB"))
        else:
            rgb_img = image_input

        h, w = rgb_img.shape[:2]

        if self.detector is None:
            return self._fallback_landmarks(w, h), None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.face_landmarks:
            return self._fallback_landmarks(w, h), None

        face_landmarks = detection_result.face_landmarks[0]
        pixel_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
        return pixel_landmarks, detection_result

    def _fallback_landmarks(self, w, h):
        """Proportional geometric approximation if face is un-detected."""
        pts = []
        for i in range(478):
            pts.append((int(w * 0.5), int(h * 0.5)))
        return pts

    def generate_region_masks(self, image_shape, pixel_landmarks):
        """
        Builds binary 2D mask [H, W] for each of the 6 facial regions.
        """
        h, w = image_shape[:2]
        masks = {}

        # Fallback if landmarks collapsed
        all_same = len(set(pixel_landmarks[:20])) <= 1
        if all_same:
            # Geometric sector bounding fallback
            masks["Forehead"] = np.zeros((h, w), dtype=np.uint8)
            masks["Forehead"][:int(h * 0.28), :] = 255
            masks["Eyebrows"] = np.zeros((h, w), dtype=np.uint8)
            masks["Eyebrows"][int(h * 0.25):int(h * 0.40), :] = 255
            masks["Eyes"] = np.zeros((h, w), dtype=np.uint8)
            masks["Eyes"][int(h * 0.35):int(h * 0.52), :] = 255
            masks["Nose"] = np.zeros((h, w), dtype=np.uint8)
            masks["Nose"][int(h * 0.45):int(h * 0.68), int(w * 0.3):int(w * 0.7)] = 255
            masks["Cheeks"] = np.zeros((h, w), dtype=np.uint8)
            masks["Cheeks"][int(h * 0.50):int(h * 0.75), :] = 255
            masks["Mouth"] = np.zeros((h, w), dtype=np.uint8)
            masks["Mouth"][int(h * 0.70):int(h * 0.95), int(w * 0.2):int(w * 0.8)] = 255
            return masks

        for region_name, indices in LANDMARK_REGIONS.items():
            mask = np.zeros((h, w), dtype=np.uint8)
            region_pts = np.array([pixel_landmarks[idx] for idx in indices if idx < len(pixel_landmarks)])

            if len(region_pts) >= 3:
                # Compute convex hull for the region
                hull = cv2.convexHull(region_pts)
                cv2.fillConvexPoly(mask, hull, 255)
                # Dilate slightly to capture surrounding expressive musculature
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                mask = cv2.dilate(mask, kernel, iterations=1)
            masks[region_name] = mask

        return masks

    def calculate_region_importance(self, heatmap, region_masks):
        """
        Computes Grad-CAM activation energy in each region mask.
        Returns:
            region_percentages: dict of {region_name: float (percentage 0-100%)}
            most_influential: list of (region_name, percentage) sorted descending
        """
        h, w = heatmap.shape[:2]
        region_energy = {}
        total_energy = 0.0

        for region_name, mask in region_masks.items():
            if mask.shape[:2] != (h, w):
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask

            binary_mask = (mask_resized > 0).astype(np.float32)
            energy = float(np.sum(heatmap * binary_mask))
            region_energy[region_name] = energy
            total_energy += energy

        # If zero total activation, divide equally
        if total_energy < 1e-6:
            equal_pct = round(100.0 / len(region_masks), 2)
            region_pcts = {r: equal_pct for r in region_masks}
        else:
            region_pcts = {
                r: round((energy / total_energy) * 100.0, 2)
                for r, energy in region_energy.items()
            }

        sorted_regions = sorted(region_pcts.items(), key=lambda x: x[1], reverse=True)
        return region_pcts, sorted_regions

    def draw_landmarks_and_regions(self, image, pixel_landmarks, region_masks=None):
        """
        Draws landmark points and regional contours on image for visualization.
        """
        if isinstance(image, Image.Image):
            vis_img = np.array(image.convert("RGB"))
        else:
            vis_img = image.copy()

        # Draw region outlines
        if region_masks:
            for r_name, mask in region_masks.items():
                color = REGION_COLORS.get(r_name, (200, 200, 200))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_img, contours, -1, color, 2)

        # Draw subtle landmark dots
        for pt in pixel_landmarks[::3]:  # sample every 3rd point for clean display
            cv2.circle(vis_img, pt, 1, (255, 255, 255), -1)

        return vis_img
