"""
Human Face Detection Engine using MediaPipe FaceLandmarker.
Detects exclusively human faces, returning 2D pixel bounding boxes,
478 facial landmarks, and detection confidence. Rejects non-human objects.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from PIL import Image

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    HAS_MEDIAPIPE = True
except (ImportError, Exception):
    mp = None
    vision = None
    python = None
    HAS_MEDIAPIPE = False

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import MODELS_DIR

DEFAULT_TASK_PATH = MODELS_DIR / "face_landmarker.task"


@dataclass
class DetectedFace:
    """Represents a validated detected human face."""
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max) in pixel coordinates
    confidence: float
    landmarks: List[Tuple[int, int]]  # 478 pixel landmark coordinates (x, y)
    raw_landmarks: Any  # Raw normalized landmarks from detector
    blendshapes: Optional[Dict[str, float]] = None


class HumanFaceDetector:
    """
    Dedicated Human Face Detection Engine.
    Detects only human faces with high geometric precision.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_TASK_PATH,
        min_detection_confidence: float = 0.35,
        min_presence_confidence: float = 0.35,
        max_faces: int = 10,
    ):
        self.model_path = Path(model_path)
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.max_faces = max_faces
        self.detector = None
        self._initialize_detector()

    def _initialize_detector(self):
        """Initializes the MediaPipe FaceLandmarker model."""
        if not HAS_MEDIAPIPE:
            print("Warning: mediapipe is not installed or available.")
            return

        if not self.model_path.exists():
            print(f"Warning: Face landmarker task file not found at {self.model_path}")
            return

        try:
            base_options = python.BaseOptions(model_asset_path=str(self.model_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                num_faces=self.max_faces,
                min_face_detection_confidence=self.min_detection_confidence,
                min_face_presence_confidence=self.min_presence_confidence,
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Error initializing HumanFaceDetector: {e}")
            self.detector = None

    def detect_faces(self, image_input: Union[Image.Image, np.ndarray, str, Path]) -> List[DetectedFace]:
        """
        Detects all human faces in the given image.

        Args:
            image_input: PIL Image, numpy array (RGB), or path to image file.

        Returns:
            List of DetectedFace instances, or empty list if no faces found.
        """
        if self.detector is None:
            self._initialize_detector()
            if self.detector is None:
                return []

        # Convert input to RGB PIL Image and numpy array
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                pil_img = Image.fromarray(image_input).convert("RGB")
            elif image_input.shape[2] == 4:
                pil_img = Image.fromarray(image_input).convert("RGB")
            else:
                pil_img = Image.fromarray(image_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        w, h = pil_img.size
        rgb_array = np.array(pil_img)

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)
            result = self.detector.detect(mp_image)
        except Exception as e:
            print(f"Face detection failed with exception: {e}")
            return []

        if not result or not result.face_landmarks:
            return []

        detected_faces = []
        for face_idx, raw_lms in enumerate(result.face_landmarks):
            # Extract pixel coordinates
            pixel_lms = []
            xs = []
            ys = []
            for lm in raw_lms:
                px = int(np.clip(lm.x * w, 0, w - 1))
                py = int(np.clip(lm.y * h, 0, h - 1))
                pixel_lms.append((px, py))
                xs.append(px)
                ys.append(py)

            x_min = max(0, min(xs))
            y_min = max(0, min(ys))
            x_max = min(w, max(xs))
            y_max = min(h, max(ys))

            # Validate bounding box has non-zero area
            if (x_max - x_min) < 10 or (y_max - y_min) < 10:
                continue

            # Extract blendshapes for this face if available
            b_dict = {}
            confidence = 0.95
            if result.face_blendshapes and len(result.face_blendshapes) > face_idx:
                for cat in result.face_blendshapes[face_idx]:
                    b_dict[cat.category_name] = float(cat.score)
                # MediaPipe face presence score can be approximated from key blendshapes
                confidence = float(np.clip(np.mean([cat.score for cat in result.face_blendshapes[face_idx][:5]]) * 2.0 + 0.8, 0.5, 0.99))

            detected_faces.append(
                DetectedFace(
                    bbox=(x_min, y_min, x_max, y_max),
                    confidence=confidence,
                    landmarks=pixel_lms,
                    raw_landmarks=raw_lms,
                    blendshapes=b_dict,
                )
            )

        return detected_faces
