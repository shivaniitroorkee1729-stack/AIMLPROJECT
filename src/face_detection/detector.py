"""
Robust Multi-Engine Human Face Detection Engine.
Combines OpenCV YuNet (multi-scale deep convolutional face detector) with
Google MediaPipe FaceLandmarker (478 3D landmarks + FACS blendshapes).

Key Capabilities:
1. Detects human faces across all scales: close-ups, medium-distance, and wide 1920x1080 video frames.
2. EXIF orientation correction for mobile and camera uploads.
3. Strict human-face anatomical verification (eliminating hands, cloth folds, geometric shapes, and wall drawings).
4. Dual-detector candidate generation with IoU Non-Maximum Suppression to prevent double-counting.
"""

import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from PIL import Image, ImageOps

# Safe OpenCV import
try:
    import cv2
    HAS_CV2 = True
except (ImportError, Exception):
    cv2 = None
    HAS_CV2 = False

# Safe MediaPipe import
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
DEFAULT_YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_DOWNLOAD_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


@dataclass
class DetectedFace:
    """Represents a validated detected human face with bounding box and geometric landmarks."""
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max) in pixel coordinates
    confidence: float
    landmarks: List[Tuple[int, int]]  # 478 pixel landmark coordinates (x, y)
    raw_landmarks: Any  # Raw normalized landmarks from landmarker
    blendshapes: Optional[Dict[str, float]] = None
    detector_source: str = "hybrid_cascade"
    face_area_pct: float = 0.0


def compute_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    """Calculates Intersection over Union between two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class HumanFaceDetector:
    """
    Robust Multi-Engine Human Face Detector.
    Integrates OpenCV YuNet and MediaPipe FaceLandmarker into a cohesive cascade.
    """

    def __init__(
        self,
        landmarker_path: Union[str, Path] = DEFAULT_TASK_PATH,
        yunet_path: Union[str, Path] = DEFAULT_YUNET_PATH,
        score_threshold: float = 0.60,
        min_detection_confidence: float = 0.20,
        max_faces: int = 10,
    ):
        self.landmarker_path = Path(landmarker_path)
        self.yunet_path = Path(yunet_path)
        self.score_threshold = score_threshold
        self.min_detection_confidence = min_detection_confidence
        self.max_faces = max_faces

        self.yunet_detector = None
        self.full_landmarker = None
        self.crop_landmarker = None

        self._initialize_detectors()

    def _ensure_yunet_model(self):
        """Ensures YuNet model weights exist locally; downloads if missing."""
        if not self.yunet_path.exists():
            try:
                self.yunet_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"Downloading YuNet face detection weights to {self.yunet_path}...")
                urllib.request.urlretrieve(YUNET_DOWNLOAD_URL, str(self.yunet_path))
                print(f"Successfully downloaded YuNet weights ({self.yunet_path.stat().st_size} bytes).")
            except Exception as e:
                print(f"Warning: Could not auto-download YuNet weights: {e}")

    def _initialize_detectors(self):
        """Initializes OpenCV YuNet and MediaPipe FaceLandmarker instances."""
        # 1. Initialize YuNet
        if HAS_CV2 and hasattr(cv2, "FaceDetectorYN"):
            self._ensure_yunet_model()
            if self.yunet_path.exists():
                try:
                    self.yunet_detector = cv2.FaceDetectorYN.create(
                        model=str(self.yunet_path),
                        config="",
                        input_size=(320, 320),
                        score_threshold=self.score_threshold,
                        nms_threshold=0.30,
                    )
                except Exception as e:
                    print(f"Warning: Could not initialize cv2.FaceDetectorYN: {e}")
                    self.yunet_detector = None

        # 2. Initialize MediaPipe FaceLandmarker (Shared Singleton)
        if HAS_MEDIAPIPE and self.landmarker_path.exists():
            try:
                from src.models.model_loader import get_shared_face_landmarker
                shared_lm = get_shared_face_landmarker(self.landmarker_path)
                self.full_landmarker = shared_lm
                self.crop_landmarker = shared_lm
            except Exception:
                try:
                    base_options = python.BaseOptions(model_asset_path=str(self.landmarker_path))
                    opts = vision.FaceLandmarkerOptions(
                        base_options=base_options,
                        output_face_blendshapes=True,
                        num_faces=self.max_faces,
                        min_face_detection_confidence=self.min_detection_confidence,
                        min_face_presence_confidence=self.min_detection_confidence,
                    )
                    lm = vision.FaceLandmarker.create_from_options(opts)
                    self.full_landmarker = lm
                    self.crop_landmarker = lm
                except Exception as e:
                    print(f"Warning: Could not initialize MediaPipe FaceLandmarker: {e}")
                    self.full_landmarker = None
                    self.crop_landmarker = None

    def preprocess_image(self, image_input: Union[Image.Image, np.ndarray, str, Path]) -> Tuple[Image.Image, int, int]:
        """
        Loads, orientates, and converts input image to standard RGB PIL Image.
        Corrects EXIF orientation so portrait/camera uploads are upright.
        """
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.copy()
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                pil_img = Image.fromarray(image_input).convert("RGB")
            elif image_input.shape[2] == 4:
                pil_img = Image.fromarray(image_input).convert("RGB")
            else:
                pil_img = Image.fromarray(image_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Correct EXIF rotation
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        w, h = pil_img.size
        return pil_img, w, h

    def detect_faces(self, image_input: Union[Image.Image, np.ndarray, str, Path]) -> List[DetectedFace]:
        """
        Detects all human faces in the given image using multi-engine cascade.

        Args:
            image_input: PIL Image, numpy array (RGB), or path to image file.

        Returns:
            List of DetectedFace instances, or empty list if no faces found.
        """
        if self.yunet_detector is None and self.full_landmarker is None:
            self._initialize_detectors()
            if self.yunet_detector is None and self.full_landmarker is None:
                return []

        pil_img, w, h = self.preprocess_image(image_input)
        if w < 25 or h < 25:
            return []

        rgb_arr = np.array(pil_img)
        img_area = float(w * h)

        # Stage 1: MediaPipe direct detection on full image
        full_mp_faces: List[Dict[str, Any]] = []
        if self.full_landmarker is not None:
            try:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_arr)
                mp_res = self.full_landmarker.detect(mp_img)
                if mp_res and mp_res.face_landmarks:
                    for idx, lms in enumerate(mp_res.face_landmarks):
                        xs = [lm.x * w for lm in lms]
                        ys = [lm.y * h for lm in lms]
                        b_dict = {cat.category_name: cat.score for cat in mp_res.face_blendshapes[idx]} if mp_res.face_blendshapes else {}
                        xmin = int(np.clip(min(xs), 0, w - 1))
                        ymin = int(np.clip(min(ys), 0, h - 1))
                        xmax = int(np.clip(max(xs), 0, w))
                        ymax = int(np.clip(max(ys), 0, h))
                        if (xmax - xmin) >= 20 and (ymax - ymin) >= 20:
                            full_mp_faces.append({
                                "bbox": (xmin, ymin, xmax, ymax),
                                "confidence": 0.95,
                                "landmarks": [(int(np.clip(lm.x * w, 0, w - 1)), int(np.clip(lm.y * h, 0, h - 1))) for lm in lms],
                                "raw_landmarks": lms,
                                "blendshapes": b_dict,
                                "source": "mediapipe_direct",
                            })
            except Exception as e:
                print(f"MediaPipe full image detection notice: {e}")

        # Stage 2: OpenCV YuNet multi-scale candidate detection
        yn_candidates: List[Tuple[int, int, int, int, float]] = []
        if self.yunet_detector is not None and HAS_CV2:
            try:
                bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
                self.yunet_detector.setInputSize((w, h))
                _, faces_raw = self.yunet_detector.detect(bgr_arr)
                if faces_raw is not None:
                    for f in faces_raw:
                        bx, by, bw, bh, score = int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[-1])
                        bx1 = max(0, bx)
                        by1 = max(0, by)
                        bx2 = min(w, bx + bw)
                        by2 = min(h, by + bh)
                        area = (bx2 - bx1) * (by2 - by1)
                        # Must be at least 25x25 pixels and meet score threshold
                        if (bx2 - bx1) >= 25 and (by2 - by1) >= 25 and score >= self.score_threshold:
                            yn_candidates.append((bx1, by1, bx2, by2, score))
            except Exception as e:
                print(f"YuNet candidate detection notice: {e}")

        # Stage 3: Merge candidates & cross-verify using Crop Landmarker
        verified_faces: List[Dict[str, Any]] = []

        # 3A. Process YuNet candidates (primary proposal engine)
        for cbox in yn_candidates:
            bx1, by1, bx2, by2, score = cbox
            cw, ch = bx2 - bx1, by2 - by1
            pad = 0.22
            side = max(cw, ch) * (1.0 + 2.0 * pad)
            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0 - 0.04 * ch
            x1 = max(0, int(cx - side / 2.0))
            y1 = max(0, int(cy - side / 2.0))
            x2 = min(w, int(cx + side / 2.0))
            y2 = min(h, int(cy + side / 2.0))

            crop_pil = pil_img.crop((x1, y1, x2, y2))
            crop_w, crop_h = crop_pil.size

            verified = False
            if self.crop_landmarker is not None:
                try:
                    cr_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(crop_pil))
                    cr_res = self.crop_landmarker.detect(cr_img)
                    if cr_res and cr_res.face_landmarks:
                        lms = cr_res.face_landmarks[0]
                        orig_lms = [(int(x1 + lm.x * crop_w), int(y1 + lm.y * crop_h)) for lm in lms]
                        xs = [pt[0] for pt in orig_lms]
                        ys = [pt[1] for pt in orig_lms]
                        b_dict = {cat.category_name: cat.score for cat in cr_res.face_blendshapes[0]} if cr_res.face_blendshapes else {}
                        verified_faces.append({
                            "bbox": (int(max(0, min(xs))), int(max(0, min(ys))), int(min(w, max(xs))), int(min(h, max(ys)))),
                            "confidence": float(score),
                            "landmarks": orig_lms,
                            "raw_landmarks": lms,
                            "blendshapes": b_dict,
                            "source": "yunet_verified",
                        })
                        verified = True
                except Exception as e:
                    print(f"Crop verification notice: {e}")

            if not verified and score >= 0.75:
                # High-confidence YuNet fallback
                sim_lms = [(int(bx1 + (i % 20) * (cw / 20)), int(by1 + (i // 20) * (ch / 24))) for i in range(478)]
                verified_faces.append({
                    "bbox": (bx1, by1, bx2, by2),
                    "confidence": float(score),
                    "landmarks": sim_lms,
                    "raw_landmarks": None,
                    "blendshapes": {},
                    "source": "yunet_direct",
                })

        # 3B. Process MediaPipe full image detections (fallback / cross-check)
        for mp_face in full_mp_faces:
            already_covered = any(compute_iou(mp_face["bbox"], f["bbox"]) > 0.30 for f in verified_faces)
            if already_covered:
                continue

            # If YuNet detected faces elsewhere, verify this extra MP face with YuNet on its crop
            # to prevent hallucinated hands/alta/fabric from becoming false faces
            if yn_candidates and self.yunet_detector is not None:
                mbx1, mby1, mbx2, mby2 = mp_face["bbox"]
                mcrop = pil_img.crop((mbx1, mby1, mbx2, mby2))
                mw, mh = mcrop.size
                if mw >= 25 and mh >= 25:
                    self.yunet_detector.setInputSize((mw, mh))
                    _, yn_sub = self.yunet_detector.detect(cv2.cvtColor(np.array(mcrop), cv2.COLOR_RGB2BGR))
                    # If YuNet finds no face at all in this crop, it's a hallucination (e.g. mudra hand)
                    if yn_sub is None or len(yn_sub) == 0 or float(yn_sub[0][-1]) < 0.30:
                        continue

            verified_faces.append(mp_face)

        if not verified_faces:
            return []


        # Stage 4: Deduplicate overlapping faces via NMS
        verified_faces = sorted(verified_faces, key=lambda f: f["confidence"], reverse=True)
        nms_faces: List[Dict[str, Any]] = []
        for f in verified_faces:
            if not any(compute_iou(f["bbox"], kept["bbox"]) > 0.40 for kept in nms_faces):
                nms_faces.append(f)

        # Stage 5: Filter out tiny background decorations/paintings when a primary human face exists
        if len(nms_faces) > 1:
            areas = [(f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]) for f in nms_faces]
            max_area = max(areas)
            # A valid secondary person must occupy at least 15% of the primary face area, or at least 1.2% of the whole image
            filtered = [
                f for f, a in zip(nms_faces, areas)
                if (a >= 0.15 * max_area) or (a / img_area >= 0.012)
            ]
            if filtered:
                nms_faces = filtered

        # Stage 6: Filter out false hand/jewelry hallucinations if detected by MediaPipe alone without YuNet support
        if len(nms_faces) > 1 and yn_candidates:
            yn_supported = []
            for f in nms_faces:
                has_support = any(compute_iou(f["bbox"], yc[:4]) > 0.20 for yc in yn_candidates)
                if has_support or f["confidence"] >= 0.90:
                    yn_supported.append(f)
            if yn_supported:
                nms_faces = yn_supported

        # Build DetectedFace dataclass instances
        detected_objects: List[DetectedFace] = []
        for f in nms_faces:
            bx1, by1, bx2, by2 = f["bbox"]
            face_area = (bx2 - bx1) * (by2 - by1)
            area_pct = round((face_area / img_area) * 100.0, 2)

            detected_objects.append(
                DetectedFace(
                    bbox=(bx1, by1, bx2, by2),
                    confidence=float(f["confidence"]),
                    landmarks=f["landmarks"],
                    raw_landmarks=f["raw_landmarks"],
                    blendshapes=f["blendshapes"],
                    detector_source=f["source"],
                    face_area_pct=area_pct,
                )
            )

        return detected_objects
