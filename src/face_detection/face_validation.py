"""
Strict Human Face Count and Geometry Validation Layer.
Implements the mandatory 3-case validation logic:
Case A: 0 faces  -> Reject all ML/XAI processing.
Case B: 1 face   -> Accept, extract proportional face crop, proceed to ML/XAI.
Case C: 2+ faces -> Reject all ML/XAI processing, display all bounding boxes.
"""

import hashlib
import io
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np
from PIL import Image, ImageOps

from src.face_detection.detector import HumanFaceDetector, DetectedFace
from src.face_detection.face_crop import extract_face_crop, annotate_image_with_bboxes


class FaceValidationStatus(str, Enum):
    NO_FACE = "NO_FACE"
    SINGLE_FACE = "SINGLE_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"


@dataclass
class FaceValidationResult:
    """Standardized validation output used across all UI pages and inference functions."""
    status: FaceValidationStatus
    face_count: int
    is_valid: bool
    message: str
    original_image: Image.Image
    annotated_image: Image.Image
    face_crop: Optional[Image.Image]
    crop_bbox: Optional[Tuple[int, int, int, int]]
    detected_faces: List[DetectedFace]
    image_hash: str
    detector_used: str = "YuNet + MediaPipe Cascade"
    confidence: float = 0.0
    face_area_pct: float = 0.0


# Module-level detector singleton and hash-based validation cache
_DETECTOR_SINGLETON: Optional[HumanFaceDetector] = None
_VALIDATION_CACHE = {}


def get_detector() -> HumanFaceDetector:
    """Returns the shared HumanFaceDetector singleton instance."""
    global _DETECTOR_SINGLETON
    if _DETECTOR_SINGLETON is None:
        _DETECTOR_SINGLETON = HumanFaceDetector()
    return _DETECTOR_SINGLETON


def compute_image_hash(image_input: Union[Image.Image, np.ndarray, str, Path, bytes]) -> str:
    """Computes a deterministic MD5 hash of an image for cache validation and change tracking."""
    if isinstance(image_input, bytes):
        return hashlib.md5(image_input).hexdigest()
    elif isinstance(image_input, (str, Path)):
        with open(image_input, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    elif isinstance(image_input, Image.Image):
        buf = io.BytesIO()
        image_input.save(buf, format="PNG")
        return hashlib.md5(buf.getvalue()).hexdigest()
    elif isinstance(image_input, np.ndarray):
        return hashlib.md5(image_input.tobytes()).hexdigest()
    else:
        return hashlib.md5(str(image_input).encode()).hexdigest()


def validate_face_image(
    image_input: Union[Image.Image, np.ndarray, str, Path, bytes],
    padding_ratio: float = 0.22,
    detector: Optional[HumanFaceDetector] = None,
) -> FaceValidationResult:
    """
    Validates the uploaded image for exact human face count.

    Strict Behavior:
    - 0 faces: Rejects. Sets status=NO_FACE, is_valid=False, face_crop=None.
    - 1 face: Accepts. Sets status=SINGLE_FACE, is_valid=True, extracts face crop.
    - 2+ faces: Rejects. Sets status=MULTIPLE_FACES, is_valid=False, annotates all faces.

    Args:
        image_input: Input image as PIL Image, numpy array, path, or bytes.
        padding_ratio: Extra padding ratio around the face crop.
        detector: Optional custom HumanFaceDetector instance.

    Returns:
        FaceValidationResult object with validation outcome and extracted assets.
    """
    img_hash = compute_image_hash(image_input)

    # Check cache
    cache_key = f"{img_hash}_{padding_ratio}"
    if cache_key in _VALIDATION_CACHE:
        return _VALIDATION_CACHE[cache_key]

    # Normalize image to RGB PIL Image with EXIF orientation correction
    if isinstance(image_input, bytes):
        original_pil = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, (str, Path)):
        original_pil = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        original_pil = image_input.copy()
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            original_pil = Image.fromarray(image_input).convert("RGB")
        elif image_input.shape[2] == 4:
            original_pil = Image.fromarray(image_input).convert("RGB")
        else:
            original_pil = Image.fromarray(image_input)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    original_pil = ImageOps.exif_transpose(original_pil).convert("RGB")

    det = detector or get_detector()
    faces = det.detect_faces(original_pil)
    face_count = len(faces)

    confidence = 0.0
    area_pct = 0.0

    if face_count == 0:
        status = FaceValidationStatus.NO_FACE
        is_valid = False
        message = "No human face detected. Please upload an image containing a clearly visible human face."
        annotated_image = original_pil.copy()
        face_crop = None
        crop_bbox = None

    elif face_count == 1:
        status = FaceValidationStatus.SINGLE_FACE
        is_valid = True
        message = "One human face detected. Proceeding with Navarasa classification."
        face_crop, crop_bbox = extract_face_crop(
            original_pil,
            faces[0].bbox,
            padding_ratio=padding_ratio,
            target_size=None,
            square=True,
        )
        annotated_image = annotate_image_with_bboxes(original_pil, faces)
        confidence = faces[0].confidence
        area_pct = faces[0].face_area_pct

    else:  # face_count >= 2
        status = FaceValidationStatus.MULTIPLE_FACES
        is_valid = False
        message = "Multiple human faces detected. Please upload an image containing exactly one human face."
        face_crop = None
        crop_bbox = None
        annotated_image = annotate_image_with_bboxes(original_pil, faces)

    result = FaceValidationResult(
        status=status,
        face_count=face_count,
        is_valid=is_valid,
        message=message,
        original_image=original_pil,
        annotated_image=annotated_image,
        face_crop=face_crop,
        crop_bbox=crop_bbox,
        detected_faces=faces,
        image_hash=img_hash,
        confidence=confidence,
        face_area_pct=area_pct,
    )

    _VALIDATION_CACHE[cache_key] = result
    return result


def get_cached_face_validation(
    image_input: Union[Image.Image, np.ndarray, str, Path, bytes],
    padding_ratio: float = 0.22,
) -> FaceValidationResult:
    """Public wrapper to obtain or compute face validation for an image."""
    return validate_face_image(image_input, padding_ratio=padding_ratio)
