"""
Face Detection and Validation Package for NAVARASA AI.
Provides strict human face verification, bounding box extraction,
and normalized facial cropping for machine learning pipelines.
"""

from src.face_detection.detector import HumanFaceDetector, DetectedFace
from src.face_detection.face_crop import extract_face_crop, annotate_image_with_bboxes
from src.face_detection.face_validation import (
    FaceValidationStatus,
    FaceValidationResult,
    validate_face_image,
    get_cached_face_validation,
    compute_image_hash,
)
from src.face_detection.ui import render_face_validation_card

__all__ = [
    "HumanFaceDetector",
    "DetectedFace",
    "extract_face_crop",
    "annotate_image_with_bboxes",
    "FaceValidationStatus",
    "FaceValidationResult",
    "validate_face_image",
    "get_cached_face_validation",
    "render_face_validation_card",
]

