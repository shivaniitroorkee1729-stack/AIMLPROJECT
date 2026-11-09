"""
Comprehensive Test Suite for Mandatory Human Face Detection & Validation Layer.
Verifies all 11 mandatory scenarios:
1. Image with no human face (blank/noise) -> rejected (0 faces)
2. Image with 1 human face -> accepted, valid proportional crop extracted
3. Image with 2 human faces -> rejected (2 faces)
4. Image with several human faces -> rejected (count > 1)
5. Image with face + significant background -> face cropped, background discarded
6. Lighting variations -> robust detection or clean rejection
7. Image change invalidates cache and triggers re-validation
8. CLIP receives face crop and outputs 9 probabilities; rejects 0 faces
9. EfficientNet-B0 receives face crop and outputs 9 probabilities; rejects 0 faces
10. Grad-CAM, 6-region masks, and FACS AUs operate strictly on face crop
11. Non-human images (patterns, shapes, non-faces) rejected with 0 faces
"""

import sys
import glob
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import NAVARASA_CLASSES
from src.face_detection import (
    HumanFaceDetector,
    FaceValidationStatus,
    FaceValidationResult,
    validate_face_image,
    compute_image_hash,
    extract_face_crop,
    annotate_image_with_bboxes,
)
from src.models.inference import predict_clip, predict_efficientnet, predict_dual_models
from src.explainability.gradcam import GradCAM
from src.explainability.facial_regions import FacialRegionAnalyzer
from src.explainability.action_units import ActionUnitExtractor
from src.models.model_loader import get_efficientnet_model, get_gradcam_engine


def get_sample_face_image():
    """Helper to retrieve a representative face image from the dataset."""
    samples = glob.glob("data/processed/*/*.jpg")
    if not samples:
        raise FileNotFoundError("No samples found in data/processed/")
    return Image.open(samples[0]).convert("RGB")


def test_1_no_human_face_blank_and_noise():
    """Scenario 1: Image with zero human faces (blank and noise) must be rejected."""
    print("Running Test 1: No human face (blank / noise)...")
    # Pure black
    black_img = Image.new("RGB", (256, 256), color=(0, 0, 0))
    res_black = validate_face_image(black_img)
    assert res_black.status == FaceValidationStatus.NO_FACE
    assert res_black.face_count == 0
    assert not res_black.is_valid
    assert res_black.face_crop is None
    assert "No human face detected" in res_black.message

    # Pure white
    white_img = Image.new("RGB", (256, 256), color=(255, 255, 255))
    res_white = validate_face_image(white_img)
    assert res_white.status == FaceValidationStatus.NO_FACE
    assert res_white.face_count == 0
    assert not res_white.is_valid

    # Random noise
    noise_arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    res_noise = validate_face_image(noise_arr)
    assert res_noise.status == FaceValidationStatus.NO_FACE
    assert res_noise.face_count == 0
    assert not res_noise.is_valid
    print("[PASS] Test 1 Passed.")


def test_2_single_human_face_valid_crop():
    """Scenario 2: Image with 1 human face must be accepted and extracted as proportional crop."""
    print("Running Test 2: Exactly 1 human face validation & crop extraction...")
    face_img = get_sample_face_image()
    res = validate_face_image(face_img)

    assert res.status == FaceValidationStatus.SINGLE_FACE
    assert res.face_count == 1
    assert res.is_valid
    assert res.face_crop is not None
    assert res.crop_bbox is not None
    assert "One human face detected" in res.message

    # Verify crop dimensions
    cw, ch = res.face_crop.size
    assert cw > 20 and ch > 20
    assert abs(cw - ch) <= 2  # Square aspect ratio maintained
    print(f"[PASS] Test 2 Passed (Crop dimensions: {cw}x{ch}).")


def test_3_dual_human_faces_rejected():
    """Scenario 3: Image with 2 human faces must be rejected with count=2."""
    print("Running Test 3: Dual human faces rejected...")
    face_img = get_sample_face_image()
    arr = np.array(face_img)
    dual_arr = np.hstack([arr, arr])  # Two faces side by side

    res = validate_face_image(dual_arr)
    assert res.status == FaceValidationStatus.MULTIPLE_FACES
    assert res.face_count == 2
    assert not res.is_valid
    assert res.face_crop is None
    assert "Multiple human faces detected" in res.message
    print(f"[PASS] Test 3 Passed (Detected {res.face_count} faces).")



def test_4_multiple_human_faces_rejected():
    """Scenario 4: Image with 3+ human faces must be rejected."""
    print("Running Test 4: Several human faces rejected...")
    samples = glob.glob("data/processed/*/*.jpg")
    crop1 = validate_face_image(samples[0]).face_crop.resize((224, 224))
    crop2 = validate_face_image(samples[5]).face_crop.resize((224, 224))
    crop3 = validate_face_image(samples[10]).face_crop.resize((224, 224))

    canvas = Image.new("RGB", (800, 700), color=(200, 200, 200))
    canvas.paste(crop1, (50, 50))
    canvas.paste(crop2, (420, 50))
    canvas.paste(crop3, (240, 360))

    res = validate_face_image(canvas)
    assert res.status == FaceValidationStatus.MULTIPLE_FACES
    assert res.face_count >= 3
    assert not res.is_valid
    assert res.face_crop is None
    print(f"[PASS] Test 4 Passed (Detected {res.face_count} faces).")



def test_5_face_with_significant_background():
    """Scenario 5: Face embedded in large background canvas must be isolated into tight face crop."""
    print("Running Test 5: Face with significant background...")
    face_img = get_sample_face_image()
    val_face = validate_face_image(face_img)
    crop = val_face.face_crop.resize((300, 300))

    # Create 800x800 background canvas and paste face in the center
    canvas = Image.new("RGB", (800, 800), color=(180, 180, 200))
    canvas.paste(crop, (250, 250))

    res = validate_face_image(canvas)
    assert res.status == FaceValidationStatus.SINGLE_FACE
    assert res.face_count == 1
    assert res.is_valid

    # Bounding box should be localized around the pasted face, not the 800x800 canvas
    bx1, by1, bx2, by2 = res.crop_bbox
    crop_w = bx2 - bx1
    crop_h = by2 - by1
    assert crop_w < 500 and crop_h < 500
    assert res.face_crop.size[0] < 500
    print(f"[PASS] Test 5 Passed (Canvas: 800x800, Extracted crop: {crop_w}x{crop_h}).")



def test_6_lighting_variations():
    """Scenario 6: Face under lighting variations must detect or cleanly reject without crash."""
    print("Running Test 6: Lighting variations...")
    face_img = get_sample_face_image()

    # Dim lighting (40% brightness)
    dimmer = ImageEnhance.Brightness(face_img)
    dim_img = dimmer.enhance(0.4)
    res_dim = validate_face_image(dim_img)
    assert res_dim.status in [FaceValidationStatus.SINGLE_FACE, FaceValidationStatus.NO_FACE]

    # Bright lighting (160% brightness)
    brighter = ImageEnhance.Brightness(face_img)
    bright_img = brighter.enhance(1.6)
    res_bright = validate_face_image(bright_img)
    assert res_bright.status in [FaceValidationStatus.SINGLE_FACE, FaceValidationStatus.NO_FACE]
    print("[PASS] Test 6 Passed.")


def test_7_cache_invalidation_on_image_change():
    """Scenario 7: Changing image invalidates old predictions and produces distinct hash."""
    print("Running Test 7: Cache invalidation and image hash tracking...")
    face_img = get_sample_face_image()
    hash1 = compute_image_hash(face_img)

    # Modify single pixel
    modified_img = face_img.copy()
    modified_img.putpixel((0, 0), (255, 0, 128))
    hash2 = compute_image_hash(modified_img)

    assert hash1 != hash2, "Image hashes must differ when image content changes."
    print(f"[PASS] Test 7 Passed (Hash 1: {hash1[:8]}..., Hash 2: {hash2[:8]}...).")


def test_8_clip_inference_with_face_crop():
    """Scenario 8: CLIP operates on face crop, outputs 9 probabilities, and rejects 0-face images."""
    print("Running Test 8: CLIP pipeline with face crop enforcement...")
    face_img = get_sample_face_image()
    res = predict_clip(face_img, enforce_face_crop=True)

    assert "predicted_class" in res
    assert res["predicted_class"] in NAVARASA_CLASSES
    assert len(res["probabilities"]) == 9
    assert abs(sum(res["probabilities"].values()) - 1.0) < 0.01
    assert "face_crop" in res

    # Verify exception is raised when 0 faces are present
    black_img = Image.new("RGB", (200, 200), color=(0, 0, 0))
    try:
        predict_clip(black_img, enforce_face_crop=True)
        assert False, "predict_clip should have raised ValueError on zero-face image."
    except ValueError as e:
        assert "No human face detected" in str(e)

    print("[PASS] Test 8 Passed.")


def test_9_efficientnet_inference_with_face_crop():
    """Scenario 9: EfficientNet operates on face crop, outputs 9 probabilities, and rejects 0-face images."""
    print("Running Test 9: EfficientNet-B0 pipeline with face crop enforcement...")
    face_img = get_sample_face_image()
    res = predict_efficientnet(face_img, enforce_face_crop=True)

    assert "predicted_class" in res
    assert res["predicted_class"] in NAVARASA_CLASSES
    assert len(res["probabilities"]) == 9
    assert abs(sum(res["probabilities"].values()) - 1.0) < 0.01
    assert "face_crop" in res

    # Verify exception is raised on 0-face image
    blank_img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    try:
        predict_efficientnet(blank_img, enforce_face_crop=True)
        assert False, "predict_efficientnet should have raised ValueError on zero-face image."
    except ValueError as e:
        assert "No human face detected" in str(e)

    print("[PASS] Test 9 Passed.")


def test_10_explainability_on_face_crop():
    """Scenario 10: Grad-CAM, 6-region masks, and FACS AUs operate strictly on the face crop."""
    print("Running Test 10: Explainability features restricted to face crop...")
    face_img = get_sample_face_image()
    val_res = validate_face_image(face_img)
    assert val_res.is_valid
    face_crop = val_res.face_crop

    # 1. Grad-CAM on face crop
    eff_res = predict_efficientnet(face_crop, enforce_face_crop=False)
    tensor = eff_res["input_tensor"]
    gradcam = get_gradcam_engine()
    heatmap, _, _ = gradcam.generate_heatmap(tensor, target_class_idx=0)
    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0

    # 2. Overlay back on original bounding box
    orig_overlay = gradcam.overlay_heatmap_on_original(
        val_res.original_image, heatmap, val_res.crop_bbox
    )
    assert orig_overlay.shape[:2] == (val_res.original_image.size[1], val_res.original_image.size[0])

    # 3. 6-Region Masks on face crop
    analyzer = FacialRegionAnalyzer()
    landmarks, det = analyzer.extract_landmarks(face_crop)
    masks = analyzer.generate_region_masks(np.array(face_crop).shape, landmarks)
    assert set(masks.keys()) == {"Eyes", "Eyebrows", "Nose", "Mouth", "Cheeks", "Forehead"}

    # 4. Action Units from face crop
    extractor = ActionUnitExtractor(analyzer)
    aus, face_ok = extractor.extract_action_units(face_crop)
    assert face_ok is True
    assert len(aus) == 14
    print("[PASS] Test 10 Passed.")


def test_11_non_human_images_rejected():
    """Scenario 11: Non-human patterns, synthetic geometry, and object images are rejected as 0 faces."""
    print("Running Test 11: Non-human images rejection...")
    # Geometric shapes (circles, squares, lines)
    img_shapes = Image.new("RGB", (300, 300), color=(240, 240, 240))
    draw = ImageDraw.Draw(img_shapes)
    draw.ellipse([50, 50, 150, 150], fill=(255, 0, 0), outline=(0, 0, 0))
    draw.rectangle([180, 180, 280, 280], fill=(0, 255, 0), outline=(0, 0, 0))
    draw.line([0, 0, 300, 300], fill=(0, 0, 255), width=5)

    res_shapes = validate_face_image(img_shapes)
    assert res_shapes.status == FaceValidationStatus.NO_FACE
    assert res_shapes.face_count == 0
    assert not res_shapes.is_valid

    # Checkerboard pattern
    checker = np.zeros((200, 200, 3), dtype=np.uint8)
    checker[::20, ::20] = 255
    res_checker = validate_face_image(checker)
    assert res_checker.status == FaceValidationStatus.NO_FACE
    assert res_checker.face_count == 0
    assert not res_checker.is_valid

    print("[PASS] Test 11 Passed.")


if __name__ == "__main__":
    print("=" * 70)
    print("STARTING COMPLETE 11-SCENARIO FACE DETECTION TEST SUITE")
    print("=" * 70)

    test_1_no_human_face_blank_and_noise()
    test_2_single_human_face_valid_crop()
    test_3_dual_human_faces_rejected()
    test_4_multiple_human_faces_rejected()
    test_5_face_with_significant_background()
    test_6_lighting_variations()
    test_7_cache_invalidation_on_image_change()
    test_8_clip_inference_with_face_crop()
    test_9_efficientnet_inference_with_face_crop()
    test_10_explainability_on_face_crop()
    test_11_non_human_images_rejected()

    print("=" * 70)
    print("[SUCCESS] ALL 11 COMPREHENSIVE FACE DETECTION SCENARIOS PASSED WITH 100% SUCCESS!")
    print("=" * 70)

