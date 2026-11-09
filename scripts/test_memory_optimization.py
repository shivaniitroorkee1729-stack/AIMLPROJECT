"""
Automated Memory Optimization & Stress Test Suite for NAVARASA AI.
Validates that peak process RSS RAM remains strictly below the Streamlit Community Cloud
limit (~1024 MB) across all pipelines, explainability modules, and repeated inference loops.
"""

import sys
import gc
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import PROCESSED_DATA_DIR, NAVARASA_CLASSES, CLASS_TO_IDX
from src.utils.memory import get_process_memory_mb, run_garbage_collection, MemoryTracker
from src.face_detection.face_validation import validate_face_image, _VALIDATION_CACHE, MAX_VALIDATION_CACHE_SIZE
from src.models.model_loader import (
    get_efficientnet_model,
    get_clip_model,
    get_gradcam_engine,
    get_facial_analyzer,
    get_au_extractor,
    get_occlusion_analyzer,
    get_counterfactual_analyzer,
    get_natural_language_explainer,
    get_shared_face_landmarker,
)
from src.data.preprocessing import get_efficientnet_transforms


def run_memory_benchmark():
    print("=" * 70)
    print("NAVARASA AI — STREAMLIT CLOUD MEMORY PROFILING & STRESS TEST")
    print("=" * 70)

    tracker = MemoryTracker("Navarasa Comprehensive Pipeline")

    # 1. Baseline
    run_garbage_collection()
    m_baseline = tracker.checkpoint("0. Baseline (Imports Only)")
    print(f"[Step 0] Initial Process Memory: {m_baseline:.2f} MB")

    # Find a real sample image from processed dataset
    sample_img_path = None
    for c in NAVARASA_CLASSES:
        p = PROCESSED_DATA_DIR / c
        if p.exists():
            imgs = list(p.glob("*.jpg"))
            if imgs:
                sample_img_path = imgs[0]
                break

    if sample_img_path and sample_img_path.exists():
        input_image = Image.open(sample_img_path).convert("RGB")
        print(f"Loaded authentic sample image: {sample_img_path.name} ({input_image.size})")
    else:
        # Create a synthetic 800x800 test image if dataset not available
        input_image = Image.fromarray(np.random.randint(60, 200, (800, 800, 3), dtype=np.uint8))
        print("Using synthetic 800x800 RGB test image.")

    # 2. Face Detection Layer
    val_res = validate_face_image(input_image)
    m_face = tracker.checkpoint("1. Face Detection & Validation")
    print(f"[Step 1] After Face Detection: {m_face:.2f} MB (Status: {val_res.status.value}, Faces: {val_res.face_count})")

    face_crop = val_res.face_crop if val_res.is_valid else input_image.resize((224, 224))

    # 3. EfficientNet-B0 Model Loading & Inference
    eff_model = get_efficientnet_model()
    transform = get_efficientnet_transforms(is_training=False)
    tensor = transform(face_crop).unsqueeze(0)
    probs_eff = eff_model.predict_probabilities(tensor)
    pred_idx = int(probs_eff.argmax(dim=-1).item())
    pred_cls = NAVARASA_CLASSES[pred_idx]
    m_eff = tracker.checkpoint("2. EfficientNet-B0 Model + Inference")
    print(f"[Step 2] After EfficientNet-B0: {m_eff:.2f} MB (Prediction: {pred_cls})")

    # 4. Grad-CAM Computation
    gradcam = get_gradcam_engine()
    heatmap, _, _ = gradcam.generate_heatmap(tensor, target_class_idx=pred_idx)
    del tensor
    run_garbage_collection()
    m_gradcam = tracker.checkpoint("3. Grad-CAM Backprop & Heatmap Generation")
    print(f"[Step 3] After Grad-CAM: {m_gradcam:.2f} MB (Heatmap Shape: {heatmap.shape})")

    # 5. MediaPipe Shared FaceLandmarker, 6-Region Masks, & FACS Action Units
    facial_analyzer = get_facial_analyzer()
    landmarks, _ = facial_analyzer.extract_landmarks(face_crop)
    masks = facial_analyzer.generate_region_masks(np.array(face_crop).shape, landmarks)
    reg_pcts, sorted_regs = facial_analyzer.calculate_region_importance(heatmap, masks)

    au_extractor = get_au_extractor()
    aus, _ = au_extractor.extract_action_units(face_crop)
    m_fa = tracker.checkpoint("4. 6-Region Segmentation + FACS Action Units")
    print(f"[Step 4] After Facial Analysis & AUs: {m_fa:.2f} MB (Regions: {len(masks)}, AUs: {len(aus)})")

    # 6. Occlusion Sensitivity Analysis (6 forward passes)
    occ_analyzer = get_occlusion_analyzer()
    occ_res = occ_analyzer.run_sensitivity_analysis(face_crop, masks, mode="gaussian_blur")
    m_occ = tracker.checkpoint("5. Occlusion Sensitivity (6 Perturbed Passes)")
    print(f"[Step 5] After Occlusion Sensitivity: {m_occ:.2f} MB (Top Sensitive: {occ_res['most_sensitive_region']})")

    # 7. Counterfactual Analysis & Natural Language Explainer
    cf_analyzer = get_counterfactual_analyzer()
    cf_res = cf_analyzer.analyze_counterfactual(
        current_class=pred_cls,
        target_class=NAVARASA_CLASSES[(pred_idx + 1) % 9],
        current_aus=aus,
        current_probs={c: float(probs_eff[0, i].item()) for i, c in enumerate(NAVARASA_CLASSES)},
    )
    explainer = get_natural_language_explainer()
    _ = explainer.generate_explanation(
        predicted_class=pred_cls,
        confidence=float(probs_eff[0, pred_idx].item()),
        probabilities={c: float(probs_eff[0, i].item()) for i, c in enumerate(NAVARASA_CLASSES)},
        top_regions=sorted_regs,
        active_action_units=[a for a in aus.values() if a["is_active"]],
    )
    m_xai = tracker.checkpoint("6. Counterfactual + Natural Language Explanations")
    print(f"[Step 6] After Counterfactual & Text Explanations: {m_xai:.2f} MB")

    # 8. CLIP ViT-B/32 Pruned Model Loading & Inference
    clip_model = get_clip_model()
    clip_cls, clip_conf, clip_probs, _ = clip_model.predict(face_crop)
    run_garbage_collection()
    m_clip = tracker.checkpoint("7. CLIP ViT-B/32 (Pruned) + Inference")
    print(f"[Step 7] After Pruned CLIP Loading: {m_clip:.2f} MB (Prediction: {clip_cls})")

    # 9. Repeated Inferences (Memory Leak Check - 15 iterations)
    print("\n--- Running 15-iteration inference leak test ---")
    mem_before_loop = get_process_memory_mb()
    for iter_i in range(1, 16):
        t_crop = transform(face_crop).unsqueeze(0)
        _ = eff_model.predict_probabilities(t_crop)
        hm, _, _ = gradcam.generate_heatmap(t_crop, target_class_idx=pred_idx)
        _ = facial_analyzer.calculate_region_importance(hm, masks)
        _ = clip_model.predict(face_crop)
        del t_crop, hm
        if iter_i % 5 == 0:
            run_garbage_collection()
            cur_mem = get_process_memory_mb()
            print(f"  Iteration {iter_i:2d}/15: Current Memory = {cur_mem:.2f} MB")

    run_garbage_collection()
    m_loop = tracker.checkpoint("8. After 15 Repeated Inference Cycles")
    leak = m_loop - mem_before_loop
    print(f"[Step 8] After 15 Inference Iterations: {m_loop:.2f} MB (Net Drift: {leak:+.2f} MB)")

    # 10. Cache Eviction Test (15 different synthetic images)
    print("\n--- Running LRU Validation Cache Eviction Test (15 images) ---")
    initial_cache_size = len(_VALIDATION_CACHE)
    for i in range(15):
        random_img = Image.fromarray(np.random.randint(50 + i, 200, (400, 400, 3), dtype=np.uint8))
        _ = validate_face_image(random_img)
        del random_img
    
    current_cache_size = len(_VALIDATION_CACHE)
    run_garbage_collection()
    m_evict = tracker.checkpoint("9. After 15 Image Uploads (Cache Eviction)")
    print(f"[Step 9] After 15 Uploads: {m_evict:.2f} MB (Cache Size: {current_cache_size}/{MAX_VALIDATION_CACHE_SIZE})")

    # Final summary & assertion
    print("\n" + "=" * 70)
    print("MEMORY BENCHMARK SUMMARY")
    print("=" * 70)
    for label, val in tracker.history:
        print(f"  {label:<50}: {val:7.2f} MB")

    peak_ram = tracker.peak_mb
    print("-" * 70)
    print(f"PEAK RAM USAGE ACROSS ALL STAGES: {peak_ram:.2f} MB")
    print(f"STREAMLIT COMMUNITY CLOUD LIMIT : 1024.00 MB")
    print(f"SAFETY HEADROOM MARGIN          : {1024.00 - peak_ram:.2f} MB ({((1024.00 - peak_ram) / 1024.00) * 100:.1f}%)")
    print("=" * 70)

    # Validations
    assert peak_ram < 1024.0, f"FAILED: Peak RAM ({peak_ram:.2f} MB) exceeded 1024 MB Streamlit limit!"
    assert current_cache_size <= MAX_VALIDATION_CACHE_SIZE, f"FAILED: Validation cache size ({current_cache_size}) exceeded limit {MAX_VALIDATION_CACHE_SIZE}!"
    assert abs(leak) < 80.0, f"FAILED: Potential memory leak detected (drift: {leak:.2f} MB)!"

    print("\n>>> ALL MEMORY CONSTRAINTS MET! APPLICATION IS FULLY STREAMLIT CLOUD SAFE. <<<")
    return True


if __name__ == "__main__":
    success = run_memory_benchmark()
    sys.exit(0 if success else 1)
