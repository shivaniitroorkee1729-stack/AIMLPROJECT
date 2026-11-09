"""
Streamlit UI Helper for Strict Human Face Validation.
Renders the standard 3-case visual display and controls execution flow.
"""

from typing import Optional
from PIL import Image
import streamlit as st

from src.face_detection.face_validation import (
    validate_face_image,
    FaceValidationResult,
    FaceValidationStatus,
)


def render_face_validation_card(input_image: Optional[Image.Image]) -> Optional[FaceValidationResult]:
    """
    Renders standardized face validation UI for Streamlit pages.

    Behavior:
    - If input_image is None: returns None.
    - Case A (0 faces):
        Shows error banner: 'No human face detected. Please upload an image containing a clearly visible human face.'
        Displays original rejected image.
        Returns result with is_valid=False.
    - Case B (1 face):
        Shows success banner: 'One human face detected. Proceeding with Navarasa classification.'
        Displays triad: 1. Original Image, 2. Bounding Box Detection, 3. Extracted Facial Crop.
        Returns result with is_valid=True.
    - Case C (2+ faces):
        Shows error banner: 'Multiple human faces detected. Please upload an image containing exactly one human face.'
        Displays original image and annotated image with all bounding boxes and face count.
        Returns result with is_valid=False.
    """
    if input_image is None:
        return None

    # Run validation (hash-cached)
    res: FaceValidationResult = validate_face_image(input_image)

    # Invalidate session state predictions if image hash changed
    if "last_image_hash" not in st.session_state or st.session_state["last_image_hash"] != res.image_hash:
        st.session_state["last_image_hash"] = res.image_hash
        st.session_state.pop("last_prediction_res", None)

    st.markdown("### 🔍 Human Face Detection & Validation")

    if res.status == FaceValidationStatus.NO_FACE:
        st.error(f"❌ **{res.message}**")
        st.markdown(
            """
            <div style="background: #FFF3F3; border-left: 4px solid #D32F2F; padding: 12px 16px; border-radius: 6px; margin-bottom: 12px; font-size: 0.95rem; color: #5C1D1D;">
                <b>Strict Face Policy:</b> The system requires a genuine, clearly visible human face before running CLIP, EfficientNet, Grad-CAM, or facial analysis. 
                Images of non-human objects, animals, blank backgrounds, or obscured faces are strictly rejected.
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_img, col_info = st.columns([1, 1.2])
        with col_img:
            st.image(res.original_image, caption="Uploaded Image (Rejected: 0 Human Faces)", use_container_width=True)
        with col_info:
            st.warning("⚠️ **All ML inference and explainability operations are locked.**")
            with st.expander("🛠️ Detection Diagnostics & Troubleshooting", expanded=True):
                st.write(f"- **Image Dimensions:** {res.original_image.size[0]}×{res.original_image.size[1]} px")
                st.write(f"- **Detected Human Faces:** `{res.face_count}`")
                st.write(f"- **Detector Engine:** `{res.detector_used}`")
                st.write(f"- **Detection Confidence:** `0.00%`")
                st.info("💡 Ensure the face is clearly visible, upright, facing the camera, and reasonably illuminated.")

    elif res.status == FaceValidationStatus.MULTIPLE_FACES:
        st.error(f"⚠️ **{res.message}**")
        st.markdown(
            f"""
            <div style="background: #FFF8E1; border-left: 4px solid #FFA000; padding: 12px 16px; border-radius: 6px; margin-bottom: 12px; font-size: 0.95rem; color: #5C4300;">
                <b>Multi-Face Ambiguity Detected:</b> Found <b>{res.face_count} human faces</b>. 
                Navarasa emotional classification requires single-subject focus to avoid ambiguous or conflicting emotional cues.
            </div>
            """,
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.image(res.original_image, caption="Original Uploaded Image", use_container_width=True)
        with col2:
            st.image(
                res.annotated_image,
                caption=f"All {res.face_count} Detected Faces (Bounding Boxes)",
                use_container_width=True,
            )
        st.warning(f"⚠️ **Total Detected Faces: {res.face_count}**. Please upload an image containing exactly one human face.")
        with st.expander("🛠️ Multi-Face Diagnostics", expanded=False):
            st.write(f"- **Image Dimensions:** {res.original_image.size[0]}×{res.original_image.size[1]} px")
            st.write(f"- **Detected Count:** `{res.face_count}`")
            for idx, face in enumerate(res.detected_faces):
                st.write(f"  - Face #{idx+1}: Box `[{face.bbox[0]}, {face.bbox[1]}, {face.bbox[2]}, {face.bbox[3]}]` | Conf: `{face.confidence*100:.1f}%` | Area: `{face.face_area_pct:.1f}%`")

    else:  # SINGLE_FACE
        st.success(f"✅ **{res.message}**")

        # Key Metrics Row
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Status", "Validated ✅")
        mcol2.metric("Faces Detected", "1 (Single Face)")
        mcol3.metric("Detection Confidence", f"{res.confidence * 100:.1f}%")
        mcol4.metric("Face Area Share", f"{res.face_area_pct:.1f}%")

        # 3-Panel Visual Display
        col_orig, col_det, col_crop = st.columns(3)
        with col_orig:
            st.markdown("##### 1. Original Image")
            st.image(res.original_image, use_container_width=True)
            st.caption(f"Input dimensions: {res.original_image.size[0]}×{res.original_image.size[1]} px")
        with col_det:
            st.markdown("##### 2. Face Bounding Box")
            st.image(res.annotated_image, use_container_width=True)
            if res.crop_bbox:
                bx1, by1, bx2, by2 = res.crop_bbox
                st.caption(f"Box: [{bx1}, {by1}] → [{bx2}, {by2}] | {res.detector_used}")
        with col_crop:
            st.markdown("##### 3. Extracted Facial Crop")
            st.image(res.face_crop, use_container_width=True)
            st.caption(f"Extracted Crop: {res.face_crop.size[0]}×{res.face_crop.size[1]} px (Fed to ML & XAI)")

    return res

