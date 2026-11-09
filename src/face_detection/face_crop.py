"""
Facial Region Extraction and Visualization Module.
Extracts proportional, aspect-ratio-preserving face crops and
renders visual bounding boxes with detection metadata.
"""

from typing import Tuple, List, Optional, Union, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def extract_face_crop(
    image: Union[Image.Image, np.ndarray],
    bbox: Tuple[int, int, int, int],
    padding_ratio: float = 0.22,
    target_size: Optional[Tuple[int, int]] = None,
    square: bool = True,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Extracts a proportional, padded facial crop from the image.

    Args:
        image: Original RGB PIL Image or numpy array.
        bbox: Bounding box tuple (x_min, y_min, x_max, y_max).
        padding_ratio: Proportion of padding around the face (default: 22%).
        target_size: Optional (width, height) to resize the final crop.
        square: Whether to force a square crop for standard model aspect ratios.

    Returns:
        Tuple of (cropped_image_pil, padded_bbox_tuple).
    """
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image).convert("RGB")
    else:
        pil_img = image.convert("RGB")

    w, h = pil_img.size
    x_min, y_min, x_max, y_max = bbox

    bw = x_max - x_min
    bh = y_max - y_min

    # Facial center, shifted slightly upward to accommodate full forehead/hairline
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0 - 0.04 * bh

    if square:
        side = max(bw, bh) * (1.0 + 2.0 * padding_ratio)
        half = side / 2.0
        x1 = int(cx - half)
        y1 = int(cy - half)
        x2 = int(cx + half)
        y2 = int(cy + half)
    else:
        pad_x = bw * padding_ratio
        pad_y = bh * padding_ratio
        x1 = int(x_min - pad_x)
        y1 = int(y_min - pad_y * 1.2)  # extra top padding for forehead
        x2 = int(x_max + pad_x)
        y2 = int(y_max + pad_y * 0.8)  # slight bottom padding for chin

    # Clamp coordinates to image boundaries
    x1_clamped = max(0, x1)
    y1_clamped = max(0, y1)
    x2_clamped = min(w, x2)
    y2_clamped = min(h, y2)

    # Ensure valid crop area
    if x2_clamped <= x1_clamped or y2_clamped <= y1_clamped:
        # Fallback to direct bbox clamping
        x1_clamped = max(0, x_min)
        y1_clamped = max(0, y_min)
        x2_clamped = min(w, x_max)
        y2_clamped = min(h, y_max)

    crop = pil_img.crop((x1_clamped, y1_clamped, x2_clamped, y2_clamped))

    if target_size is not None:
        crop = crop.resize(target_size, Image.Resampling.LANCZOS)

    return crop, (x1_clamped, y1_clamped, x2_clamped, y2_clamped)


def annotate_image_with_bboxes(
    image: Union[Image.Image, np.ndarray],
    faces: List[Any],
    box_color: Optional[Tuple[int, int, int]] = None,
    line_width: int = 4,
) -> Image.Image:
    """
    Renders bounding boxes and index tags on the image for all detected faces.

    Args:
        image: Original RGB PIL Image or numpy array.
        faces: List of DetectedFace objects.
        box_color: RGB tuple for box outline. Defaults to Green if single face, Red if multiple.
        line_width: Width of the bounding box borders.

    Returns:
        Annotated PIL Image.
    """
    if isinstance(image, np.ndarray):
        annotated = Image.fromarray(image).convert("RGB")
    else:
        annotated = image.copy().convert("RGB")

    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    # Choose color based on face count: Emerald green for 1 face, Crimson red for multiple
    num_faces = len(faces)
    if box_color is None:
        if num_faces == 1:
            box_color = (34, 139, 34)   # Forest green
            label_bg = (34, 139, 34)
            text_color = (255, 255, 255)
        else:
            box_color = (220, 20, 60)   # Crimson red
            label_bg = (220, 20, 60)
            text_color = (255, 255, 255)
    else:
        label_bg = box_color
        text_color = (255, 255, 255)

    # Use default font or fallback
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, face in enumerate(faces):
        xmin, ymin, xmax, ymax = face.bbox

        # Draw outer rectangle
        for offset in range(line_width):
            draw.rectangle(
                [xmin - offset, ymin - offset, xmax + offset, ymax + offset],
                outline=box_color,
            )

        # Label badge
        if num_faces == 1:
            label_text = f" Human Face #1 (Validated) "
        else:
            label_text = f" Face #{i+1} "

        # Compute label position
        label_w = len(label_text) * 8 + 8
        label_h = 20
        label_y1 = max(0, ymin - label_h - 2)
        label_y2 = label_y1 + label_h

        draw.rectangle(
            [xmin, label_y1, xmin + label_w, label_y2],
            fill=label_bg,
        )
        draw.text(
            (xmin + 4, label_y1 + 3),
            label_text,
            fill=text_color,
            font=font,
        )

    # If multiple faces, draw a prominent alert banner at top
    if num_faces >= 2:
        banner_h = 32
        draw.rectangle([0, 0, w, banner_h], fill=(220, 20, 60))
        banner_text = f" MULTIPLE FACES DETECTED ({num_faces}) — EXACTLY ONE REQUIRED "
        draw.text((12, 8), banner_text, fill=(255, 255, 255), font=font)

    return annotated
