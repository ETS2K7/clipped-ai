import os
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import get_logger, DEFAULT_FONT_PATH, OUT_WIDTH, OUT_HEIGHT

logger = get_logger(__name__)


def extract_hook_frame(video_path: str, timestamp_s: float = 2.0) -> Optional[np.ndarray]:
    """
    Extracts a high-quality video frame at the given timestamp for thumbnail generation.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    target_frame = int(timestamp_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    ret, frame = cap.read()
    if not ret or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    cap.release()
    return frame if ret else None


def generate_hook_thumbnail(
    video_path: str,
    output_image_path: str,
    hook_text: str,
    virality_score: int = 90,
    timestamp_s: float = 2.0,
    aspect_ratio: str = "9:16",
) -> str:
    """
    Generates a high-CTR thumbnail with gradient contrast vignette,
    bold hook typography, and a virality badge.
    Supports both 9:16 vertical and original (e.g. 16:9 widescreen) aspect ratios.
    """
    frame = extract_hook_frame(video_path, timestamp_s)
    if frame is None:
        raise RuntimeError(f"Could not extract frame from video: {video_path}")

    # Determine dimensions based on aspect ratio mode
    if aspect_ratio == "original":
        target_h, target_w = frame.shape[0], frame.shape[1]
    else:
        target_w, target_h = OUT_WIDTH, OUT_HEIGHT
        if frame.shape[0] != OUT_HEIGHT or frame.shape[1] != OUT_WIDTH:
            frame = cv2.resize(frame, (OUT_WIDTH, OUT_HEIGHT))

    # Convert to RGB PIL Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)

    # Add subtle vignette gradient at top and bottom for text legibility
    vignette = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    vignette_draw = ImageDraw.Draw(vignette)

    # Top gradient
    top_vignette_h = int(target_h * 0.22)
    for y in range(top_vignette_h):
        alpha = int(140 * (1.0 - (y / float(max(1, top_vignette_h)))))
        vignette_draw.line([(0, y), (target_w, y)], fill=(0, 0, 0, alpha))

    # Bottom gradient
    bottom_vignette_h = int(target_h * 0.26)
    start_bottom_y = target_h - bottom_vignette_h
    for y in range(start_bottom_y, target_h):
        alpha = int(180 * ((y - start_bottom_y) / float(max(1, bottom_vignette_h))))
        vignette_draw.line([(0, y), (target_w, y)], fill=(0, 0, 0, alpha))

    img = Image.alpha_composite(img.convert("RGBA"), vignette)
    draw = ImageDraw.Draw(img)

    # Format hook text into 2-3 short, bold lines
    words = hook_text.upper().split()
    lines = []
    curr_line = []

    words_per_line = 4 if (aspect_ratio == "original" and target_w > target_h) else 3
    for w in words:
        curr_line.append(w)
        if len(curr_line) >= words_per_line:
            lines.append(" ".join(curr_line))
            curr_line = []
    if curr_line:
        lines.append(" ".join(curr_line))

    # Keep at most top 3 lines
    lines = lines[:3]

    base_font_size = int(round(86 * (target_h / 1920.0 * 1.5))) if aspect_ratio == "original" and target_w > target_h else 86
    font_size = max(42, base_font_size)
    try:
        font_large = ImageFont.truetype(str(DEFAULT_FONT_PATH), font_size)
    except Exception:
        font_large = ImageFont.load_default()

    # Auto-scale font size if any line exceeds frame width
    max_text_boundary = target_w - int(target_w * 0.1)
    if lines:
        max_w = max(
            draw.textbbox((0, 0), l, font=font_large)[2] - draw.textbbox((0, 0), l, font=font_large)[0]
            for l in lines
        )
        if max_w > max_text_boundary:
            scale = max_text_boundary / float(max_w)
            font_size = max(36, int(font_size * scale))
            try:
                font_large = ImageFont.truetype(str(DEFAULT_FONT_PATH), font_size)
            except Exception:
                font_large = ImageFont.load_default()

    # Center hook text vertically and horizontally
    line_spacing = int(font_size * 1.28)
    total_text_h = (len(lines) - 1) * line_spacing + font_size
    start_y = (target_h - total_text_h) // 2

    outline_w = max(4, int(round(8 * (target_h / 1920.0)))) if aspect_ratio == "original" else 8
    for idx, line in enumerate(lines):
        # Calculate text bounding box to center horizontally
        bbox = draw.textbbox((0, 0), line, font=font_large)
        text_w = bbox[2] - bbox[0]
        pos_x = (target_w - text_w) // 2
        pos_y = start_y + (idx * line_spacing)

        # Highlight color for second line
        fill_color = (255, 255, 0, 255) if idx == 1 else (255, 255, 255, 255)

        # Thick black outline for pop
        for ox in range(-outline_w, outline_w + 1):
            for oy in range(-outline_w, outline_w + 1):
                if ox != 0 or oy != 0:
                    draw.text((pos_x + ox, pos_y + oy), line, font=font_large, fill=(0, 0, 0, 255))

        draw.text((pos_x, pos_y), line, font=font_large, fill=fill_color)

    Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(output_image_path, "JPEG", quality=92)
    logger.info("Generated hook thumbnail: %s", output_image_path)
    return output_image_path
