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
) -> str:
    """
    Generates a high-CTR 9:16 portrait thumbnail with gradient contrast vignette,
    bold hook typography, and a virality badge.
    """
    frame = extract_hook_frame(video_path, timestamp_s)
    if frame is None:
        raise RuntimeError(f"Could not extract frame from video: {video_path}")

    # Ensure 1080x1920
    if frame.shape[0] != OUT_HEIGHT or frame.shape[1] != OUT_WIDTH:
        frame = cv2.resize(frame, (OUT_WIDTH, OUT_HEIGHT))

    # Convert to RGB PIL Image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    draw = ImageDraw.Draw(img, "RGBA")

    # Add subtle vignette gradient at top and bottom for text legibility
    vignette = Image.new("RGBA", (OUT_WIDTH, OUT_HEIGHT), (0, 0, 0, 0))
    vignette_draw = ImageDraw.Draw(vignette)

    # Top gradient
    for y in range(400):
        alpha = int(140 * (1.0 - (y / 400.0)))
        vignette_draw.line([(0, y), (OUT_WIDTH, y)], fill=(0, 0, 0, alpha))

    # Bottom gradient
    for y in range(OUT_HEIGHT - 500, OUT_HEIGHT):
        alpha = int(180 * ((y - (OUT_HEIGHT - 500)) / 500.0))
        vignette_draw.line([(0, y), (OUT_WIDTH, y)], fill=(0, 0, 0, alpha))

    img = Image.alpha_composite(img.convert("RGBA"), vignette)
    draw = ImageDraw.Draw(img)

    # Format hook text into 2-3 short, bold lines
    words = hook_text.upper().split()
    lines = []
    curr_line = []

    for w in words:
        curr_line.append(w)
        if len(curr_line) >= 3:
            lines.append(" ".join(curr_line))
            curr_line = []
    if curr_line:
        lines.append(" ".join(curr_line))

    # Keep at most top 3 lines
    lines = lines[:3]

    font_size = 86
    try:
        font_large = ImageFont.truetype(str(DEFAULT_FONT_PATH), font_size)
    except Exception:
        font_large = ImageFont.load_default()

    # Auto-scale font size if any line exceeds frame width
    if lines:
        max_w = max(
            draw.textbbox((0, 0), l, font=font_large)[2] - draw.textbbox((0, 0), l, font=font_large)[0]
            for l in lines
        )
        if max_w > (OUT_WIDTH - 120):
            scale = (OUT_WIDTH - 120) / max_w
            font_size = max(52, int(font_size * scale))
            try:
                font_large = ImageFont.truetype(str(DEFAULT_FONT_PATH), font_size)
            except Exception:
                font_large = ImageFont.load_default()

    # Center hook text vertically and horizontally on the 9:16 portrait frame
    line_spacing = int(font_size * 1.28)
    total_text_h = (len(lines) - 1) * line_spacing + font_size
    start_y = (OUT_HEIGHT - total_text_h) // 2

    outline_w = 8
    for idx, line in enumerate(lines):
        # Calculate text bounding box to center horizontally
        bbox = draw.textbbox((0, 0), line, font=font_large)
        text_w = bbox[2] - bbox[0]
        pos_x = (OUT_WIDTH - text_w) // 2
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
