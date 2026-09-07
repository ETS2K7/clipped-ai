import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import get_logger, FONTS_DIR, DEFAULT_FONT_PATH

logger = get_logger(__name__)

STYLE_PRESETS = {
    "hormozi": {
        "font_name": "Komika Axis",
        "font_size": 110,
        "primary_color": "&H00FFFFFF",      # White text
        "outline_color": "&H00000000",      # Black outline
        "outline_width": 6,
        "highlight_colors": ["&H0000FFFF", "&H0000FF00"], # Yellow & Green
        "blur": 4,
    },
    "minimal": {
        "font_name": "Arial",
        "font_size": 95,
        "primary_color": "&H00F0F0F0",
        "outline_color": "&H00202020",
        "outline_width": 4,
        "highlight_colors": ["&H0033CCFF"], # Soft Gold
        "blur": 2,
    },
    "cyber": {
        "font_name": "Impact",
        "font_size": 115,
        "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "outline_width": 7,
        "highlight_colors": ["&H00FFFF00", "&H00FF00FF"], # Cyan & Magenta
        "blur": 6,
    },
}


def ms_to_ass_time(ms: float) -> str:
    """Converts milliseconds to ASS timestamp format: H:MM:SS.CC"""
    ms = max(0.0, ms)
    hours = int(ms // 3600000)
    minutes = int((ms % 3600000) // 60000)
    seconds = int((ms % 60000) // 1000)
    centiseconds = int((ms % 1000) // 10)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def generate_karaoke_ass(
    words: List[Dict[str, Any]],
    clip_start_s: float,
    clip_end_s: float,
    output_ass_path: str,
    preset_name: str = "hormozi",
) -> str:
    """
    Generates an Advanced SubStation Alpha (.ass) subtitle file with word-by-word
    karaoke glow animation and mobile safe zone positioning.
    """
    style = STYLE_PRESETS.get(preset_name, STYLE_PRESETS["hormozi"])
    clip_start_ms = clip_start_s * 1000.0
    clip_end_ms = clip_end_s * 1000.0

    # Filter words belonging to this clip
    clip_words = [
        w for w in words
        if w.get("start", 0) >= clip_start_ms and w.get("end", 0) <= clip_end_ms
    ]

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ViralCaptions,{style['font_name']},{style['font_size']},{style['primary_color']},&H000000FF,{style['outline_color']},&H80000000,-1,0,0,0,100,100,0,0,1,{style['outline_width']},0,2,40,40,480,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # Group words into 2-3 word chunks based on punctuation and pauses
    chunks = []
    current_chunk = []
    max_words_per_chunk = 2
    max_pause_ms = 300

    for i, w in enumerate(clip_words):
        current_chunk.append(w)
        is_last = (i == len(clip_words) - 1)

        if not is_last:
            next_w = clip_words[i + 1]
            pause_dur = next_w.get("start", 0) - w.get("end", 0)
            ends_punct = any(str(w["text"]).endswith(p) for p in [".", "?", "!", ","])
            too_long = len(current_chunk) >= max_words_per_chunk
            long_pause = pause_dur > max_pause_ms

            if ends_punct or too_long or long_pause:
                chunks.append(current_chunk)
                current_chunk = []
        else:
            chunks.append(current_chunk)

    dialogue_lines = []
    highlight_colors = style["highlight_colors"]

    for chunk_idx, chunk in enumerate(chunks):
        next_chunk_start = (
            chunks[chunk_idx + 1][0].get("start", float("inf"))
            if chunk_idx + 1 < len(chunks)
            else float("inf")
        )

        for w_idx, active_word in enumerate(chunk):
            w_start = active_word.get("start", 0) - clip_start_ms
            if w_idx < len(chunk) - 1:
                w_end = max(w_start + 10, chunk[w_idx + 1].get("start", 0) - clip_start_ms)
            else:
                w_end = active_word.get("end", 0) - clip_start_ms
                actual_end = active_word.get("end", 0)
                pause_to_next = next_chunk_start - actual_end
                if pause_to_next > 0:
                    w_end += min(pause_to_next, 400)

            w_start = max(0, w_start)
            w_end = max(0, min(w_end, clip_end_ms - clip_start_ms))
            if w_end <= w_start:
                continue

            ass_start = ms_to_ass_time(w_start)
            ass_end = ms_to_ass_time(w_end)

            text_parts = []
            for j, word in enumerate(chunk):
                raw_txt = str(word.get("text", "")).upper()
                clean_txt = raw_txt.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

                if j == w_idx:
                    hl_col = highlight_colors[w_idx % len(highlight_colors)]
                    blur_val = style["blur"]
                    # Glowing highlighted word
                    text_parts.append(f"{{\\c{hl_col}\\4c{hl_col}\\blur{blur_val}}}{clean_txt}{{\\r}}")
                else:
                    text_parts.append(clean_txt)

            full_text = " ".join(text_parts)
            dialogue_lines.append(f"Dialogue: 0,{ass_start},{ass_end},ViralCaptions,,0,0,0,,{full_text}")

    Path(output_ass_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        for line in dialogue_lines:
            f.write(line + "\n")

    logger.info("Generated karaoke ASS subtitles: %s (%d events)", output_ass_path, len(dialogue_lines))
    return output_ass_path


def burn_subtitles_to_video(
    video_path: str,
    ass_path: str,
    output_path: str,
    fonts_dir: Optional[str] = None,
) -> str:
    """
    Burns .ass subtitles into the portrait video with font directory support.
    """
    fonts_folder = fonts_dir or str(FONTS_DIR)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Escape path characters for ffmpeg subtitle filter
    escaped_ass = ass_path.replace(":", "\\:").replace("'", "\\'")
    escaped_fonts = fonts_folder.replace(":", "\\:").replace("'", "\\'")

    filter_str = f"subtitles='{escaped_ass}':fontsdir='{escaped_fonts}'"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", filter_str,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path,
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        logger.info("Successfully burned subtitles into: %s", output_path)
        return output_path
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="replace")[-1000:] if e.stderr else str(e)
        logger.error("Failed to burn subtitles via FFmpeg: %s", err)
        raise RuntimeError(f"FFmpeg subtitle burn failed: {err}")
