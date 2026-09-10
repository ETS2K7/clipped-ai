"""
Module for generating dynamic ASS subtitles with karaoke animations and layout boundaries.
"""

from typing import List, Dict, Any, Optional
import re
from config import get_logger

logger = get_logger(__name__)

# ─── Default subtitle styling ─────────────────────────────────────────────────
DEFAULT_FONT_FAMILY = "Komika Axis"
DEFAULT_FONT_SIZE = 115
DEFAULT_FONT_COLOR = "&H00FFFFFF"   # ASS format: white

FONT_NAME_RE = re.compile(r"^[A-Za-z0-9 _.-]{1,128}$")


def hex_to_ass_color(value: Optional[str]) -> str:
    """Convert #RRGGBB to ASS &H00BBGGRR format."""
    if not value or not re.fullmatch(r"#[0-9A-Fa-f]{6}", value):
        return DEFAULT_FONT_COLOR
    red = value[1:3]
    green = value[3:5]
    blue = value[5:7]
    return f"&H00{blue}{green}{red}".upper()


def resolve_font_family(value: Optional[str]) -> str:
    if not value:
        return DEFAULT_FONT_FAMILY
    cleaned = value.replace("_", " ").strip()
    return cleaned if FONT_NAME_RE.fullmatch(cleaned) else DEFAULT_FONT_FAMILY


def ms_to_ass_time(ms: float) -> str:
    """Converts milliseconds to ASS video format (H:MM:SS.CC)."""
    hours = int(ms // 3600000)
    minutes = int((ms % 3600000) // 60000)
    seconds = int((ms % 60000) // 1000)
    cents = int((ms % 1000) // 10)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{cents:02d}"


def generate_subtitles(
    words: List[Dict[str, Any]],
    clip: Dict[str, Any],
    idx: int,
    framing_meta: List[Dict[str, Any]],
    aspect_ratio: str = "9:16",
    video_width: int = 1080,
    video_height: int = 1920,
    font_family: Optional[str] = None,
    font_size: Optional[int] = None,
    font_color: Optional[str] = None,
    work_dir: str = "",
) -> str:
    """
    Generates an .ASS subtitle file dynamically mapping words iteratively to
    the bounding box framing logic dictating its positional styling.

    Supports both 9:16 vertical (Shorts/Reels/TikTok) and original widescreen layouts.
    Accepts optional font configuration from the frontend typography bridge:
      - font_family: e.g. "Montserrat", "Impact" (default: "Komika Axis")
      - font_size:   e.g. 40, 60, 115
      - font_color:  hex string e.g. "#FFFFFF", "#FFD700" (default: white)
    """
    logger.info(
        f"==================== PHASE 6: SUBTITLE GENERATION (Clip {idx}) ===================="
    )
    out = f"{work_dir}/temp_subtitles_{idx}.ass" if work_dir else f"temp_subtitles_{idx}.ass"
    start_ms = clip["start_time"] * 1000
    end_ms = clip["end_time"] * 1000

    # Configure canvas dimensions and responsive layout geometry
    is_original = (aspect_ratio == "original")
    res_x = int(video_width) if (is_original and video_width > 0) else 1080
    res_y = int(video_height) if (is_original and video_height > 0) else 1920

    resolved_family = resolve_font_family(font_family)
    if font_size and isinstance(font_size, int) and 30 <= font_size <= 250:
        resolved_size = font_size
    elif is_original:
        # Scale proportionally to vertical resolution for landscape (e.g. ~64pt at 1080p)
        resolved_size = max(36, int(round(DEFAULT_FONT_SIZE * (res_y / 1920.0))))
    else:
        resolved_size = DEFAULT_FONT_SIZE

    margin_v = int(res_y * 0.08) if is_original else 450
    outline_val = max(3, int(round(6 * (res_y / 1920.0)))) if is_original else 6
    resolved_ass_color = hex_to_ass_color(font_color)

    logger.info(
        f"Subtitle style locked: {resolved_family} / {resolved_size}pt / "
        f"canvas={res_x}x{res_y} / margin_v={margin_v} / color={resolved_ass_color}"
    )

    def get_layout_for_time(ms: float) -> str:
        for meta in framing_meta:
            if meta["start_ms"] <= ms <= meta["end_ms"]:
                return meta["flag"]
        return "SINGLE"

    clip_words = [
        w for w in words if w.get("start", 0) >= start_ms and w.get("end", 0) <= end_ms
    ]

    # —— Romanized Hindi Mapping Pass (V6: Anchor-Pair Sync Lock) ——
    # Uses 'Roman:Original' pairs to lock transliterations to exact timestamps.
    rom_input = clip.get("romanized_words")
    if isinstance(rom_input, list) and len(clip_words) > 0:
        logger.info(f"Applying Anchor-Pair sync lock for clip {idx} ({len(rom_input)} segments)")
        word_idx = 0
        for segment in rom_input:
            segment_pairs = [p.strip() for p in segment.split("|") if ":" in p]
            for pair in segment_pairs:
                if word_idx >= len(clip_words):
                    break
                parts = pair.split(":", 1)
                if len(parts) == 2:
                    roman = parts[0].strip()
                    clip_words[word_idx]["text"] = roman
                word_idx += 1
    elif clip.get("romanized_transcript"):
        # Fallback to V1 (Space-separated string) for backward compatibility with cached clips
        rom_words = str(clip["romanized_transcript"]).strip().split()
        if abs(len(rom_words) - len(clip_words)) <= max(2, len(clip_words) // 5):
            for i in range(min(len(rom_words), len(clip_words))):
                clip_words[i]["text"] = rom_words[i]

    # Dynamically build ASS header with resolved font and canvas configuration
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Hormozi,{resolved_family},{resolved_size},{resolved_ass_color},&H000000FF,&H00000000,&H80000000,-1,0,0,0,110,100,0,0,1,{outline_val},0,2,10,10,{margin_v},1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = []
    chunks = []
    current_chunk = []
    max_words_per_chunk = 2
    max_pause_ms = 300

    # 1. Group words chunks dynamically based on pause timers + string punctuation max length breaks
    for i, w in enumerate(clip_words):
        current_chunk.append(w)
        is_last = i == len(clip_words) - 1

        if not is_last:
            next_w = clip_words[i + 1]
            pause_dur = next_w.get("start", 0) - w.get("end", 0)
            ends_with_punct = any(str(w["text"]).endswith(p) for p in [".", "?", "!"])
            too_long = len(current_chunk) >= max_words_per_chunk
            long_pause = pause_dur > max_pause_ms

            if ends_with_punct or too_long or long_pause:
                chunks.append(current_chunk)
                current_chunk = []
        else:
            chunks.append(current_chunk)

    for c_idx, chunk in enumerate(chunks):
        next_chunk_start = (
            chunks[c_idx + 1][0].get("start", float("inf"))
            if c_idx + 1 < len(chunks)
            else float("inf")
        )

        for w_idx, w in enumerate(chunk):
            w_start = w.get("start", 0) - start_ms

            if w_idx < len(chunk) - 1:
                w_end = max(w_start + 10, chunk[w_idx + 1].get("start", 0) - start_ms)
            else:
                w_end = w.get("end", 0) - start_ms
                actual_end = w.get("end", 0)
                pause_to_next_chunk = next_chunk_start - actual_end

                # Dynamic soft padding
                if pause_to_next_chunk > 0:
                    pad = min(pause_to_next_chunk, 400)
                    w_end += pad

            w_start = max(0, w_start)
            w_end = max(0, min(w_end, end_ms - start_ms))

            if w_end <= w_start:
                continue

            ass_start = ms_to_ass_time(w_start)
            ass_end = ms_to_ass_time(w_end)
            layout = get_layout_for_time(w_start)

            text_parts = []
            
            if not is_original:
                if layout == "SPLIT":
                    text_parts.append("{\\an5\\pos(540,960)}")

            for j, cw in enumerate(chunk):
                # Escape ASS special syntax characters to prevent subtitle corruption
                raw_txt = str(cw.get("text", "")).upper()
                clean_txt = raw_txt.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
                if j == w_idx:
                    # Alternate highlight color between green and yellow based on word index
                    hl_color = "&H0000FFFF" if w_idx % 2 == 0 else "&H0000FF00"
                    # Add Neon Glow Effect (\4c for shadow color, \blur for soft aura)
                    text_parts.append(
                        f"{{\\c{hl_color}\\4c{hl_color}\\blur5}}{clean_txt}{{\\r}}"
                    )
                else:
                    text_parts.append(clean_txt)

            full_text = " ".join(text_parts)
            line = f"Dialogue: 0,{ass_start},{ass_end},Hormozi,,0,0,0,,{full_text}"
            lines.append(line)

    with open(out, "w", encoding="utf-8") as f:
        f.write(header)
        for line in lines:
            f.write(line + "\n")

    return out


def generate_karaoke_ass(
    words: List[Dict[str, Any]],
    clip_start_s: float,
    clip_end_s: float,
    output_ass_path: str,
    framing_meta: Optional[List[Dict[str, Any]]] = None,
    aspect_ratio: str = "9:16",
    video_width: int = 1080,
    video_height: int = 1920,
    preset_name: str = "hormozi",
) -> str:
    """Compatibility wrapper for generate_subtitles supporting direct path output."""
    clip_dict = {"start_time": clip_start_s, "end_time": clip_end_s}
    import tempfile
    import shutil
    with tempfile.TemporaryDirectory() as td:
        out_file = generate_subtitles(
            words=words,
            clip=clip_dict,
            idx=0,
            framing_meta=framing_meta or [],
            aspect_ratio=aspect_ratio,
            video_width=video_width,
            video_height=video_height,
            work_dir=td,
        )
        shutil.copy2(out_file, output_ass_path)
    return output_ass_path


def burn_subtitles_to_video(
    video_path: str,
    ass_path: str,
    output_path: str,
    fonts_dir: Optional[str] = None,
) -> str:
    """Burns ASS subtitles into video using FFmpeg libass filter."""
    import subprocess
    import pathlib
    safe_ass = str(pathlib.Path(ass_path).resolve()).replace("\\", "/").replace(":", "\\:")
    vf_filter = f"ass={safe_ass}"
    if fonts_dir and pathlib.Path(fonts_dir).exists():
        safe_fonts = str(pathlib.Path(fonts_dir).resolve()).replace("\\", "/").replace(":", "\\:")
        vf_filter += f":fontsdir={safe_fonts}"
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", vf_filter,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_path
