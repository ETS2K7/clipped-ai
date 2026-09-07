import os
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
from config import get_logger, STORAGE_DIR

logger = get_logger(__name__)

YOUTUBE_URL_REGEX = re.compile(
    r"^(https?://)?(www\.|m\.)?(youtube\.com/(watch\?v=|embed/|v/)|youtu\.be/)([\w-]{11})"
)


def extract_youtube_id(url: str) -> Optional[str]:
    match = YOUTUBE_URL_REGEX.search(url.strip())
    return match.group(5) if match else None


def get_video_info(video_path: str) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()

    return {
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration": round(duration, 2),
    }


def extract_audio(video_path: str, output_dir: Optional[Path] = None) -> str:
    """
    Extracts a lightweight mono audio payload using FFmpeg.
    Reduces upload payload to speech services by ~90% (e.g. 150MB video -> ~4MB audio).
    """
    video_path_obj = Path(video_path)
    target_dir = output_dir or video_path_obj.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    audio_path = str(target_dir / f"{video_path_obj.stem}_audio.ogg")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "libopus",
        "-b:a", "48k",
        "-ac", "1",
        audio_path,
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        logger.info("Extracted audio track: %s", audio_path)
        return audio_path
    except subprocess.CalledProcessError as e:
        logger.warning(
            "Opus audio extraction failed (%s), falling back to AAC extraction",
            e.stderr.decode("utf-8", errors="replace")[-300:] if e.stderr else str(e),
        )

        fallback_path = str(target_dir / f"{video_path_obj.stem}_audio.m4a")
        fallback_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-c:a", "aac",
            "-b:a", "64k",
            "-ac", "1",
            fallback_path,
        ]
        subprocess.run(fallback_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return fallback_path


def download_youtube(url: str, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Downloads YouTube video via yt-dlp, constrained to 1080p max to balance quality and speed.
    """
    video_id = extract_youtube_id(url)
    if not video_id:
        raise ValueError(f"Invalid YouTube URL: {url}")

    save_dir = output_dir or (STORAGE_DIR / f"yt_{video_id}")
    save_dir.mkdir(parents=True, exist_ok=True)

    output_template = str(save_dir / "original.%(ext)s")

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--format", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--output", output_template,
        url,
    ]

    logger.info("Downloading YouTube video: %s -> %s", url, save_dir)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("yt-dlp error: %s", result.stderr[-1000:])
        raise RuntimeError(f"Failed to download YouTube video: {result.stderr[-300:]}")

    original_file = save_dir / "original.mp4"
    if not original_file.exists():
        candidates = list(save_dir.glob("original.*"))
        candidates = [c for c in candidates if not c.name.endswith(".json")]
        if candidates:
            original_file = candidates[0]
        else:
            raise FileNotFoundError(f"yt-dlp completed but output file not found in {save_dir}")

    # Read metadata if info.json was generated
    info_files = list(save_dir.glob("*.info.json"))
    metadata: Dict[str, Any] = {}
    if info_files:
        try:
            with open(info_files[0], "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning("Failed to parse yt-dlp metadata JSON: %s", e)

    video_info = get_video_info(str(original_file))

    return {
        "video_id": video_id,
        "video_path": str(original_file),
        "title": metadata.get("title", f"YouTube Video {video_id}"),
        "thumbnail_url": metadata.get("thumbnail"),
        "channel": metadata.get("uploader") or metadata.get("channel"),
        "duration": video_info["duration"],
        "width": video_info["width"],
        "height": video_info["height"],
        "fps": video_info["fps"],
    }
