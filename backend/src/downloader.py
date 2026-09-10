import os
import sys
import re
import json
import shutil
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
from config import (
    get_logger,
    STORAGE_DIR,
    PROJECT_DIR,
    SOURCES_CACHE_DIR,
    AUDIO_CACHE_DIR,
)

logger = get_logger(__name__)

def _find_ytdlp() -> str:
    """Finds the most up-to-date yt-dlp binary (prioritizing virtual environment)."""
    env_bin = Path(sys.executable).parent / "yt-dlp"
    if env_bin.exists():
        return str(env_bin)
    venv_bin = PROJECT_DIR / "venv" / "bin" / "yt-dlp"
    if venv_bin.exists():
        return str(venv_bin)
    return "yt-dlp"


YOUTUBE_URL_REGEX = re.compile(
    r"^(https?://)?(www\.|m\.)?(youtube\.com/(watch\?v=|embed/|v/)|youtu\.be/)([\w-]{11})"
)


def extract_youtube_id(url: str) -> Optional[str]:
    match = YOUTUBE_URL_REGEX.search(url.strip())
    return match.group(5) if match else None


def get_source_fingerprint(video_source: str) -> str:
    """
    Computes a deterministic, collision-resistant identifier for any video source
    (YouTube URL or local video file) so downstream stages are cached and reused across runs.
    """
    video_source_clean = video_source.strip().strip("\"' ")
    yt_id = extract_youtube_id(video_source_clean)
    if yt_id:
        return f"yt_{yt_id}"

    video_path = Path(video_source_clean).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_source_clean}")

    stat = video_path.stat()
    hasher = hashlib.sha256()
    hasher.update(str(video_path).encode("utf-8"))
    hasher.update(str(stat.st_size).encode("utf-8"))
    hasher.update(str(stat.st_mtime_ns).encode("utf-8"))

    clean_stem = re.sub(r"[^\w-]", "_", video_path.stem)[:30]
    return f"local_{clean_stem}_{hasher.hexdigest()[:10]}"


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


def extract_audio(
    video_path: str,
    output_dir: Optional[Path] = None,
    cache_key: Optional[str] = None,
) -> str:
    """
    Extracts a lightweight mono audio payload using FFmpeg.
    Reuses cached audio if already extracted to avoid duplicate processing.
    """
    video_path_obj = Path(video_path)
    if cache_key:
        target_dir = output_dir or AUDIO_CACHE_DIR
        audio_name = f"{cache_key}_audio.ogg"
        fallback_name = f"{cache_key}_audio.m4a"
    else:
        target_dir = output_dir or video_path_obj.parent
        audio_name = f"{video_path_obj.stem}_audio.ogg"
        fallback_name = f"{video_path_obj.stem}_audio.m4a"

    target_dir.mkdir(parents=True, exist_ok=True)
    audio_path = str(target_dir / audio_name)
    fallback_path = str(target_dir / fallback_name)

    # Return cached audio if present and non-empty
    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
        logger.info("Reusing cached audio track: %s", audio_path)
        return audio_path
    if os.path.exists(fallback_path) and os.path.getsize(fallback_path) > 1000:
        logger.info("Reusing cached audio track: %s", fallback_path)
        return fallback_path

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
    Downloads YouTube video via yt-dlp, constrained to 1080p max.
    Caches downloads in SOURCES_CACHE_DIR to avoid re-downloading the same video.
    """
    video_id = extract_youtube_id(url)
    if not video_id:
        raise ValueError(f"Invalid YouTube URL: {url}")

    canonical_dir = SOURCES_CACHE_DIR / f"yt_{video_id}"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_file = canonical_dir / "original.mp4"

    # 1. Check if already present in canonical cache
    if canonical_file.exists() and canonical_file.stat().st_size > 50000:
        logger.info("Reusing cached YouTube download: %s", canonical_file)
    else:
        # Clean up any partial files from previously aborted downloads
        for partial in canonical_dir.glob("original.*"):
            if not partial.name.endswith(".info.json"):
                try:
                    partial.unlink()
                except OSError:
                    pass

        output_template = str(canonical_dir / "original.%(ext)s")
        cmd = [
            _find_ytdlp(),
            "--no-playlist",
            "--format", "bestvideo[height<=1080][format_note!*=?Premium]+bestaudio/best[height<=1080]/best",
            "--merge-output-format", "mp4",
            "--retries", "10",
            "--fragment-retries", "10",
            "--extractor-args", "youtube:player_client=default,web",
            "--write-info-json",
            "--output", output_template,
            url,
        ]

        logger.info("Downloading YouTube video: %s -> %s", url, canonical_dir)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("yt-dlp error: %s", result.stderr[-1000:])
            raise RuntimeError(f"Failed to download YouTube video: {result.stderr[-300:]}")

        if not canonical_file.exists():
            candidates = list(canonical_dir.glob("original.*"))
            candidates = [c for c in candidates if not c.name.endswith(".json")]
            if candidates:
                canonical_file = candidates[0]
            else:
                raise FileNotFoundError(f"yt-dlp completed but output file not found in {canonical_dir}")

    # Read metadata if info.json was generated
    info_files = list(canonical_dir.glob("*.info.json"))
    metadata: Dict[str, Any] = {}
    if info_files:
        try:
            with open(info_files[0], "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning("Failed to parse yt-dlp metadata JSON: %s", e)

    target_file = canonical_file
    if output_dir and Path(output_dir).resolve() != canonical_dir.resolve():
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        dest_video = target_dir / canonical_file.name
        if not dest_video.exists() or dest_video.stat().st_size != canonical_file.stat().st_size:
            shutil.copy2(canonical_file, dest_video)
            if info_files:
                shutil.copy2(info_files[0], target_dir / info_files[0].name)
        target_file = dest_video

    video_info = get_video_info(str(target_file))

    return {
        "video_id": video_id,
        "video_path": str(target_file),
        "title": metadata.get("title", f"YouTube Video {video_id}"),
        "thumbnail_url": metadata.get("thumbnail"),
        "channel": metadata.get("uploader") or metadata.get("channel"),
        "duration": video_info["duration"],
        "width": video_info["width"],
        "height": video_info["height"],
        "fps": video_info["fps"],
    }


def download_video(url: str) -> str:
    """Compatibility wrapper matching Documents/clippedai download_video signature."""
    meta = download_youtube(url)
    return meta["video_path"]
