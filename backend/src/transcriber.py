import os
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from config import get_logger, get_assemblyai_key, STORAGE_DIR
from src.downloader import extract_audio

logger = get_logger(__name__)

ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
ASSEMBLYAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"
TRANSCRIPTS_CACHE_DIR = STORAGE_DIR / "transcripts"
TRANSCRIPTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _compute_file_sha256(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def transcribe_video(
    video_path: str,
    use_cache: bool = True,
    source_fingerprint: Optional[str] = None,
    on_progress: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Transcribes video speech using AssemblyAI Universal-2 with speaker diarization.
    Returns word-level timestamps and speaker IDs. Caches results deterministically.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 1. Fast check: source_fingerprint cache
    if use_cache and source_fingerprint:
        fingerprint_cache = TRANSCRIPTS_CACHE_DIR / f"{source_fingerprint}.json"
        if fingerprint_cache.exists():
            logger.info("Loaded transcript from fingerprint cache: %s", fingerprint_cache.name)
            with open(fingerprint_cache, "r", encoding="utf-8") as f:
                data = json.load(f)
                if on_progress:
                    on_progress("transcription_complete", "Transcription loaded from cache.")
                return data

    # Extract lightweight audio to minimize upload bandwidth
    if on_progress:
        on_progress("extracting_audio", "Extracting speech audio track...")

    audio_path = extract_audio(video_path, cache_key=source_fingerprint)
    file_hash = _compute_file_sha256(audio_path)
    cache_file = TRANSCRIPTS_CACHE_DIR / f"{file_hash}.json"

    if use_cache and cache_file.exists():
        logger.info("Loaded transcript from cache: %s", cache_file.name)
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if source_fingerprint:
                # also create the fingerprint symlink/file for even faster future hits
                try:
                    with open(TRANSCRIPTS_CACHE_DIR / f"{source_fingerprint}.json", "w", encoding="utf-8") as ff:
                        json.dump(data, ff)
                except OSError:
                    pass
            if on_progress:
                on_progress("transcription_complete", "Transcription loaded from cache.")
            return data

    api_key = get_assemblyai_key()
    headers = {"authorization": api_key}

    if on_progress:
        on_progress("uploading_audio", "Uploading audio to speech engine...")

    logger.info("Uploading audio payload (%s) to AssemblyAI...", audio_path)
    with open(audio_path, "rb") as f:
        upload_res = requests.post(
            ASSEMBLYAI_UPLOAD_URL,
            headers=headers,
            data=f,
            timeout=300,
        )

    if upload_res.status_code != 200:
        raise RuntimeError(f"AssemblyAI upload failed: {upload_res.status_code} - {upload_res.text}")

    upload_url = upload_res.json()["upload_url"]

    # Submit transcription request
    payload = {
        "audio_url": upload_url,
        "speech_models": ["universal-2"],
        "speaker_labels": True,
        "punctuate": True,
        "format_text": True,
    }

    if on_progress:
        on_progress("transcribing", "Transcribing speech with speaker diarization...")

    logger.info("Initiating transcription job...")
    start_res = requests.post(
        ASSEMBLYAI_TRANSCRIPT_URL,
        headers=headers,
        json=payload,
        timeout=30,
    )

    if start_res.status_code != 200:
        raise RuntimeError(f"AssemblyAI start failed: {start_res.status_code} - {start_res.text}")

    transcript_id = start_res.json()["id"]
    poll_url = f"{ASSEMBLYAI_TRANSCRIPT_URL}/{transcript_id}"

    # Poll for completion with exponential backoff
    poll_interval = 1.5
    max_poll_interval = 6.0
    max_attempts = 120  # ~8-10 minutes maximum

    for attempt in range(max_attempts):
        time.sleep(poll_interval)
        poll_res = requests.get(poll_url, headers=headers, timeout=30)
        
        if poll_res.status_code != 200:
            logger.warning("Poll returned status %d, retrying...", poll_res.status_code)
            continue

        res_data = poll_res.json()
        status = res_data.get("status")

        if status == "completed":
            raw_words = res_data.get("words", [])
            utterances = res_data.get("utterances", [])
            full_text = res_data.get("text", "")

            # Format words uniformly: start and end in milliseconds
            words: List[Dict[str, Any]] = []
            for w in raw_words:
                words.append({
                    "text": w["text"],
                    "start": int(w["start"]),
                    "end": int(w["end"]),
                    "confidence": round(float(w.get("confidence", 1.0)), 2),
                    "speaker": w.get("speaker", "A"),
                })

            result = {
                "transcript_id": transcript_id,
                "text": full_text,
                "words": words,
                "utterances": utterances,
                "word_count": len(words),
            }

            # Cache the result
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            if source_fingerprint:
                try:
                    with open(TRANSCRIPTS_CACHE_DIR / f"{source_fingerprint}.json", "w", encoding="utf-8") as ff:
                        json.dump(result, ff, indent=2)
                except OSError:
                    pass

            logger.info(
                "Transcription completed: %d words, cached to %s",
                len(words),
                cache_file.name,
            )

            if on_progress:
                on_progress("transcription_complete", f"Transcribed {len(words)} words successfully.")

            return result

        if status == "error":
            error_msg = res_data.get("error", "Unknown AssemblyAI error")
            raise RuntimeError(f"AssemblyAI transcription failed: {error_msg}")

        # Gradual backoff
        poll_interval = min(poll_interval * 1.3, max_poll_interval)

    raise TimeoutError(f"Transcription timed out after {max_attempts} polling attempts")
