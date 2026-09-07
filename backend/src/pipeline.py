import os
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

from config import get_logger, STORAGE_DIR
from src.downloader import (
    get_video_info,
    download_youtube,
    extract_youtube_id,
    get_source_fingerprint,
)
from src.transcriber import transcribe_video
from src.virality import select_viral_clips
from src.tracker import (
    FastASDManager,
    compute_smoothed_camera_positions,
    render_portrait_crop,
)
from src.subtitles import generate_karaoke_ass, burn_subtitles_to_video
from src.thumbnails import generate_hook_thumbnail
from src.export_pack import build_creator_seo_pack, generate_srt_subtitles

logger = get_logger(__name__)


def extract_clip_segment(
    source_video: str,
    start_s: float,
    end_s: float,
    output_path: str,
) -> str:
    """Extracts a high-quality video subclip using FFmpeg with re-encoding. Reuses existing clip if present."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        logger.info("Reusing cached raw subclip: %s", output_path)
        return output_path

    dur = max(1.0, end_s - start_s)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-i", source_video,
        "-t", str(dur),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        output_path,
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return output_path


def run_pipeline(
    video_source: str,
    task_id: str,
    caption_style: str = "hormozi",
    user_focus: Optional[str] = None,
    burn_subtitles: bool = True,
    on_progress: Optional[Callable[[str, str, int], None]] = None,
) -> Dict[str, Any]:
    """
    Executes the complete video-to-viral-clips pipeline.
    Streams real-time progress percentages and status messages via callback.
    """
    task_dir = STORAGE_DIR / f"task_{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)

    def report(stage: str, message: str, percent: int):
        logger.info("[%s] (%d%%) %s", stage, percent, message)
        if on_progress:
            on_progress(stage, message, percent)

    start_time = time.time()
    report("ingesting", "Ingesting source video...", 5)

    source_fingerprint = get_source_fingerprint(video_source)

    # 1. Resolve source video (YouTube download or local upload)
    if extract_youtube_id(video_source):
        yt_meta = download_youtube(video_source, output_dir=task_dir / "source")
        video_path = yt_meta["video_path"]
        source_title = yt_meta["title"]
    else:
        video_path = video_source
        source_title = Path(video_source).stem

    video_info = get_video_info(video_path)
    report("ingesting", f"Ingested '{source_title}' ({video_info['duration']}s)", 15)

    # 2. Transcribe with speaker diarization
    report("transcribing", "Transcribing speech and extracting word timestamps...", 25)
    transcript_result = transcribe_video(
        video_path,
        use_cache=True,
        source_fingerprint=source_fingerprint,
        on_progress=lambda stage, msg: report("transcribing", msg, 35),
    )
    words = transcript_result.get("words", [])
    if not words:
        raise RuntimeError("Transcription produced no words. Audio may be silent or corrupted.")

    report("analyzing", "Identifying high-retention hooks and viral moments...", 45)
    clips = select_viral_clips(
        words,
        user_focus=user_focus,
        source_fingerprint=source_fingerprint,
    )
    if not clips:
        raise RuntimeError("No suitable viral moments were identified.")

    report("analyzing", f"Selected {len(clips)} viral clips with high retention scores.", 55)

    # 3. Process each clip through Fast-ASD tracking, reframing, and styling
    clips_dir = task_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    processed_clips: List[Dict[str, Any]] = []

    progress_step = 35.0 / len(clips)
    current_progress = 55.0

    asd_manager = FastASDManager.get_instance()

    for idx, clip in enumerate(clips, start=1):
        clip_prefix = f"clip_{idx}"
        clip_start = clip["start_time"]
        clip_end = clip["end_time"]
        clip_dur = clip["duration"]

        report(
            "processing_clip",
            f"Processing clip {idx}/{len(clips)}: '{clip['title']}'",
            int(current_progress),
        )

        # 3a. Extract 16:9 segment
        raw_clip_path = str(clips_dir / f"{clip_prefix}_raw.mp4")
        extract_clip_segment(video_path, clip_start, clip_end, raw_clip_path)

        # 3b. Fast-ASD active speaker tracking
        report(
            "tracking",
            f"Running Fast-ASD active speaker detection for clip {idx}...",
            int(current_progress + 5),
        )
        asd_cache_key = f"{source_fingerprint}_{int(round(clip_start * 1000))}_{int(round(clip_end * 1000))}"
        tracking_data = asd_manager.detect_active_speakers(raw_clip_path, cache_key=asd_cache_key)

        # 3c. Smooth camera coordinates
        clip_info = get_video_info(raw_clip_path)
        smoothed_cx = compute_smoothed_camera_positions(
            tracking_data,
            clip_info["frame_count"],
        )

        # 3d. 9:16 Portrait crop
        reframed_path = str(clips_dir / f"{clip_prefix}_portrait.mp4")
        render_portrait_crop(raw_clip_path, reframed_path, smoothed_cx)

        # 3e. Dynamic Subtitles
        ass_path = str(clips_dir / f"{clip_prefix}_subtitles.ass")
        srt_path = str(clips_dir / f"{clip_prefix}_subtitles.srt")
        generate_karaoke_ass(words, clip_start, clip_end, ass_path, preset_name=caption_style)
        generate_srt_subtitles(words, clip_start, clip_end, srt_path)

        final_video_path = reframed_path
        if burn_subtitles:
            burned_path = str(clips_dir / f"{clip_prefix}_final.mp4")
            burn_subtitles_to_video(reframed_path, ass_path, burned_path)
            final_video_path = burned_path

        # 3f. Hook Thumbnail
        thumb_path = str(clips_dir / f"{clip_prefix}_thumb.jpg")
        generate_hook_thumbnail(
            final_video_path,
            thumb_path,
            hook_text=clip["title"],
            virality_score=clip["virality_score"],
            timestamp_s=min(2.0, clip_dur / 2.0),
        )

        # 3g. Creator SEO Pack
        snippet_words = [
            w["text"] for w in words
            if w.get("start", 0) >= clip_start * 1000 and w.get("end", 0) <= clip_end * 1000
        ]
        seo_pack = build_creator_seo_pack(
            clip_title=clip["title"],
            hook_type=clip["hook_type"],
            virality_score=clip["virality_score"],
            transcript_snippet=" ".join(snippet_words[:40]),
        )

        processed_clips.append({
            "id": f"{task_id}_{idx}",
            "index": idx,
            "title": clip["title"],
            "start_time": clip_start,
            "end_time": clip_end,
            "duration": clip_dur,
            "virality_score": clip["virality_score"],
            "hook_type": clip["hook_type"],
            "hook_rationale": clip["hook_rationale"],
            "video_path": final_video_path,
            "thumbnail_path": thumb_path,
            "ass_path": ass_path,
            "srt_path": srt_path,
            "seo_pack": seo_pack,
        })

        current_progress += progress_step

    elapsed_time = round(time.time() - start_time, 1)
    report("completed", f"Finished {len(processed_clips)} clips in {elapsed_time}s", 100)

    return {
        "task_id": task_id,
        "source_title": source_title,
        "video_path": video_path,
        "total_duration": video_info["duration"],
        "elapsed_seconds": elapsed_time,
        "clips": processed_clips,
    }
