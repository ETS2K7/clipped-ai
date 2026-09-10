import os
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List

from config import get_logger, STORAGE_DIR, FONTS_DIR
from src.downloader import (
    get_video_info,
    download_youtube,
    extract_youtube_id,
    get_source_fingerprint,
)
from src.transcriber import transcribe_video
from src.virality import select_viral_clips
from fast_asd_local import LocalFastASDTracker
from src.video_processing import (
    extract_segment,
    track_speaker_and_frame,
    merge_and_cleanup,
)
from src.subtitles import generate_subtitles
from src.thumbnails import generate_hook_thumbnail
from src.export_pack import build_creator_seo_pack, generate_srt_subtitles

logger = get_logger(__name__)


def run_pipeline(
    video_source: str,
    task_id: str,
    aspect_ratio: str = "9:16",
    clip_video: bool = True,
    caption_style: str = "hormozi",
    user_focus: Optional[str] = None,
    burn_subtitles: bool = True,
    on_progress: Optional[Callable[[str, str, int], None]] = None,
) -> Dict[str, Any]:
    """
    Executes the complete video-to-viral-clips pipeline.
    Streams real-time progress percentages and status messages via callback.

    Supports:
      - aspect_ratio: "9:16" (vertical reframing) or "original" (preserves widescreen/source format)
      - clip_video: True (extract top viral clips) or False (process complete video without clipping)
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
    vid_width = video_info.get("width", 1920)
    vid_height = video_info.get("height", 1080)
    total_dur = video_info["duration"]
    report("ingesting", f"Ingested '{source_title}' ({total_dur}s, {vid_width}x{vid_height})", 15)

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

    if not clip_video:
        report("analyzing", "Full video mode: preserving entire video without clipping...", 50)
        clips = [{
            "index": 1,
            "title": f"{source_title[:40]} (Full Video)",
            "start_time": 0.0,
            "end_time": float(total_dur),
            "duration": float(total_dur),
            "virality_score": 100,
            "hook_type": "full_video",
            "hook_rationale": "Full video presentation with synchronized kinetic subtitles.",
        }]
    else:
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

    progress_step = 35.0 / max(1, len(clips))
    current_progress = 55.0

    local_tracker = LocalFastASDTracker.get_instance() if aspect_ratio != "original" else None

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

        work_dir = str(clips_dir)

        # 3a. Extract segment (H.264/AAC re-encode for stable OpenCV/FFmpeg processing)
        report(
            "extracting",
            f"Extracting segment for clip {idx} ({clip_start}s - {clip_end}s)...",
            int(current_progress + 2),
        )
        ext_vid = extract_segment(video_path, clip, idx, work_dir=work_dir, use_gpu=False)

        # 3b. Framing: 9:16 portrait tracking or keep original aspect ratio
        if aspect_ratio == "original":
            trk_vid = ext_vid
            chunk_meta = [{"start_ms": 0, "end_ms": clip_dur * 1000, "flag": "ORIGINAL"}]
        else:
            report(
                "tracking",
                f"Running Fast-ASD active speaker tracking & adaptive framing for clip {idx}...",
                int(current_progress + 8),
            )
            trk_vid, chunk_meta = track_speaker_and_frame(
                clip_file=ext_vid,
                idx=idx,
                clip=clip,
                words=words,
                work_dir=work_dir,
                tracker=local_tracker,
                use_gpu=False,
            )

        # 3c. Layout-aware ASS Subtitles & SRT
        ass_path = str(clips_dir / f"{clip_prefix}_subtitles.ass")
        srt_path = str(clips_dir / f"{clip_prefix}_subtitles.srt")
        generate_srt_subtitles(words, clip_start, clip_end, srt_path)

        generated_ass = generate_subtitles(
            words=words,
            clip=clip,
            idx=idx,
            framing_meta=chunk_meta,
            aspect_ratio=aspect_ratio,
            video_width=vid_width,
            video_height=vid_height,
            work_dir=work_dir,
        )
        import shutil
        shutil.copy2(generated_ass, ass_path)

        sub_file = generated_ass if burn_subtitles else None

        # 3d. Hook Thumbnail (generate from clean video before subtitle burn-in)
        thumb_path = str(clips_dir / f"{clip_prefix}_thumb.jpg")
        generate_hook_thumbnail(
            trk_vid,
            thumb_path,
            hook_text=clip["title"],
            virality_score=clip["virality_score"],
            timestamp_s=min(2.0, clip_dur / 2.0),
            aspect_ratio=aspect_ratio,
        )

        # 3e. Merge and mux final video
        report(
            "rendering",
            f"Merging audio and burning subtitles for clip {idx}...",
            int(current_progress + 15),
        )
        fonts_dir_str = str(FONTS_DIR) if FONTS_DIR.exists() else ""
        merge_and_cleanup(
            tracked_vid=trk_vid,
            extract_vid=ext_vid,
            sub_file=sub_file,
            idx=idx,
            work_dir=work_dir,
            use_gpu=False,
            fonts_dir=fonts_dir_str,
        )
        final_video_path = str(clips_dir / f"clip_{idx}.mp4")

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
        "aspect_ratio": aspect_ratio,
        "clip_video": clip_video,
        "total_duration": video_info["duration"],
        "elapsed_seconds": elapsed_time,
        "clips": processed_clips,
    }
