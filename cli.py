#!/usr/bin/env python3
import os
import sys
import re
import time
import shutil
import uuid
import json
import subprocess
from pathlib import Path
from typing import Dict, Any

# Ensure project root and backend modules are in sys.path
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
TALKNET_DIR = BACKEND_DIR / "fast-asd" / "talknet"

# Auto-reexec with local virtualenv if available
venv_python = ROOT_DIR / "venv" / "bin" / "python3"
if venv_python.exists() and sys.executable != str(venv_python):
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

for path in (str(BACKEND_DIR), str(TALKNET_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.downloader import extract_youtube_id
from src.pipeline import run_pipeline


def sanitize_filename(name: str) -> str:
    """Sanitizes a string for use as a folder or file name."""
    if "/" in name or "\\" in name or "." in name:
        name = Path(name).stem
    # Strip common channel suffixes like '|', ' - ', or '–'
    first_part = re.split(r"\s*[|\-–—]\s*", name)[0].strip()
    target = first_part if len(first_part) >= 4 else name
    clean = re.sub(r"[\\/*?:\"<>|#%&{}\\<>*?/$!\'\":@+`|=()\[\].]", "", target).strip()
    clean = re.sub(r"\s+", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean[:60] or "video"


def get_video_title(video_source: str, is_youtube: bool) -> str:
    """Gets human-readable title for the video source."""
    if not is_youtube:
        return Path(video_source).stem
    yt_id = extract_youtube_id(video_source)
    if yt_id:
        cache_info = ROOT_DIR / "storage" / "sources" / f"yt_{yt_id}" / "original.info.json"
        if cache_info.exists():
            try:
                with open(cache_info, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("title"):
                        return data["title"]
            except Exception:
                pass
        try:
            from src.downloader import _find_ytdlp
            cmd = [_find_ytdlp(), "--no-playlist", "--print", "title", video_source]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            pass
        return f"yt_{yt_id}"
    return "video"



def write_seo_summary(assets_dir: Path, clips: list):
    """Writes a human-readable text file with social media copy and hashtags."""
    summary_path = assets_dir / "seo_summary.txt"
    lines = [
        "=" * 60,
        "  CLIPPEDAI — CREATOR DISTRIBUTION PACK",
        "=" * 60,
        "",
    ]

    for clip in clips:
        lines.append(f"--- CLIP {clip['index']}: {clip['title']} ---")
        lines.append(f"Virality Score: {clip['virality_score']}/100")
        lines.append(f"Hook Type:      {clip['hook_type'].replace('_', ' ').title()}")
        lines.append(f"Duration:       {clip['duration']}s")
        lines.append(f"Hook Rationale: {clip['hook_rationale']}")
        lines.append("")

        seo = clip.get("seo_pack", {})

        # YouTube Shorts
        yt = seo.get("youtube_shorts", {})
        lines.append("[ YouTube Shorts ]")
        lines.append("Title Options:")
        for t in yt.get("title_options", []):
            lines.append(f"  • {t}")
        lines.append("Description:")
        lines.append(yt.get("description", ""))
        lines.append(f"Tags: {', '.join(yt.get('tags', []))}")
        lines.append("")

        # TikTok
        tt = seo.get("tiktok", {})
        lines.append("[ TikTok ]")
        lines.append(f"Caption: {tt.get('caption', '')}")
        lines.append(f"Sound:   {tt.get('recommended_sound', '')}")
        lines.append(f"Tags:    {' '.join('#' + t.lstrip('#') for t in tt.get('hashtags', []))}")
        lines.append("")

        # Instagram Reels
        reels = seo.get("instagram_reels", {})
        lines.append("[ Instagram Reels ]")
        lines.append(f"Caption: {reels.get('caption', '')}")
        lines.append(f"CTA:     {reels.get('call_to_action', '')}")
        lines.append(f"Tags:    {' '.join('#' + t.lstrip('#') for t in reels.get('hashtags', []))}")
        lines.append("\n" + "=" * 60 + "\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("\n" + "=" * 60)
    print("      ClippedAI — Video Clipping & Channel Automation CLI")
    print("=" * 60 + "\n")

    # 1. Prompt for Video Source
    while True:
        raw_input = input("Enter YouTube URL or local video file path: ").strip()
        cleaned_source = raw_input.strip("\"' ")

        if not cleaned_source:
            print("❌ Input cannot be empty. Please try again.\n")
            continue

        if extract_youtube_id(cleaned_source):
            video_source = cleaned_source
            is_youtube = True
            break
        elif os.path.isfile(cleaned_source):
            video_source = os.path.abspath(cleaned_source)
            is_youtube = False
            break
        else:
            print(f"❌ File not found or invalid YouTube URL: {cleaned_source}")
            print("   Please provide a valid file path or YouTube link.\n")

    # 2. Subtitle Style
    caption_style = "hormozi"
    print("\nSubtitle Style: Hormozi (Bold alternating yellow/green neon highlight)")

    # 3. Prompt for Keyword / Focus (Optional)
    focus_input = input("Specific moment or keyword to prioritize (press Enter to skip): ").strip()
    user_focus = focus_input if focus_input else None

    # 4. Prepare Destination Folder inside clips/
    clips_base_dir = ROOT_DIR / "clips"
    clips_base_dir.mkdir(parents=True, exist_ok=True)

    task_id = f"yt_{extract_youtube_id(video_source)}" if is_youtube else sanitize_filename(Path(video_source).stem)
    video_title = get_video_title(video_source, is_youtube)
    folder_name = sanitize_filename(video_title)

    output_subfolder = clips_base_dir / folder_name
    is_overwrite = output_subfolder.exists() and any(output_subfolder.iterdir())
    output_subfolder.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 60)
    print(f"🚀 Starting Clipping Pipeline [Task: {task_id}]")
    print(f"📁 Output Folder: {output_subfolder.relative_to(ROOT_DIR)}" + (" (overwriting existing clips)" if is_overwrite else ""))
    print("🎨 Caption Style: Hormozi (Bold alternating yellow/green neon highlight)")
    if user_focus:
        print(f"🎯 Target Focus:  '{user_focus}'")
    print("-" * 60 + "\n")

    # Progress reporting in terminal
    def cli_progress(stage: str, message: str, percent: int):
        bar_len = 24
        filled = int(bar_len * (percent / 100.0))
        bar = "█" * filled + "░" * (bar_len - filled)
        sys.stdout.write(f"\r[{bar}] {percent:3d}% | {message[:60]:<60}")
        sys.stdout.flush()
        if percent >= 100 or stage == "completed":
            sys.stdout.write("\n")

    # 5. Run Pipeline
    try:
        pipeline_result = run_pipeline(
            video_source=video_source,
            task_id=task_id,
            caption_style=caption_style,
            user_focus=user_focus,
            burn_subtitles=True,
            on_progress=cli_progress,
        )
    except Exception as e:
        print(f"\n\n❌ Pipeline Error: {e}")
        sys.exit(1)

    # 6. Copy Deliverables to Destination Subfolder
    source_title = pipeline_result.get("source_title")
    if source_title:
        refined_folder_name = sanitize_filename(source_title)
        if refined_folder_name != folder_name and not any(output_subfolder.iterdir()):
            try:
                output_subfolder.rmdir()
            except Exception:
                pass
            output_subfolder = clips_base_dir / refined_folder_name
            output_subfolder.mkdir(parents=True, exist_ok=True)

    assets_subfolder = output_subfolder / "assets"
    assets_subfolder.mkdir(parents=True, exist_ok=True)

    generated_clips = pipeline_result.get("clips", [])
    final_clips_info = []

    for clip in generated_clips:
        idx = clip["index"]

        # Copy only the vertical video clip
        dest_video = output_subfolder / f"clip_{idx}.mp4"
        shutil.copy2(clip["video_path"], dest_video)

        # Copy thumbnail to assets/
        if clip.get("thumbnail_path") and os.path.exists(clip["thumbnail_path"]):
            dest_thumb = assets_subfolder / f"clip_{idx}_thumb.jpg"
            shutil.copy2(clip["thumbnail_path"], dest_thumb)

        final_clips_info.append({
            **clip,
            "dest_video": str(dest_video),
        })

    # Save social media copy and hashtags into assets/
    write_seo_summary(assets_subfolder, generated_clips)

    # 7. Print Completion Summary Table
    print("\n" + "=" * 60)
    print("                 PIPELINE COMPLETE! 🎉")
    print("=" * 60)
    print(f"Source:   {pipeline_result.get('source_title')}")
    print(f"Duration: {pipeline_result.get('total_duration')}s | Processed in: {pipeline_result.get('elapsed_seconds')}s")
    print(f"Output:   {output_subfolder.resolve()}\n")

    print(f"{'#':<3} {'Virality':<12} {'Hook Type':<20} {'Duration':<10} {'Title'}")
    print("-" * 75)
    for c in final_clips_info:
        score_badge = f"{c['virality_score']}/100 🔥"
        hook_name = c['hook_type'].replace('_', ' ').title()
        dur_str = f"{c['duration']}s"
        title_trunc = c['title'][:32] + ("..." if len(c['title']) > 32 else "")
        print(f"{c['index']:<3} {score_badge:<12} {hook_name:<20} {dur_str:<10} {title_trunc}")

    print("-" * 75)
    print(f"\nAll vertical clips saved to:")
    print(f"👉 {output_subfolder.resolve()}\n")
    print(f"Thumbnails and SEO pack saved to:")
    print(f"📁 {assets_subfolder.resolve()}\n")
    print(f"To view files in Finder, run: open '{output_subfolder.resolve()}'\n")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n\nOperation cancelled. Exiting.")
        sys.exit(0)
