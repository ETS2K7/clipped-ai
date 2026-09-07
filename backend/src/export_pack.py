import os
import re
from pathlib import Path
from typing import List, Dict, Any

from config import get_logger

logger = get_logger(__name__)


def ms_to_srt_time(ms: float) -> str:
    hours = int(ms // 3600000)
    minutes = int((ms % 3600000) // 60000)
    seconds = int((ms % 60000) // 1000)
    millis = int(ms % 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def ms_to_vtt_time(ms: float) -> str:
    hours = int(ms // 3600000)
    minutes = int((ms % 3600000) // 60000)
    seconds = int((ms % 60000) // 1000)
    millis = int(ms % 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def generate_srt_subtitles(
    words: List[Dict[str, Any]],
    clip_start_s: float,
    clip_end_s: float,
    output_srt_path: str,
) -> str:
    """Generates standard .srt format subtitle file."""
    clip_start_ms = clip_start_s * 1000.0
    clip_end_ms = clip_end_s * 1000.0

    clip_words = [
        w for w in words
        if w.get("start", 0) >= clip_start_ms and w.get("end", 0) <= clip_end_ms
    ]

    entries = []
    current_chunk = []
    max_chunk = 4

    for i, w in enumerate(clip_words):
        current_chunk.append(w)
        if len(current_chunk) >= max_chunk or i == len(clip_words) - 1:
            start_ms = max(0.0, current_chunk[0]["start"] - clip_start_ms)
            end_ms = min(clip_end_ms - clip_start_ms, current_chunk[-1]["end"] - clip_start_ms)
            text = " ".join(cw["text"] for cw in current_chunk)

            entries.append((start_ms, end_ms, text))
            current_chunk = []

    lines = []
    for idx, (s, e, txt) in enumerate(entries, start=1):
        lines.append(str(idx))
        lines.append(f"{ms_to_srt_time(s)} --> {ms_to_srt_time(e)}")
        lines.append(txt)
        lines.append("")

    Path(output_srt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_srt_path


def build_creator_seo_pack(
    clip_title: str,
    hook_type: str,
    virality_score: int,
    transcript_snippet: str,
) -> Dict[str, Any]:
    """
    Generates tailored, high-converting social media metadata and captions
    for YouTube Shorts, TikTok, and Instagram Reels.
    """
    clean_title = re.sub(r'[^\w\s-]', '', clip_title).strip()
    words = [w.lower() for w in clean_title.split() if len(w) > 3]
    topic_tag = f"#{words[0]}" if words else "#creators"

    # YouTube Shorts Package
    yt_titles = [
        clean_title[:70],
        f"The Truth About {words[0].capitalize() if words else 'This'}"[:70],
        f"Why Nobody Talks About This ({clean_title[:45]})"[:70],
    ]
    yt_description = (
        f"{clip_title}\n\n"
        f"Key takeaway from the conversation: \"{transcript_snippet[:140]}...\"\n\n"
        "Subscribe for daily insights on creator growth and modern tools.\n\n"
        f"#shorts #viral #creator {topic_tag} #trending"
    )

    # TikTok Package
    tiktok_caption = (
        f"{clean_title} 🤯 wait till the end! {topic_tag} #fyp #foryou #viral #growthmindset"
    )

    # Instagram Reels Package
    reels_caption = (
        f"{clean_title}\n\n"
        "Do you agree with this take? Drop your thoughts below 👇\n\n"
        f".\n.\n.\n#reels #viralvideos #contentcreator {topic_tag} #explorepage #mindset"
    )

    return {
        "youtube_shorts": {
            "title_options": yt_titles,
            "description": yt_description,
            "tags": ["shorts", "viral", "podcast", "clips", words[0] if words else "trending"],
        },
        "tiktok": {
            "caption": tiktok_caption,
            "recommended_sound": "Original Audio or Trending Ambient Beat",
            "hashtags": ["fyp", "foryou", "viral", "trend", words[0] if words else "content"],
        },
        "instagram_reels": {
            "caption": reels_caption,
            "call_to_action": "Save this reel for later 📌",
            "hashtags": ["reels", "viralvideos", "creator", "explorepage"],
        },
    }
