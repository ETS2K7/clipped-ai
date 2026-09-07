import os
import re
import json
import logging
import requests
from typing import List, Dict, Any, Optional

from config import (
    get_logger,
    MIN_CLIP_DURATION,
    MAX_CLIP_DURATION,
    VIRALITY_CACHE_DIR,
)

logger = get_logger(__name__)

HOOK_KEYWORDS = {
    "curiosity": [
        "secret", "why", "how", "nobody knows", "hidden", "reason", "truth",
        "actually", "realized", "discovered", "shocking", "insane", "crazy",
    ],
    "contrarian": [
        "stop", "wrong", "lie", "never", "myth", "worst", "mistake", "don't",
        "bad advice", "scam", "waste of time", "failed", "quit",
    ],
    "insight": [
        "formula", "framework", "step", "rule", "lesson", "strategy", "hack",
        "system", "principle", "key", "method", "learned",
    ],
}


def _snap_to_word_boundary(
    raw_start: float,
    raw_end: float,
    words: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Snaps raw LLM timestamps to the nearest actual spoken word boundaries.
    Applies a natural breath buffer: -0.2s at start, +0.3s at end.
    """
    if not words:
        return {"start": raw_start, "end": raw_end}

    start_ms = raw_start * 1000.0
    end_ms = raw_end * 1000.0

    closest_start_word = min(words, key=lambda w: abs(w["start"] - start_ms))
    closest_end_word = min(words, key=lambda w: abs(w["end"] - end_ms))

    snapped_start = max(0.0, (closest_start_word["start"] / 1000.0) - 0.2)
    snapped_end = (closest_end_word["end"] / 1000.0) + 0.3

    return {
        "start": round(snapped_start, 2),
        "end": round(snapped_end, 2),
    }


def _group_into_sentences(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sentences = []
    current_words = []
    sentence_start = None

    for w in words:
        if sentence_start is None:
            sentence_start = w["start"] / 1000.0
        current_words.append(w["text"])

        ends_sentence = any(w["text"].endswith(p) for p in [".", "?", "!"])
        if ends_sentence:
            sentence_end = w["end"] / 1000.0
            sentences.append({
                "text": " ".join(current_words),
                "start": sentence_start,
                "end": sentence_end,
                "duration": sentence_end - sentence_start,
            })
            current_words = []
            sentence_start = None

    if current_words and sentence_start is not None:
        sentence_end = words[-1]["end"] / 1000.0
        sentences.append({
            "text": " ".join(current_words),
            "start": sentence_start,
            "end": sentence_end,
            "duration": sentence_end - sentence_start,
        })

    return sentences


def _heuristic_clip_selection(
    words: List[Dict[str, Any]],
    target_count: int = 3,
    user_focus: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    High-accuracy heuristic selector used if LLM providers are unavailable or quota-limited.
    Evaluates hook momentum, question/exclamation density, speech pace, and user focus keywords.
    """
    sentences = _group_into_sentences(words)
    if not sentences:
        total_duration = (words[-1]["end"] - words[0]["start"]) / 1000.0 if words else 30.0
        return [{
            "title": "Top Moment from Video",
            "start_time": 0.0,
            "end_time": min(total_duration, 35.0),
            "virality_score": 85,
            "hook_type": "curiosity_gap",
            "hook_rationale": "High-retention segment capturing the primary narrative flow.",
        }]

    focus_terms = (
        [t for t in re.split(r"\W+", user_focus.lower()) if len(t) > 2]
        if user_focus
        else []
    )

    scored_candidates = []
    min_dur = MIN_CLIP_DURATION
    max_dur = MAX_CLIP_DURATION

    # Scan multi-sentence windows within duration bounds
    for i in range(len(sentences)):
        curr_text = []
        start_t = sentences[i]["start"]
        for j in range(i, len(sentences)):
            curr_text.append(sentences[j]["text"])
            end_t = sentences[j]["end"]
            dur = end_t - start_t

            if min_dur <= dur <= max_dur:
                combined_text = " ".join(curr_text).lower()
                opening_sentence = sentences[i]["text"].lower()

                score = 60
                hook_type = "insight"

                # Boost if matches user focus topic/keyword
                if focus_terms:
                    matched_focus = [t for t in focus_terms if t in combined_text]
                    if matched_focus:
                        score += 25 * len(matched_focus)
                        if any(t in opening_sentence for t in matched_focus):
                            score += 15

                # Check opening hook strength
                for kw in HOOK_KEYWORDS["curiosity"]:
                    if kw in opening_sentence:
                        score += 8
                        hook_type = "curiosity_gap"
                for kw in HOOK_KEYWORDS["contrarian"]:
                    if kw in opening_sentence:
                        score += 10
                        hook_type = "contrarian"

                if "?" in opening_sentence:
                    score += 6
                if "!" in opening_sentence:
                    score += 4

                # Check speech cadence (words per minute)
                clip_words_count = sum(len(sentences[k]["text"].split()) for k in range(i, j + 1))
                wpm = (clip_words_count / dur) * 60.0
                if 130 <= wpm <= 180:
                    score += 6

                score = min(score, 98)

                scored_candidates.append({
                    "start_time": start_t,
                    "end_time": end_t,
                    "text": " ".join(curr_text),
                    "virality_score": score,
                    "hook_type": hook_type,
                    "title": sentences[i]["text"].strip()[:65],
                })

    scored_candidates.sort(key=lambda x: x["virality_score"], reverse=True)

    # Pick non-overlapping top clips
    selected: List[Dict[str, Any]] = []
    for cand in scored_candidates:
        if len(selected) >= target_count:
            break
        overlap = any(
            not (cand["end_time"] <= s["start_time"] or cand["start_time"] >= s["end_time"])
            for s in selected
        )
        if not overlap:
            selected.append({
                "title": cand["title"],
                "start_time": cand["start_time"],
                "end_time": cand["end_time"],
                "virality_score": cand["virality_score"],
                "hook_type": cand["hook_type"],
                "hook_rationale": f"High pacing and strong opening hook ({cand['hook_type'].replace('_', ' ')}).",
            })

    # If no window matched, return a single fallback clip
    if not selected:
        dur = min(sentences[-1]["end"], 30.0)
        selected.append({
            "title": sentences[0]["text"][:60],
            "start_time": 0.0,
            "end_time": dur,
            "virality_score": 82,
            "hook_type": "curiosity_gap",
            "hook_rationale": "High-retention opening segment.",
        })

    return selected


def _call_gemini_llm(transcript_prompt: str) -> Optional[List[Dict[str, Any]]]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        system_prompt = (
            "You are an elite short-form video editor specializing in viral retention on TikTok, "
            "YouTube Shorts, and Instagram Reels. Select 3 high-impact, standalone clips between "
            "20 and 45 seconds each. Each clip MUST open with a strong hook and conclude on a satisfying thought."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=transcript_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "clips": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "title": {"type": "STRING"},
                                    "start_time": {"type": "NUMBER"},
                                    "end_time": {"type": "NUMBER"},
                                    "virality_score": {"type": "INTEGER"},
                                    "hook_type": {"type": "STRING"},
                                    "hook_rationale": {"type": "STRING"},
                                },
                                "required": ["title", "start_time", "end_time", "virality_score", "hook_type", "hook_rationale"],
                            },
                        }
                    },
                    "required": ["clips"],
                },
                temperature=0.6,
            ),
        )
        data = json.loads(response.text)
        return data.get("clips", [])
    except Exception as e:
        logger.warning("Gemini LLM call failed: %s", e)
        return None


def _call_groq_llm(transcript_prompt: str) -> Optional[List[Dict[str, Any]]]:
    api_key = os.getenv("GROQ_KEY")
    if not api_key:
        return None

    models = [
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    system_instruction = (
        "You are an elite short-form video editor specializing in viral retention on TikTok, "
        "YouTube Shorts, and Instagram Reels. Select 3 high-impact, standalone clips between "
        "20 and 45 seconds each. Each clip MUST open with a strong hook and conclude on a satisfying thought. "
        "Always respond with valid JSON containing a 'clips' array with keys: "
        "title, start_time, end_time, virality_score, hook_type, hook_rationale."
    )

    for model in models:
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": transcript_prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.4,
            }
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=25,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                clips = parsed.get("clips", [])
                if clips and isinstance(clips, list):
                    logger.info("Successfully selected viral clips using Groq (%s)", model)
                    return clips
            else:
                logger.warning("Groq model %s returned status %d", model, resp.status_code)
        except Exception as e:
            logger.warning("Groq call failed for %s: %s", model, e)

    return None


def _call_openrouter_llm(transcript_prompt: str) -> Optional[List[Dict[str, Any]]]:
    api_key = os.getenv("OPENROUTER_KEY")
    if not api_key:
        return None

    models = [
        "google/gemini-2.5-flash",
        "meta-llama/llama-3.3-70b-instruct",
        "anthropic/claude-3.5-haiku",
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    system_instruction = (
        "You are an elite short-form video editor specializing in viral retention on TikTok, "
        "YouTube Shorts, and Instagram Reels. Select 3 high-impact, standalone clips between "
        "20 and 45 seconds each. Each clip MUST open with a strong hook and conclude on a satisfying thought. "
        "Always respond with valid JSON containing a 'clips' array with keys: "
        "title, start_time, end_time, virality_score, hook_type, hook_rationale."
    )

    for model in models:
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": transcript_prompt},
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 1500,
                "temperature": 0.4,
            }
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                clips = parsed.get("clips", [])
                if clips and isinstance(clips, list):
                    logger.info("Successfully selected viral clips using OpenRouter (%s)", model)
                    return clips
            else:
                logger.warning("OpenRouter model %s returned status %d", model, resp.status_code)
        except Exception as e:
            logger.warning("OpenRouter call failed for %s: %s", model, e)

    return None


def select_viral_clips(
    words: List[Dict[str, Any]],
    user_focus: Optional[str] = None,
    source_fingerprint: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Selects top viral moments from transcript, snaps them to exact word boundaries,
    and returns rich creator metadata for each clip.
    Caches selection to avoid unnecessary recomputation for identical inputs.
    """
    if not words:
        return []

    # Check disk cache
    if source_fingerprint:
        clean_focus = re.sub(r"\W+", "_", (user_focus or "all").strip().lower())
        cache_file = VIRALITY_CACHE_DIR / f"{source_fingerprint}_{clean_focus}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    if cached:
                        logger.info("Loaded viral clip selection from cache: %s", cache_file.name)
                        return cached
            except Exception as e:
                logger.warning("Failed to read virality cache: %s", e)

    # Build timestamped transcript string
    sentences = _group_into_sentences(words)
    formatted_lines = [
        f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text']}"
        for s in sentences
    ]
    transcript_text = "\n".join(formatted_lines)

    prompt = (
        f"TRANSCRIPT WITH TIMESTAMPS:\n{transcript_text}\n\n"
        "Identify 3 viral clip segments (20-45s duration each)."
    )
    if user_focus:
        prompt += f"\nFocus specifically on moments related to: '{user_focus}'."

    # Try LLMs in cascade: Gemini -> Groq -> OpenRouter -> Heuristic
    raw_clips = _call_gemini_llm(prompt)
    if not raw_clips:
        raw_clips = _call_groq_llm(prompt)
    if not raw_clips:
        raw_clips = _call_openrouter_llm(prompt)
    if not raw_clips:
        logger.info("Using heuristic viral selection engine...")
        raw_clips = _heuristic_clip_selection(words, user_focus=user_focus)

    # Snap boundaries and validate
    validated_clips: List[Dict[str, Any]] = []
    for raw in raw_clips:
        start_s = float(raw.get("start_time", 0.0))
        end_s = float(raw.get("end_time", 30.0))

        snapped = _snap_to_word_boundary(start_s, end_s, words)
        dur = snapped["end"] - snapped["start"]

        # Ensure valid length
        if dur < 10.0:
            snapped["end"] = snapped["start"] + 20.0
            dur = 20.0

        raw_score = int(raw.get("virality_score", 88))
        if 1 <= raw_score <= 10:
            raw_score *= 10
        raw_score = max(1, min(100, raw_score))

        validated_clips.append({
            "title": raw.get("title", "Viral Moment"),
            "start_time": snapped["start"],
            "end_time": snapped["end"],
            "duration": round(dur, 2),
            "virality_score": raw_score,
            "hook_type": raw.get("hook_type", "curiosity_gap"),
            "hook_rationale": raw.get("hook_rationale", "High emotional momentum and strong viewer hook."),
        })

    # Sort by virality score descending
    validated_clips.sort(key=lambda x: x["virality_score"], reverse=True)
    selected = validated_clips[:3]

    # Save to disk cache
    if source_fingerprint and selected:
        clean_focus = re.sub(r"\W+", "_", (user_focus or "all").strip().lower())
        cache_file = VIRALITY_CACHE_DIR / f"{source_fingerprint}_{clean_focus}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(selected, f, indent=2)
        except OSError:
            pass

    return selected
