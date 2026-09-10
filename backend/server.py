import os
import uuid
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from config import get_logger, STORAGE_DIR, BACKEND_DIR
from src.pipeline import run_pipeline

logger = get_logger(__name__)

app = FastAPI(
    title="ClippedAI Engine API",
    description="Autonomous AI Content Engine for Video Channels",
    version="1.0.0",
)

# CORS configuration for frontend / Codex integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount storage directory for static media serving (clips, thumbnails, subtitles)
app.mount("/media", StaticFiles(directory=str(STORAGE_DIR)), name="media")

DOCS_DIR = (BACKEND_DIR.parent / "docs").resolve()
if DOCS_DIR.exists():
    app.mount("/docs-visual", StaticFiles(directory=str(DOCS_DIR), html=True), name="docs-visual")

CLIPS_DIR = (BACKEND_DIR.parent / "clips").resolve()
if CLIPS_DIR.exists():
    app.mount("/clips", StaticFiles(directory=str(CLIPS_DIR)), name="clips")

# In-memory task state tracking
tasks_state: Dict[str, Dict[str, Any]] = {}
task_progress_queues: Dict[str, asyncio.Queue] = {}


class ProcessRequest(BaseModel):
    youtube_url: Optional[str] = None
    aspect_ratio: str = "9:16"
    clip_video: bool = True
    caption_style: str = "hormozi"
    user_focus: Optional[str] = None
    burn_subtitles: bool = True


def _get_or_create_queue(task_id: str) -> asyncio.Queue:
    if task_id not in task_progress_queues:
        task_progress_queues[task_id] = asyncio.Queue()
    return task_progress_queues[task_id]


def _background_pipeline_worker(
    video_source: str,
    task_id: str,
    aspect_ratio: str,
    clip_video: bool,
    caption_style: str,
    user_focus: Optional[str],
    burn_subtitles: bool,
):
    """Executes the pipeline in background thread and pushes events to the SSE queue."""
    queue = _get_or_create_queue(task_id)

    def on_progress(stage: str, message: str, percent: int):
        event_payload = {
            "stage": stage,
            "message": message,
            "percent": percent,
        }
        tasks_state[task_id]["stage"] = stage
        tasks_state[task_id]["message"] = message
        tasks_state[task_id]["percent"] = percent

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(queue.put(event_payload), loop)
            else:
                queue.put_nowait(event_payload)
        except Exception:
            pass

    try:
        tasks_state[task_id]["status"] = "processing"
        result = run_pipeline(
            video_source=video_source,
            task_id=task_id,
            aspect_ratio=aspect_ratio,
            clip_video=clip_video,
            caption_style=caption_style,
            user_focus=user_focus,
            burn_subtitles=burn_subtitles,
            on_progress=on_progress,
        )

        # Convert local file paths to static HTTP URLs
        server_clips = []
        for clip in result.get("clips", []):
            clip_copy = dict(clip)
            for key in ("video_path", "thumbnail_path", "ass_path", "srt_path"):
                local_path = clip_copy.get(key)
                if local_path:
                    rel_path = os.path.relpath(local_path, str(STORAGE_DIR))
                    clip_copy[key.replace("_path", "_url")] = f"/media/{rel_path}"
            server_clips.append(clip_copy)

        result["clips"] = server_clips
        tasks_state[task_id]["status"] = "completed"
        tasks_state[task_id]["result"] = result
        queue.put_nowait({"stage": "completed", "message": "All clips rendered successfully!", "percent": 100})
    except Exception as e:
        logger.exception("Pipeline failed for task %s", task_id)
        tasks_state[task_id]["status"] = "failed"
        tasks_state[task_id]["error"] = str(e)
        queue.put_nowait({"stage": "error", "message": f"Pipeline error: {str(e)}", "percent": 100})


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "clippedai-engine"}


@app.post("/api/process")
async def start_processing_job(
    background_tasks: BackgroundTasks,
    request: Optional[ProcessRequest] = None,
    file: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    aspect_ratio: str = Form("9:16"),
    clip_video: bool = Form(True),
    caption_style: str = Form("hormozi"),
    user_focus: Optional[str] = Form(None),
    burn_subtitles: bool = Form(True),
):
    """
    Starts an asynchronous video processing job from YouTube URL or multipart file upload.
    """
    # Parse input from either JSON or multipart form
    target_youtube = (request.youtube_url if request else None) or youtube_url
    target_aspect = (request.aspect_ratio if request else None) or aspect_ratio
    target_clip = (request.clip_video if request else None) if request else clip_video
    target_style = (request.caption_style if request else None) or caption_style
    target_focus = (request.user_focus if request else None) or user_focus
    target_burn = (request.burn_subtitles if request else None) if request else burn_subtitles

    task_id = str(uuid.uuid4())[:8]
    task_dir = STORAGE_DIR / f"task_{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)

    if file:
        file_path = str(task_dir / file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        video_source = file_path
    elif target_youtube:
        video_source = target_youtube.strip()
    else:
        raise HTTPException(
            status_code=400,
            detail="Must provide either 'youtube_url' or an uploaded video file.",
        )

    tasks_state[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "stage": "queued",
        "message": "Job queued in processing pipeline...",
        "percent": 0,
        "result": None,
    }

    _get_or_create_queue(task_id)

    # Spawn asynchronous background processing
    background_tasks.add_task(
        _background_pipeline_worker,
        video_source=video_source,
        task_id=task_id,
        aspect_ratio=target_aspect,
        clip_video=target_clip,
        caption_style=target_style,
        user_focus=target_focus,
        burn_subtitles=target_burn,
    )

    return {
        "task_id": task_id,
        "status": "queued",
        "progress_url": f"/api/progress/{task_id}",
        "task_url": f"/api/tasks/{task_id}",
    }


@app.get("/api/progress/{task_id}")
async def stream_progress(task_id: str):
    """
    Server-Sent Events (SSE) endpoint providing live status updates for the task.
    """
    if task_id not in tasks_state:
        raise HTTPException(status_code=404, detail="Task not found")

    queue = _get_or_create_queue(task_id)

    async def event_generator():
        # Yield current initial status
        initial_state = tasks_state[task_id]
        yield f"data: {json.dumps({'stage': initial_state['stage'], 'message': initial_state['message'], 'percent': initial_state['percent']})}\n\n"

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("stage") in ("completed", "error"):
                    break
            except asyncio.TimeoutError:
                # Keep-alive heartbeat ping
                yield ": ping\n\n"
                current_status = tasks_state.get(task_id, {}).get("status")
                if current_status in ("completed", "failed"):
                    break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/tasks/{task_id}")
def get_task_details(task_id: str):
    """
    Returns full results, metadata, and download links for the processed task.
    """
    if task_id not in tasks_state:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks_state[task_id]
    return {
        "task_id": task["task_id"],
        "status": task["status"],
        "stage": task.get("stage"),
        "message": task.get("message"),
        "percent": task.get("percent", 0),
        "result": task.get("result"),
        "error": task.get("error"),
    }


@app.get("/api/sample")
def get_sample_clips():
    """
    Provides instant sample demonstration data for zero-latency testing and evaluation.
    """
    return {
        "task_id": "sample_demo",
        "status": "completed",
        "source_title": "The Secrets of High-Retention Content",
        "total_duration": 180.0,
        "elapsed_seconds": 24.5,
        "clips": [
            {
                "id": "sample_1",
                "index": 1,
                "title": "The Biggest Mistake 99% Of Creators Make",
                "start_time": 12.4,
                "end_time": 39.8,
                "duration": 27.4,
                "virality_score": 96,
                "hook_type": "contrarian",
                "hook_rationale": "Opens with an intriguing counter-intuitive claim, challenges conventional wisdom, and ends on a curiosity cliffhanger.",
                "video_url": "/media/sample_1.mp4",
                "thumbnail_url": "/media/sample_1_thumb.jpg",
                "seo_pack": {
                    "youtube_shorts": {
                        "title_options": [
                            "The Biggest Mistake 99% Of Creators Make",
                            "Why Most Creators Fail in the First 3 Seconds",
                            "Stop Doing This If You Want Views",
                        ],
                        "description": "The harsh reality about video hooks and viewer retention that nobody tells you.\n\n#shorts #viral #creator",
                        "tags": ["shorts", "viral", "contentcreator", "growth"],
                    },
                    "tiktok": {
                        "caption": "The biggest mistake 99% of creators make 🤯 wait till the end! #fyp #creator #viral",
                        "hashtags": ["fyp", "foryou", "contentcreator", "viral"],
                    },
                    "instagram_reels": {
                        "caption": "Are you guilty of this? Be honest in the comments 👇\n\n#reels #creators #viral",
                        "call_to_action": "Save this reel for your next upload 📌",
                    },
                },
            }
        ],
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info("Starting ClippedAI Engine server on http://%s:%d", host, port)
    uvicorn.run("server:app", host=host, port=port, reload=True)
