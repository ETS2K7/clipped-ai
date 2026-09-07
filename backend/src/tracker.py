import os
import sys
import json
import hashlib
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
import scipy.ndimage as ndimage

from config import (
    get_logger,
    STORAGE_DIR,
    OUT_WIDTH,
    OUT_HEIGHT,
    STATIONARY_STD_THRESHOLD,
    CLUSTER_GAP,
    SIGMA_SMOOTHING,
)

logger = get_logger(__name__)

TALKNET_DIR = (Path(__file__).resolve().parent.parent / "fast-asd" / "talknet").resolve()
ASD_CACHE_DIR = STORAGE_DIR / "asd_cache"
ASD_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class FastASDManager:
    """
    Manages the TalkNet + S3FD model instances and executes audio-visual active speaker detection.
    """
    _instance = None

    def __init__(self):
        self.talknet_model = None
        self.face_detector = None
        self._initialized = False

    @classmethod
    def get_instance(cls) -> "FastASDManager":
        if cls._instance is None:
            cls._instance = FastASDManager()
        return cls._instance

    def initialize(self):
        if self._initialized:
            return

        model_path = TALKNET_DIR / "pretrain_TalkSet.model"
        s3fd_path = TALKNET_DIR / "model" / "faceDetector" / "s3fd" / "sfd_face.pth"

        if not model_path.exists() or not s3fd_path.exists():
            raise FileNotFoundError(
                f"Missing Fast-ASD weights: {model_path} or {s3fd_path}"
            )

        logger.info("Initializing Fast-ASD (TalkNet + S3FD)...")
        original_cwd = os.getcwd()
        try:
            os.chdir(TALKNET_DIR)
            if str(TALKNET_DIR) not in sys.path:
                sys.path.insert(0, str(TALKNET_DIR))

            import demoTalkNet
            self.talknet_model, self.face_detector = demoTalkNet.setup()
            self._initialized = True
            logger.info("Fast-ASD models loaded and ready.")
        finally:
            os.chdir(original_cwd)

    def detect_active_speakers(
        self,
        video_path: str,
        cache_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs TalkNet + S3FD on the specified video file.
        Uses named cache_key and SHA-256 disk cache to skip inference on already processed media.
        """
        # 1. Fast check: named cache_key
        if cache_key:
            named_cache = ASD_CACHE_DIR / f"asd_{cache_key}.json"
            if named_cache.exists():
                logger.info("Loaded Fast-ASD tracking from cache: %s", named_cache.name)
                with open(named_cache, "r", encoding="utf-8") as f:
                    return json.load(f)

        # 2. File content hash check
        with open(video_path, "rb") as f:
            file_bytes = f.read()
        clip_hash = hashlib.sha256(file_bytes).hexdigest()
        cache_file = ASD_CACHE_DIR / f"asd_{clip_hash}.json"

        if cache_file.exists():
            logger.info("Loaded Fast-ASD tracking from hash cache: %s", cache_file.name)
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if cache_key:
                    try:
                        with open(ASD_CACHE_DIR / f"asd_{cache_key}.json", "w", encoding="utf-8") as ff:
                            json.dump(data, ff)
                    except OSError:
                        pass
                return data

        self.initialize()
        logger.info("Running Fast-ASD tracking on %s...", video_path)
        original_cwd = os.getcwd()
        try:
            os.chdir(TALKNET_DIR)
            if str(TALKNET_DIR) not in sys.path:
                sys.path.insert(0, str(TALKNET_DIR))

            import demoTalkNet
            results = demoTalkNet.main(
                s=self.talknet_model,
                DET=self.face_detector,
                video_path=video_path,
                start_seconds=0,
                end_seconds=-1,
                return_visualization=False,
                in_memory_threshold=0,
            )

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(results, f)

            if cache_key:
                try:
                    with open(ASD_CACHE_DIR / f"asd_{cache_key}.json", "w", encoding="utf-8") as ff:
                        json.dump(results, ff)
                except OSError:
                    pass

            logger.info("Fast-ASD tracking complete: cached to %s", cache_file.name)
            return results
        finally:
            os.chdir(original_cwd)


def compute_smoothed_camera_positions(
    tracking_data: List[Dict[str, Any]],
    total_frames: int,
    default_center: float = 0.5,
) -> np.ndarray:
    """
    Computes smooth horizontal camera position coordinates for each video frame.
    Uses stationary cluster locking to prevent micro-jitter on still subjects,
    while smoothly tracking genuinely moving subjects with Gaussian smoothing.
    """
    raw_positions = np.full(total_frames, -1.0)
    frame_faces: Dict[int, List[Dict[str, Any]]] = {}

    for item in tracking_data:
        frame_faces[item.get("frame_number", 0)] = item.get("faces", [])

    for frame_idx in range(total_frames):
        faces = frame_faces.get(frame_idx, [])
        if not faces:
            continue

        # Look for the active speaker first
        active_speaker = None
        for face in faces:
            if face.get("speaking", False):
                active_speaker = face
                break

        # Fallback to the face with the highest raw confidence score
        if not active_speaker:
            active_speaker = max(faces, key=lambda f: f.get("raw_score", -999.0))

        # Normalized horizontal center [0, 1]
        cx = (active_speaker["x1"] + active_speaker["x2"]) / 2.0
        raw_positions[frame_idx] = float(np.clip(cx, 0.0, 1.0))

    valid_mask = raw_positions != -1.0
    if not np.any(valid_mask):
        return np.full(total_frames, default_center)

    valid_points = raw_positions[valid_mask]

    # Cluster stationary positions
    sorted_pts = np.sort(valid_points)
    gaps = np.diff(sorted_pts)
    gap_indices = np.where(gaps > CLUSTER_GAP)[0]
    boundaries = np.concatenate([[-1], gap_indices, [len(sorted_pts) - 1]])

    clusters = []
    for i in range(len(boundaries) - 1):
        start = int(boundaries[i]) + 1
        end = int(boundaries[i + 1]) + 1
        clusters.append(sorted_pts[start:end])

    largest_cluster = max(clusters, key=len)
    is_stationary = float(np.std(largest_cluster)) < STATIONARY_STD_THRESHOLD

    if is_stationary:
        # Station locked mode: lock camera to cluster centers to avoid micro-jitter
        cluster_centers = [float(np.median(c)) for c in clusters]
        out = np.full(total_frames, default_center)

        first_valid_idx = np.where(valid_mask)[0][0]
        current_center = min(
            cluster_centers,
            key=lambda c: abs(c - raw_positions[first_valid_idx]),
        )

        for i in range(total_frames):
            if raw_positions[i] != -1.0:
                target = raw_positions[i]
                best_center = min(cluster_centers, key=lambda c: abs(c - target))
                if abs(current_center - best_center) > CLUSTER_GAP:
                    current_center = best_center
            out[i] = current_center

        # Enforce minimum shot duration (25 frames ~ 1s) to prevent rapid jump flickering
        min_shot_frames = 25
        stabilized = out.copy()
        run_start = 0

        for i in range(1, total_frames):
            if out[i] != out[i - 1]:
                if (i - run_start) < min_shot_frames:
                    if run_start > 0:
                        stabilized[run_start:i] = stabilized[run_start - 1]
                    else:
                        stabilized[run_start:i] = out[i]
                run_start = i

        return stabilized

    # Moving subject mode: fill gaps via linear interpolation + Gaussian smoothing
    filled = raw_positions.copy()
    indices = np.arange(total_frames)
    first_valid = int(np.argmax(valid_mask))
    last_valid = int(total_frames - 1 - np.argmax(valid_mask[::-1]))

    filled[:first_valid] = filled[first_valid]
    filled[last_valid + 1:] = filled[last_valid]

    unfilled_mask = filled == -1.0
    if np.any(unfilled_mask):
        filled[unfilled_mask] = np.interp(
            indices[unfilled_mask],
            indices[~unfilled_mask],
            filled[~unfilled_mask],
        )

    effective_sigma = min(SIGMA_SMOOTHING, max(1, total_frames // 4))
    return ndimage.gaussian_filter1d(filled, sigma=effective_sigma)


def render_portrait_crop(
    input_video_path: str,
    output_video_path: str,
    smoothed_cx: np.ndarray,
) -> str:
    """
    Renders input 16:9 video to 9:16 portrait (1080x1920) following smoothed speaker coordinates.
    Muxes the original audio track into the final output.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Standard 9:16 crop width for 1080p source is 608px
    crop_w = int(round(height * 9.0 / 16.0))
    crop_w = min(crop_w, width)

    # Use temporary file for intermediate video stream
    temp_dir = Path(output_video_path).parent
    temp_video = str(temp_dir / f"temp_tracked_{Path(output_video_path).stem}.mp4")

    # Use FFmpeg pipe for high-efficiency encoding
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{OUT_WIDTH}x{OUT_HEIGHT}",
        "-pix_fmt", "bgr24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        temp_video,
    ]

    pipe = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        cx_val = smoothed_cx[frame_idx] if frame_idx < len(smoothed_cx) else 0.5
        center_x = int(cx_val * width)

        x_start = max(0, min(center_x - (crop_w // 2), width - crop_w))
        x_end = x_start + crop_w

        # Slice 9:16 region and resize to 1080x1920
        cropped_slice = frame[0:height, x_start:x_end]
        portrait_frame = cv2.resize(
            cropped_slice,
            (OUT_WIDTH, OUT_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )

        pipe.stdin.write(portrait_frame.tobytes())
        frame_idx += 1

    cap.release()
    pipe.stdin.close()
    pipe.wait()

    # Mux original audio into final MP4
    mux_cmd = [
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", input_video_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        output_video_path,
    ]

    subprocess.run(mux_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(temp_video):
        try:
            os.remove(temp_video)
        except OSError:
            pass

    logger.info("Rendered portrait clip: %s (%d frames)", output_video_path, frame_idx)
    return output_video_path
