"""
fast_asd_local.py — Local equivalent of the modal_fast_asd.py FastASDTracker.

Mirrors the exact same interface as the Modal class:
    tracker = LocalFastASDTracker()
    tracker.setup()
    result_json = tracker.process_video(video_bytes)

Runs TalkNet + S3FD face detection locally on Apple Silicon (MPS) or CPU
without any Modal dependency.
"""

import json
import logging
import os
import pathlib
import sys
import tempfile

logger = logging.getLogger("fast_asd_local")

# Absolute paths — independent of working directory
TALKNET_DIR = str(pathlib.Path(__file__).parent / "fast-asd" / "talknet")
MODEL_PATH = str(pathlib.Path(TALKNET_DIR) / "pretrain_TalkSet.model")
S3FD_PATH = str(pathlib.Path(TALKNET_DIR) / "model" / "faceDetector" / "s3fd" / "sfd_face.pth")


def _check_weights():
    missing = []
    if not pathlib.Path(MODEL_PATH).exists():
        missing.append(MODEL_PATH)
    if not pathlib.Path(S3FD_PATH).exists():
        missing.append(S3FD_PATH)
    if missing:
        raise RuntimeError(
            f"Fast-ASD model weights not found:\n" + "\n".join(f"  {p}" for p in missing) +
            "\nRun: cd backend && ./run_local.sh (they should auto-download on first run)"
        )


class LocalFastASDTracker:
    """
    Drop-in local replacement for modal_fast_asd.FastASDTracker.
    Call setup() once at server startup, then process_video() per clip.
    """
    _instance = None

    @classmethod
    def get_instance(cls) -> "LocalFastASDTracker":
        if cls._instance is None:
            cls._instance = LocalFastASDTracker()
            cls._instance.setup()
        return cls._instance

    def __init__(self):
        self.s = None
        self.DET = None
        self._original_cwd = os.getcwd()

    def setup(self):
        """Load TalkNet model and S3FD face detector into memory (done once at startup)."""
        if self.s is not None and self.DET is not None:
            return
        _check_weights()
        logger.info("[FastASD] Loading TalkNet + S3FD models…")

        # demoTalkNet uses relative paths for save/ dir — must run from talknet dir
        os.chdir(TALKNET_DIR)
        if TALKNET_DIR not in sys.path:
            sys.path.insert(0, TALKNET_DIR)

        import demoTalkNet  # noqa: PLC0415
        self.s, self.DET = demoTalkNet.setup()
        os.chdir(self._original_cwd)
        logger.info("[FastASD] Models loaded and ready")

    def process_video(self, video_bytes: bytes) -> str:
        """
        Run full ASD pipeline on a video clip.

        Args:
            video_bytes: Raw MP4 bytes of the extracted clip.

        Returns:
            JSON string — list of {frame_number, faces: [{x1,y1,x2,y2,raw_score,speaking}]}
            Identical format to what modal_fast_asd.FastASDTracker.process_video returns.
        """
        if self.s is None or self.DET is None:
            self.setup()

        # ── Fast-ASD Caching Layer ──
        import hashlib
        import pathlib
        cache_dir = pathlib.Path.home() / ".clippedai" / "cache" / "asd"
        cache_dir.mkdir(parents=True, exist_ok=True)
        video_hash = hashlib.sha256(video_bytes).hexdigest()
        cache_file = cache_dir / f"fastasd_{video_hash}.json"

        if cache_file.exists():
            logger.info(f"[FastASD] 🟢 Cache hit! Skipping Face Detection & TalkNet for hash {video_hash[:8]}")
            return cache_file.read_text("utf-8")

        alt_cache_dir = pathlib.Path(__file__).parent.parent / "storage" / "asd_cache"
        alt_cache_file = alt_cache_dir / f"asd_{video_hash}.json"
        if alt_cache_file.exists():
            logger.info(f"[FastASD] 🟢 Alt cache hit! Skipping Face Detection for hash {video_hash[:8]}")
            content = alt_cache_file.read_text("utf-8")
            cache_file.write_text(content, "utf-8")
            return content

        logger.info(f"[FastASD] 🔴 Cache miss. Running Face Detection natively...")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tf.write(video_bytes)
            tf_path = tf.name

        try:
            # Must chdir to talknet dir for relative save/ paths inside demoTalkNet.main()
            os.chdir(TALKNET_DIR)
            if TALKNET_DIR not in sys.path:
                sys.path.insert(0, TALKNET_DIR)

            import demoTalkNet  # noqa: PLC0415

            logger.info(f"[FastASD] Processing {os.path.getsize(tf_path) / 1e6:.1f}MB clip")
            results = demoTalkNet.main(
                s=self.s,
                DET=self.DET,
                video_path=tf_path,
                start_seconds=0,
                end_seconds=-1,
                return_visualization=False,
                in_memory_threshold=0,  # force disk mode for stability (same as modal_fast_asd.py)
            )
            result_json = json.dumps(results)
            cache_file.write_text(result_json, "utf-8")
            logger.info(f"[FastASD] Cached results securely to {cache_file.name}")
            return result_json

        finally:
            os.chdir(self._original_cwd)
            if os.path.exists(tf_path):
                os.remove(tf_path)
