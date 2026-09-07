import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Base directories
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(PROJECT_DIR / "storage"))).resolve()
FONTS_DIR = (BACKEND_DIR / "fonts").resolve()

# Dedicated cache directories for deterministic reuse
SOURCES_CACHE_DIR = (STORAGE_DIR / "sources").resolve()
AUDIO_CACHE_DIR = (STORAGE_DIR / "audio_cache").resolve()
TRANSCRIPTS_CACHE_DIR = (STORAGE_DIR / "transcripts").resolve()
VIRALITY_CACHE_DIR = (STORAGE_DIR / "virality_cache").resolve()
ASD_CACHE_DIR = (STORAGE_DIR / "asd_cache").resolve()

for cache_d in (
    STORAGE_DIR,
    SOURCES_CACHE_DIR,
    AUDIO_CACHE_DIR,
    TRANSCRIPTS_CACHE_DIR,
    VIRALITY_CACHE_DIR,
    ASD_CACHE_DIR,
):
    cache_d.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(PROJECT_DIR / ".env")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Silence noisy external libraries
for lib in ("urllib3", "httpx", "httpcore", "yt_dlp"):
    logging.getLogger(lib).setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    return logger

# API Keys
def get_assemblyai_key() -> str:
    key = os.getenv("ASSEMBLYAI_KEY", "")
    if not key:
        raise ValueError("ASSEMBLYAI_KEY is required for speech transcription.")
    return key

ASSEMBLYAI_KEY = get_assemblyai_key

def get_gemini_key() -> str:
    return os.getenv("GEMINI_API_KEY", "")

def get_groq_key() -> str:
    return os.getenv("GROQ_KEY", "")

# Video framing & render constants
OUT_WIDTH = 1080
OUT_HEIGHT = 1920
CROP_WIDTH_1 = 608  # 9:16 window width from 1080p source (1080 * 9 / 16 = 607.5 -> 608)

# Stabilization & tracking parameters
STATIONARY_STD_THRESHOLD = 0.06
CLUSTER_GAP = 0.04
SIGMA_SMOOTHING = 12
MIN_CLIP_DURATION = 20.0
MAX_CLIP_DURATION = 60.0

DEFAULT_FONT_PATH = FONTS_DIR / "Komika_Axis.ttf"
