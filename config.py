"""
config.py
Central configuration for VISION.
Loads API keys from .env and defines model routing, paths, and app settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env into the environment
load_dotenv()

# ── Paths ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MEMORY_DB_PATH = BASE_DIR / "memory" / "vision.db"
LOG_FILE_PATH = BASE_DIR / "logs" / "vision.log"

# ── API Keys (loaded from .env) ──────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")   # optional, added later
PORCUPINE_API_KEY = os.getenv("PORCUPINE_API_KEY")     # optional, added later
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# ── OpenRouter Settings ───────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
USER_NAME = "David"

# ── Model Routing ─────────────────────────────────────
# Free-first, paid-fallback chains.
# router.py decides WHICH chain to use (coding vs everyday).
# model_fallback.py walks each chain top to bottom until one responds.

CODING_MODEL_CHAIN = [
    "openrouter/free",                     # auto free-tier selector, tried first
    "deepseek/deepseek-v4-flash",          # cheap paid fallback (~$0.09/M input)
    "z-ai/glm-5.2",                        # stronger paid fallback for hard tasks
]

EVERYDAY_MODEL_CHAIN = [
    "openrouter/free",                     # auto free-tier selector, tried first
    "google/gemini-2.5-flash-lite",        # cheap paid fallback
    "deepseek/deepseek-v4-flash",          # secondary fallback
]

# ── Wake Word ─────────────────────────────────────────
WAKE_WORD = "vision"          # phrase VISION listens for
WAKE_WORD_SENSITIVITY = 0.5   # 0.0 (strict) to 1.0 (loose)

# ── App Behavior ──────────────────────────────────────
MAX_FALLBACK_ATTEMPTS = 3     # how many models to try before giving up
REQUEST_TIMEOUT_SECONDS = 30  # per-model timeout before falling back

# ── Sanity check on startup ───────────────────────────
def validate_config():
    """Called by main.py at startup to catch missing keys early."""
    missing = []
    if not OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            f"Check your .env file."
        )