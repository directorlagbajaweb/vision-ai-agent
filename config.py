"""
config.py
Central configuration for VISION.
Loads API keys from .env and defines model routing, paths, and app settings.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent


def _resource_dir() -> Path:
    """Where VISION's bundled files (.env, ui/static) actually live.

    py2app sets sys.frozen and puts everything from DATA_FILES in
    Contents/Resources (symlinks back to this source tree in an alias build).
    Running from source, that's just the project directory. Either way this is
    resolved from the executable/module location, never from the working
    directory the app was launched with — a double-clicked .app starts in /.
    """
    if getattr(sys, "frozen", False):
        resources = Path(sys.executable).resolve().parent.parent / "Resources"
        if resources.is_dir():
            return resources
    return BASE_DIR


RESOURCE_DIR = _resource_dir()


def resource_path(*parts) -> Path:
    """A bundled file, falling back to the source tree if it isn't bundled."""
    bundled = RESOURCE_DIR.joinpath(*parts)
    return bundled if bundled.exists() else BASE_DIR.joinpath(*parts)


# Load variables from .env into the environment. Explicit path: load_dotenv()
# with no argument walks up from the working directory, which finds nothing
# when launched as an .app.
ENV_PATH = resource_path(".env")
load_dotenv(ENV_PATH if ENV_PATH.exists() else None)


def _data_dir() -> Path:
    """Where VISION writes: the conversation DB and logs.

    Running from source, that stays in the project. Inside a py2app bundle it
    has to move out — Contents/Resources is the wrong place to write to (it is
    read-only once the .app is signed or relocated), so use the standard
    per-user location instead.
    """
    if getattr(sys, "frozen", False):
        data = Path.home() / "Library" / "Application Support" / "VISION"
        (data / "memory").mkdir(parents=True, exist_ok=True)
        (data / "logs").mkdir(parents=True, exist_ok=True)
        return data
    return BASE_DIR


DATA_DIR = _data_dir()
MEMORY_DB_PATH = DATA_DIR / "memory" / "vision.db"
LOG_FILE_PATH = DATA_DIR / "logs" / "vision.log"

# ── API Keys (loaded from .env) ──────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")   # optional, added later
PORCUPINE_API_KEY = os.getenv("PORCUPINE_API_KEY")     # optional, added later
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SLACK_USER_TOKEN = os.getenv("SLACK_USER_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_USER_ID = os.getenv("SLACK_USER_ID")

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