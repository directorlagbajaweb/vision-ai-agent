"""
setup.py
Packages VISION as a macOS .app with py2app.

    # fast, uses this venv's site-packages via symlinks — iterate on this first
    .venv/bin/python setup.py py2app -A

    # slow, fully standalone — only once the alias build is known good
    .venv/bin/python setup.py py2app

Notes for the standalone build (not needed for the alias build):
  * sounddevice / pywhispercpp / chromadb all ship compiled libraries, so they
    are listed in `packages` rather than `includes` — py2app copies a package
    directory wholesale, which keeps their .dylib/.so payloads next to the
    Python modules that dlopen them.
  * argv_emulation stays off: it needs Carbon, which is gone on arm64.
"""

import sys

import certifi
from setuptools import setup

# py2app's dependency scanner (modulegraph) walks the import graph recursively,
# burning ~8 Python frames per module, and re-enters itself for every nested
# import. The torch/chromadb/sentence-transformers graph is deep enough to blow
# the default 1000-frame limit before the build even starts copying files.
sys.setrecursionlimit(10_000)

APP = ["main.py"]

# Everything VISION reads at runtime. These land in Contents/Resources, which
# config.RESOURCE_DIR resolves to — so nothing depends on the working
# directory the app happens to be launched from.
DATA_FILES = [
    ("", [".env"]),
    # A real on-disk CA bundle. py2app's ssl recipe assumes the building
    # interpreter has one; the python.org builds do not, so without this every
    # HTTPS call in the packaged app fails (see the note in main.py).
    ("certs", [certifi.where()]),
    ("ui/static", [
        "ui/static/index.html",
        "ui/static/style.css",
        "ui/static/app.js",
    ]),
]

OPTIONS = {
    "argv_emulation": False,
    "iconfile": "AppIcon.icns",
    "plist": {
        "CFBundleName": "VISION",
        "CFBundleDisplayName": "VISION",
        "CFBundleIdentifier": "local.vision.app",
        "CFBundleVersion": "1.0.0",
        "CFBundleShortVersionString": "1.0.0",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        # TCC prompt strings. Screen Recording and Accessibility have no
        # Info.plist key — macOS prompts for those on first use and they are
        # granted per-app in System Settings, so the new bundle identity has
        # to be granted them again.
        "NSMicrophoneUsageDescription":
            "VISION listens for the wake word and for what you say to it.",
        "NSCameraUsageDescription":
            "VISION can look through the camera when you ask it to.",
        "NSAppleEventsUsageDescription":
            "VISION controls apps like Spotify, Calendar and Mail on your behalf.",
    },
    "packages": [
        # native / compiled payloads. Only real packages belong here —
        # py2app resolves each one with the legacy imp.find_module, which
        # needs a directory with an __init__.py. A PEP 420 namespace package
        # (google, which is how google-genai ships) fails the build outright
        # with "No module named 'google'", and a single-file module
        # (sounddevice.py) has no package directory to collect. Both are
        # picked up by the dependency scanner from the imports anyway; only
        # _sounddevice_data needs naming, because it carries the PortAudio
        # dylib that sounddevice dlopens at runtime.
        "_sounddevice_data",
        "pywhispercpp",
        "chromadb",
        "sentence_transformers",
        "cv2",
        "numpy",
        "PIL",
        "mss",
        # pure python, but big / dynamically imported
        # anyio picks its backend at runtime (anyio._backends._asyncio), which
        # the static scanner never sees — without this, httpx client teardown
        # raises ModuleNotFoundError inside the packaged app.
        "anyio",
        "webview",
        "tavily",
        "slack_sdk",
        "dotenv",
        "requests",
        "openai",
    ],
    "includes": [
        # Resolved through modulegraph's real import machinery, which does
        # understand namespace packages and single-file modules.
        "sounddevice",
        "google.genai",
        "objc",
        "Foundation",
        "AppKit",
        "Quartz",
        "WebKit",
    ],
    "excludes": [
        "tkinter",
        "matplotlib",
        "pytest",
    ],
}

setup(
    app=APP,
    name="VISION",
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
