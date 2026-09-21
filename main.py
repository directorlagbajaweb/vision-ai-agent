"""
main.py
VISION entry point: opens the HUD window and starts the Gemini Live
voice connection in a background thread, wired together so the ring
reflects idle/listening/speaking/muted state in real time.
"""

import asyncio
import os
import sys

# py2app's ssl recipe unconditionally points SSL_CERT_FILE/SSL_CERT_DIR at
# Contents/Resources/openssl.ca — a directory it only creates if the building
# interpreter has an on-disk CA bundle. The python.org framework builds report
# cafile=None (they rely on certifi), so the path never exists and the first
# TLS call dies with FileNotFoundError before the window even opens. Point both
# at the certifi bundle shipped in DATA_FILES instead. This has to run before
# anything constructs an SSLContext, so it stays above the other imports.
if getattr(sys, "frozen", False):
    _ca = os.path.join(os.environ.get("RESOURCEPATH", ""), "certs", "cacert.pem")
    if os.path.exists(_ca):
        os.environ["SSL_CERT_FILE"] = _ca
        os.environ["SSL_CERT_DIR"] = os.path.dirname(_ca)
        os.environ["REQUESTS_CA_BUNDLE"] = _ca

import webview

import config
from voice.vision_live import VisionLive
from memory.db import init_db


class VisionAPI:
    """Exposed to JavaScript as window.pywebview.api"""

    def __init__(self):
        self.live = None  # set once the backend starts

    def toggle_mute(self, muted: bool):
        if self.live:
            self.live.set_muted(muted)
        return {"muted": muted}


def start_voice_backend(window, api):
    init_db()
    live = VisionLive(ui_window=window)
    api.live = live
    asyncio.run(live.run())


def main():
    # Resolved from the bundle/source location, never the working directory —
    # a double-clicked .app starts in "/".
    html_path = config.resource_path("ui", "static", "index.html")
    api = VisionAPI()

    window = webview.create_window(
        "VISION",
        str(html_path),
        js_api=api,
        width=900,
        height=740,
        min_size=(420, 520),
        background_color="#09090b",
        easy_drag=True,
    )

    webview.start(start_voice_backend, (window, api), http_server=True)


if __name__ == "__main__":
    main()