"""
main.py
VISION entry point: opens the HUD window and starts the Gemini Live
voice connection in a background thread, wired together so the ring
reflects idle/listening/speaking/muted state in real time.
"""

import asyncio
from pathlib import Path

import webview
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
    html_path = Path(__file__).parent / "ui" / "static" / "index.html"
    api = VisionAPI()

    window = webview.create_window(
        "VISION",
        str(html_path),
        js_api=api,
        width=900,
        height=700,
        background_color="#050810",
        easy_drag=True,
    )

    webview.start(start_voice_backend, (window, api), http_server=True)


if __name__ == "__main__":
    main()