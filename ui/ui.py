"""
ui/ui.py
Launches the VISION HUD as a native macOS window using pywebview,
and bridges JS calls to the Python brain (brain/router.py).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import webview
from brain.router import route, confirm_pending_action
from memory.db import init_db


class VisionAPI:
    """Exposed to JavaScript as window.pywebview.api"""

    def handle_message(self, text: str) -> dict:
        print(f"[ui] Received: {text}")
        result = route(text)
        return result

    def confirm_action(self, token: str, approved: bool) -> dict:
        # Unlike the voice pipeline, a real human click can reach this bridge,
        # so `approved=False` here can reflect a genuine user "no".
        return confirm_pending_action(token, approved)


def main():
    init_db()

    html_path = Path(__file__).parent / "static" / "index.html"

    api = VisionAPI()
    webview.create_window(
        "VISION",
        str(html_path),
        js_api=api,
        width=900,
        height=700,
        background_color="#050810",
        frameless=False,
        easy_drag=True,
    )
    webview.start(http_server=True)


if __name__ == "__main__":
    main()