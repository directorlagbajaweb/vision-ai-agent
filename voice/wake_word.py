"""
voice/wake_word.py
Continuous wake-word detection using Whisper. Model loads once
(via stt.py's shared instance) and is reused for every chunk.
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from voice.stt import record_audio, transcribe  # triggers one-time model load

WAKE_PHRASE = config.WAKE_WORD.lower()
CHUNK_DURATION = 3.5    # slightly longer for more reliable detection
COMMAND_DURATION = 5.0


def wait_for_wake_word():
    print(f"[wake_word] Listening for '{config.WAKE_WORD}'...")
    while True:
        audio_path = record_audio(duration=CHUNK_DURATION)
        text = transcribe(audio_path)

        if text and WAKE_PHRASE in text.lower():
            print(f"[wake_word] Wake word detected in: '{text}'")
            return

        time.sleep(0.2)


def listen_for_command() -> str:
    print("[wake_word] Listening for command...")
    audio_path = record_audio(duration=COMMAND_DURATION)
    return transcribe(audio_path)


def run_voice_loop(on_command):
    print("[wake_word] VISION voice loop started. Press Ctrl+C to stop.\n")
    try:
        while True:
            wait_for_wake_word()
            command_text = listen_for_command()

            if not command_text.strip():
                print("[wake_word] Didn't catch a command, going back to listening.\n")
                continue

            on_command(command_text)
            print()

    except KeyboardInterrupt:
        print("\n[wake_word] Voice loop stopped.")


if __name__ == "__main__":
    def handle_command(text):
        print(f"[wake_word] Command received: {text}")

    run_voice_loop(handle_command)