"""
voice/tts.py
Text-to-speech for VISION. Defaults to macOS's built-in `say` command
(free, instant, no setup). Can switch to ElevenLabs for a more natural
voice by setting TTS_ENGINE = "elevenlabs" in config.py.
"""

import sys
import subprocess
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config


def speak_with_say(text: str, voice: str = "Samantha", rate: int = 190):
    """
    Uses macOS's built-in `say` command. Free, offline, no API key needed.
    Run `say -v ?` in Terminal to see all available voice names.
    """
    try:
        subprocess.run(
            ["say", "-v", voice, "-r", str(rate), text],
            check=True,
        )
    except Exception as e:
        print(f"[tts] 'say' failed: {e}")


def speak_with_elevenlabs(text: str):
    """
    Uses the ElevenLabs API for a more natural voice.
    Requires ELEVENLABS_API_KEY in .env.
    """
    if not config.ELEVENLABS_API_KEY:
        print("[tts] No ElevenLabs API key set — falling back to 'say'.")
        speak_with_say(text)
        return

    try:
        import requests

        voice_id = "21m00Tcm4TlvDq8ikWAM"  # default ElevenLabs voice, swap later
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        headers = {
            "xi-api-key": config.ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.4, "similarity_boost": 0.8},
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(response.content)
            temp_path = f.name

        subprocess.run(["afplay", temp_path], check=True)
        Path(temp_path).unlink(missing_ok=True)

    except Exception as e:
        print(f"[tts] ElevenLabs failed: {e} — falling back to 'say'.")
        speak_with_say(text)


def speak(text: str):
    """Main entry point — routes to whichever engine is configured."""
    if not text or not text.strip():
        return

    print(f"[tts] Speaking: {text[:80]}{'...' if len(text) > 80 else ''}")

    if config.TTS_ENGINE == "elevenlabs":
        speak_with_elevenlabs(text)
    else:
        speak_with_say(text)


# Quick manual test
if __name__ == "__main__":
    speak("Hello, I'm VISION. Voice output is now working.")