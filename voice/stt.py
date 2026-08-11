"""
voice/stt.py
Speech-to-text for VISION. Records from the mic, then transcribes
using whisper.cpp (local, free, runs on-device). The model is loaded
ONCE and reused across calls instead of reloading every time.
"""

import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import sounddevice as sd
import numpy as np
import wave

SAMPLE_RATE = 16000  # whisper expects 16kHz audio

# ── Load the model ONCE at import time, reuse for every call ──
from pywhispercpp.model import Model
_model = Model("base")


def record_audio(duration: float = 5.0) -> str:
    """Records from the default microphone for `duration` seconds."""
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()

    temp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(temp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    return temp_path


def transcribe(audio_path: str) -> str:
    """Transcribes a .wav file to text using the pre-loaded Whisper model."""
    try:
        segments = _model.transcribe(audio_path)
        text = " ".join(segment.text for segment in segments).strip()
        return text
    except Exception as e:
        print(f"[stt] Transcription failed: {e}")
        return ""
    finally:
        Path(audio_path).unlink(missing_ok=True)


def listen(duration: float = 5.0) -> str:
    """Main entry point: records then transcribes."""
    audio_path = record_audio(duration)
    return transcribe(audio_path)


if __name__ == "__main__":
    print("[stt] Recording for 5 seconds... speak now.")
    text = listen(duration=5.0)
    print(f"\nYou said: {text}")