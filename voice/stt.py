"""
voice/stt.py
Speech-to-text for VISION. Records from the mic, then transcribes
using whisper.cpp (local, free, runs on-device). The model is loaded
ONCE and reused across calls instead of reloading every time.
"""

import sys
import tempfile
import threading
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import sounddevice as sd
import numpy as np
import wave

SAMPLE_RATE = 16000  # whisper expects 16kHz audio

# sd.rec()/sd.wait() share ONE global (sounddevice._last_callback): starting a
# new recording force-stops-and-closes whatever stream that global still
# points at. record_audio() runs inside asyncio.to_thread(), and cancelling
# the asyncio Task around it (e.g. a proactive trigger cutting off the
# dormant wake-word loop) does NOT stop the underlying thread -- the
# cancelled-but-still-running call keeps waiting on its own stream's
# completion event. If a fresh record_audio() call starts before that
# orphaned call finishes, the new call's stop-the-previous-stream step and
# the orphaned call's own close-when-done step can both call close() on the
# exact same PortAudio stream from two threads at once: a real double free
# that crashes native code with SIGABRT, not a catchable Python exception.
# Serializing every recording through this lock means an orphaned call is
# merely waited out, never raced.
_record_lock = threading.Lock()

# How much longer than the requested duration a recording may take before
# it is treated as a wedged audio device. sd.wait() waits on a
# threading.Event with NO timeout, so if PortAudio stops delivering
# callbacks (device unplugged mid-record, a Bluetooth output dropping, a
# CoreAudio glitch after sleep/wake) it blocks forever. That call sits in
# the wake-word loop via asyncio.to_thread, so a single hang there means
# VISION never hears its name again -- permanently deaf, with the window
# still up and no crash to show for it.
RECORD_TIMEOUT_MARGIN = 5.0

# ── Load the model ONCE at import time, reuse for every call ──
from pywhispercpp.model import Model
_model = Model("base")


def _wait_for_recording(duration: float) -> None:
    """sd.wait() with a deadline.

    sounddevice exposes no timeout on its blocking wait, so wait on the
    same completion event directly and abort the stream if it never fires.
    Falls back to the plain wait only if sounddevice's internals change
    shape under us.
    """
    timeout = duration + RECORD_TIMEOUT_MARGIN
    callback = getattr(sd, "_last_callback", None)
    event = getattr(callback, "event", None)

    if event is None:
        sd.wait()
        return

    if not event.wait(timeout=timeout):
        # stop() stops and closes the stream, releasing the device so the
        # next attempt can succeed once the hardware settles. Guard with
        # .closed: under _record_lock this stream can no longer be raced by
        # another record_audio() call, but stop() also affects whatever
        # sd.rec()/sd.play() elsewhere last touched _last_callback, so it's
        # cheap insurance against closing something already closed.
        try:
            if not callback.stream.closed:
                sd.stop()
        except Exception:
            pass
        raise TimeoutError(
            f"Audio capture stalled: no completion after {timeout:.1f}s "
            f"for a {duration:.1f}s recording. The input device is not "
            f"delivering callbacks."
        )

    # The normal path closes the stream inside wait(); do the same here.
    try:
        if not callback.stream.closed:
            callback.stream.close(True)
    except Exception:
        pass


def record_audio(duration: float = 5.0) -> str:
    """Records from the default microphone for `duration` seconds.

    Raises TimeoutError if the device stops producing audio, rather than
    blocking forever.
    """
    with _record_lock:
        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        _wait_for_recording(duration)

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