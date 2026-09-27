"""
voice/acoustic_cues.py
Short synthesized acoustic acknowledgments ("Rocky"-style) that cover the
gap while VISION is processing — a "thinking" blip right after the user
stops talking, "success"/"error" chirps after a tool result, "waiting"
for a longer-running action. Pure stdlib tone synthesis, no new dependency
and no external sound-asset files.

Each cue plays on its own short-lived RawOutputStream, separate from the
persistent one vision_live.py uses for Gemini's own audio, so cues never
contend with or interleave into that stream.

Deliberately NOT sd.play(): play()/rec() share one global stream slot
(sounddevice._last_callback), and every play() first closes whatever stream
is in it. Cues fire from asyncio.to_thread workers while stt.record_audio()
closes its own sd.rec() stream on another thread, so two threads could call
Pa_CloseStream on the same pointer -- a double free that SIGABRTs the whole
app (the 2026-09-24 crash). A private stream is never touched by anyone else.
"""

import threading

import math
from array import array

import sounddevice as sd

SAMPLE_RATE = 24000
FADE_MS = 8  # short fade in/out to avoid audible clicks at start/end

# (frequency_start_hz, frequency_end_hz, duration_s, volume)
CUE_PRESETS = {
    "thinking": (600, 600, 0.12, 0.15),
    "success": (500, 800, 0.18, 0.18),
    "error": (500, 300, 0.18, 0.18),
    "waiting": (400, 400, 0.15, 0.10),
}

_cache: dict[str, array] = {}

# Serializes cues so a burst ("waiting" then "success") plays in order
# instead of opening overlapping output streams.
_play_lock = threading.Lock()


def _synth_tone(freq_start: float, freq_end: float, duration_s: float, volume: float) -> array:
    n_samples = int(SAMPLE_RATE * duration_s)
    fade_samples = max(1, int(SAMPLE_RATE * FADE_MS / 1000))
    buf = array("h", bytes(2 * n_samples))

    for i in range(n_samples):
        t = i / SAMPLE_RATE
        freq = freq_start + (freq_end - freq_start) * (i / max(1, n_samples - 1))
        sample = math.sin(2 * math.pi * freq * t)

        if i < fade_samples:
            sample *= i / fade_samples
        elif i > n_samples - fade_samples:
            sample *= (n_samples - i) / fade_samples

        buf[i] = int(sample * volume * 32767)

    return buf


def play_acoustic_cue(cue_type: str) -> None:
    """Synthesizes (or reuses a cached) short tone and plays it, blocking for
    the ~0.2s it lasts -- every call site runs this via asyncio.to_thread. Unknown cue_type is a silent no-op rather than an error —
    an acoustic cue should never be the thing that breaks a call site."""
    if cue_type not in CUE_PRESETS:
        print(f"[acoustic_cues] Unknown cue_type: {cue_type}")
        return

    if cue_type not in _cache:
        _cache[cue_type] = _synth_tone(*CUE_PRESETS[cue_type])

    try:
        with _play_lock, sd.RawOutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
        ) as stream:
            stream.write(_cache[cue_type].tobytes())
    except Exception as e:
        print(f"[acoustic_cues] Playback failed: {e}")
