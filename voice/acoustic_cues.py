"""
voice/acoustic_cues.py
Short synthesized acoustic acknowledgments ("Rocky"-style) that cover the
gap while VISION is processing — a "thinking" blip right after the user
stops talking, "success"/"error" chirps after a tool result, "waiting"
for a longer-running action. Pure stdlib tone synthesis, no new dependency
and no external sound-asset files.

Deliberately plays via a one-shot sd.play() call rather than the
persistent RawOutputStream vision_live.py uses for Gemini's own audio
output, so cues never contend with or interleave into that stream.
"""

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
    """Synthesizes (or reuses a cached) short tone and plays it immediately,
    non-blocking. Unknown cue_type is a silent no-op rather than an error —
    an acoustic cue should never be the thing that breaks a call site."""
    if cue_type not in CUE_PRESETS:
        print(f"[acoustic_cues] Unknown cue_type: {cue_type}")
        return

    if cue_type not in _cache:
        _cache[cue_type] = _synth_tone(*CUE_PRESETS[cue_type])

    try:
        sd.play(_cache[cue_type], samplerate=SAMPLE_RATE, blocking=False)
    except Exception as e:
        print(f"[acoustic_cues] Playback failed: {e}")
