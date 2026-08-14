"""
system_control/media_control.py
System-wide media playback control via simulated HARDWARE media key
presses (NX_SYSDEFINED CGEvents) -- this is deliberately NOT generic
mouse/keyboard GUI automation. It works regardless of which app or
website currently has focus (YouTube in Chrome, Safari, Prime Video,
Spotify, etc.) because it sends the same low-level signal a physical
keyboard's Play/Pause key would, rather than targeting any specific
on-screen button. That's why this lives in its own module, separate
from system_control/computer_control.py's click-based automation.

There is no universal way to verify actual playback state this way —
unlike Spotify's AppleScript player-state query, arbitrary apps/websites
expose nothing we can check. Every function here is honest about that:
"success" means the key press was sent, not that we confirmed what
happened as a result.

Investigated (2026-08-14) after a report that this was unreliable for a
YouTube tab in Chrome:
  - Event construction (NSEventTypeSystemDefined, subtype 8, NX_KEYTYPE_*
    data1 encoding, both key-down and key-up posted via CGEventPost) was
    checked against the standard known-good pattern for this technique
    and matches it.
  - Accessibility permission for the exact live-app process (the
    Python.app-launched interpreter running main.py) was confirmed
    granted via AXIsProcessTrusted().
  - Repeated, direct, real-world tests (a real HTML5 video in Chrome,
    plus the actual YouTube player, with Spotify simultaneously playing
    in the background) consistently showed the key correctly reaching
    Chrome, not Spotify -- no reproduction of Spotify intercepting it.
  - macOS's modern "Now Playing" session API (MediaRemote, private
    framework) was probed as a way to log which app currently owns
    system media-key routing, but it returns no registered app (pid 0)
    for every app tested on this machine/macOS version, including
    Apple's own Music.app -- i.e. it's locked down for third-party
    callers here and isn't a usable diagnostic, so it's not used below.
  - No exception was reproduced in isolated or threaded (asyncio.to_thread
    calls this synchronously from a worker thread, same as here) testing.
  - Conclusion: the mechanism itself checks out. The print statements
    below exist so that IF this recurs, the terminal shows the real
    exception/outcome instead of just a vague spoken "unable to X" with
    nothing to diagnose from.
"""

import Quartz
from AppKit import NSEvent, NSEventTypeSystemDefined

# NX_KEYTYPE_* constants (IOKit/hidsystem/ev_keymap.h) -- stable across
# macOS versions, the standard way to simulate hardware media keys from
# userspace without a physical keyboard.
NX_KEYTYPE_PLAY = 16
NX_KEYTYPE_NEXT = 17
NX_KEYTYPE_PREVIOUS = 18


def _post_media_key(key_code: int):
    """Simulates a full press-and-release of a hardware media key."""
    for key_down in (True, False):
        flags = 0xA00 if key_down else 0xB00
        data1 = (key_code << 16) | ((0xA if key_down else 0xB) << 8)
        event = NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
            NSEventTypeSystemDefined, (0, 0), flags, 0, 0, 0, 8, data1, -1
        )
        cg_event = event.CGEvent()
        if cg_event is None:
            raise RuntimeError("NSEvent.CGEvent() returned None -- could not construct the underlying CGEvent")
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, cg_event)


def toggle_media_playback() -> dict:
    """Sends the hardware Play/Pause media key -- toggles whatever app
    currently holds 'now playing' focus, the same as a real keyboard key."""
    print("[media_control] Sending Play/Pause media key (NX_KEYTYPE_PLAY)...")
    try:
        _post_media_key(NX_KEYTYPE_PLAY)
        print("[media_control] Play/Pause media key posted successfully.")
        return {
            "success": True,
            "detail": (
                "Play/Pause media key sent. Playback state can't be verified "
                "generically across arbitrary apps/websites -- this confirms "
                "the key press was sent, not what happened as a result."
            ),
        }
    except Exception as e:
        print(f"[media_control] Failed to send Play/Pause media key: {type(e).__name__}: {e}")
        return {"success": False, "error": f"Failed to send Play/Pause media key: {type(e).__name__}: {e}"}


def next_media_track() -> dict:
    """Sends the hardware Next-track media key."""
    print("[media_control] Sending Next-track media key (NX_KEYTYPE_NEXT)...")
    try:
        _post_media_key(NX_KEYTYPE_NEXT)
        print("[media_control] Next-track media key posted successfully.")
        return {
            "success": True,
            "detail": "Next-track media key sent. Result can't be verified generically.",
        }
    except Exception as e:
        print(f"[media_control] Failed to send next-track media key: {type(e).__name__}: {e}")
        return {"success": False, "error": f"Failed to send next-track media key: {type(e).__name__}: {e}"}


def previous_media_track() -> dict:
    """Sends the hardware Previous-track media key."""
    print("[media_control] Sending Previous-track media key (NX_KEYTYPE_PREVIOUS)...")
    try:
        _post_media_key(NX_KEYTYPE_PREVIOUS)
        print("[media_control] Previous-track media key posted successfully.")
        return {
            "success": True,
            "detail": "Previous-track media key sent. Result can't be verified generically.",
        }
    except Exception as e:
        print(f"[media_control] Failed to send previous-track media key: {type(e).__name__}: {e}")
        return {"success": False, "error": f"Failed to send previous-track media key: {type(e).__name__}: {e}"}
