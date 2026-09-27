"""
voice/vision_live.py
VISION's voice layer using Gemini Live API. Dormant/awake cycle gated
by saying "vision" — or triggered proactively. Supports mute, code
panel, search with images, code execution, full-screen webpage
rendering, screen/camera access, memory, file/calendar/email access,
reliability handling, and headphone-aware interruption + affective
(emotionally responsive) dialog.
"""

import sys
import io
import time
import json
import wave
import asyncio
import queue
import tempfile
import threading
import traceback
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import sounddevice as sd
import mss
import cv2
from PIL import Image
from google import genai
from google.genai import types

import config
import vision_logging
from memory.db import init_db, save_message, get_recent_history, save_fact, get_all_facts, delete_fact
from memory.semantic import add_message as semantic_add, search_relevant as semantic_search
from system_control import dispatcher
from system_control import computer_control
from system_control import slack_control
from system_control.dispatcher import dispatch
from system_control.mac_actions import get_calendar_events
from voice.stt import record_audio, transcribe
from voice.acoustic_cues import play_acoustic_cue

LIVE_MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

WAKE_PHRASE = config.WAKE_WORD.lower()
WAKE_CHUNK_DURATION = 3.0
SHUTDOWN_GRACE_PERIOD = 2.0

# How long the mic can go idle (no real user speech, no VISION speaking)
# before forwarding to Gemini Live auto-pauses. This does NOT end the
# session or close the connection -- see set_muted()/_muted_auto below.
AUTO_MUTE_IDLE_SECONDS = 75
SCREEN_CAPTURE_INTERVAL = 2.0
CAMERA_CAPTURE_INTERVAL = 2.0

# Hard ceilings on anything that could otherwise block forever. Every one
# of these guards a call that has no timeout of its own: an unacknowledged
# websocket write, a WKWebView that never runs its completion handler, or
# an audio device that stops draining. Exceeding one is treated as a dead
# connection -- the task raises, and _run_session reconnects.
SESSION_SEND_TIMEOUT = 20.0      # a single send_* on the Live websocket
SESSION_SEND_LOCK_TIMEOUT = 30.0 # waiting for another task's send to finish
AUDIO_WRITE_TIMEOUT = 10.0       # one blocking write to the output device
UI_EVAL_TIMEOUT = 5.0            # one evaluate_js round-trip into the HUD
# Beyond this many queued HUD updates the bridge is presumed wedged and
# further updates are dropped rather than queued forever. The HUD is a
# view; losing frames of it must never stall the voice pipeline.
UI_QUEUE_LIMIT = 200

PROACTIVE_CHECK_INTERVAL = 60
PROACTIVE_LOOKAHEAD_MINUTES = 10

HEADPHONE_KEYWORDS = ["headphone", "airpods", "earpods", "beats", "earbuds", "headset"]

SYSTEM_PROMPT = (
    f"You are VISION, a warm and genuinely present voice companion running on "
    f"{config.USER_NAME}'s Mac. Talk like a thoughtful friend having a real "
    f"conversation, not a formal assistant reading out information — use natural "
    f"phrasing, brief reactions, and let your tone genuinely shift with what "
    f"{config.USER_NAME} is feeling. If they sound stressed, be calm and steady. "
    f"If they're excited, share that energy. If something's funny, you can be "
    f"a little playful. Keep responses short and conversational since you're "
    f"speaking out loud — you don't need to over-explain or over-apologize. "
    f"Use your tools directly when the user asks you to open apps, close apps, "
    f"open links, adjust volume, or run Shortcuts — don't just describe how to do it. "
    f"For close_app and run_shortcut specifically: call the tool first; if it "
    f"returns a confirmation_token, ask the user to confirm out loud, then call "
    f"the tool again passing that same confirmation_token only after they say yes. "
    f"Whenever a dedicated tool exists for what you're being asked to do, use it — "
    f"Spotify is the clear example: to play/pause/resume/skip a track, call "
    f"play_spotify_track/pause_spotify/resume_spotify/next_spotify_track directly. "
    f"For generic 'pause this'/'play this'/'skip this' where the target isn't "
    f"specifically Spotify (a YouTube video, something in Safari, Prime Video, etc.), "
    f"call toggle_media_playback/next_media_track/previous_media_track — these send "
    f"a real hardware media-key press and work regardless of what has focus, but "
    f"cannot verify what actually happened, so say something like 'I sent play/pause' "
    f"rather than claiming to know it's now playing. If NO dedicated tool exists for "
    f"a requested action, tell the user honestly that you can't do that yet — do not "
    f"attempt to click through an app's UI or read the screen to work around it; that "
    f"capability has been intentionally disabled as unreliable. "
    f"Whenever you write or provide code, ALWAYS call the show_code tool with the "
    f"code instead of speaking it aloud — just give a brief spoken summary of what "
    f"it does. "
    f"When asked to build a webpage, landing page, or website, write complete HTML "
    f"with inline CSS/JS and call show_code with language 'html'. If the user then "
    f"asks you to run, show, or preview it, call render_webpage with that same full "
    f"HTML content — it will render live and full-screen for them. render_webpage is "
    f"ONLY for pages you wrote yourself — never call it to try to show a real "
    f"website like YouTube, Spotify's web player, or any site with a real URL; it "
    f"cannot load real external content, and doing this instead of actually "
    f"navigating a real browser is a mistake. "
    f"When the user asks what's on their screen, call view_screen, describe what "
    f"you see, then ask if they want you to keep watching; only call "
    f"stop_viewing_screen once they confirm. "
    f"When the user asks what they're holding or asks you to look at something, "
    f"call view_camera, describe what you see, and keep it on for follow-ups "
    f"until they ask you to turn it off — then call stop_camera. "
    f"Whenever the user tells you something durable and worth remembering long-term, "
    f"call remember_fact with a short key and the value. If they ask you to forget "
    f"something, call forget_fact. "
    f"If the user references something from a past conversation not in your current "
    f"context, call recall_memory to search their full conversation history. "
    f"When the user asks about their calendar, schedule, or upcoming events, call "
    f"get_calendar_events. When they ask about email or unread messages, call "
    f"get_recent_emails. When they ask you to read, open, or check a file, call "
    f"read_file with the full path. When they ask what's in a folder, call "
    f"list_directory. When they ask you to find a file, call search_files. "
    f"When asked about current events, live prices, or anything that could have "
    f"changed since your training, call web_search — results including images will "
    f"appear visually for the user, so just briefly summarize what you found out loud. "
    f"When you want to actually verify code works or compute something precisely, "
    f"call execute_python — if it returns a confirmation_token, ask the user to "
    f"confirm first, then call it again passing that confirmation_token. The "
    f"output will appear visually, so briefly state the result out loud. "
    f"When the user says 'close search', 'close this', 'close the page', or "
    f"similar, call close_visual_panel. "
    f"If this conversation was started BY YOU proactively (marked as a proactive "
    f"notification in the directive), briefly and naturally tell the user what you "
    f"noticed, then ask if they need anything else. "
    f"For a Slack DM or @-mention directive specifically: draft a reply, call "
    f"send_slack_message ONCE with no confirmation_token first (this only mints a "
    f"token — it never sends anything), THEN speak the draft to the user in the "
    f"form 'X in Y said: <summary>. I'd reply: <draft>. Send it?' — this spoken "
    f"question IS the confirmation ask, so if the user then says something "
    f"affirmative, call send_slack_message again with the confirmation_token you "
    f"already have and do NOT ask again. If they ask you to change the reply, "
    f"update the draft and repeat the whole silent-call-then-speak cycle (you'll "
    f"get a fresh token). If they decline ('skip it', 'no'), don't call the tool "
    f"again — just acknowledge and move on. If a confirmation_token comes back "
    f"expired (the conversation took a while), silently get a fresh one and ask "
    f"again rather than showing a raw error. Never send a Slack message under any "
    f"circumstance without the user having said something affirmative first. "
    f"If the user says something like 'shut down', 'go to sleep', or 'goodbye', "
    f"say a brief warm goodbye and call the shutdown_vision tool. You will wake up "
    f"again the next time the user says your name. "
    f"When the user asks you to find or open something specific on a website (a "
    f"video, a song, an article), use web_search to find a concrete, real URL, then "
    f"call open_url with that exact URL to navigate straight there. If playback "
    f"doesn't start on its own once there, use toggle_media_playback to send the "
    f"play key. Be accurate about what actually happened — never claim something is "
    f"playing or done unless you have real evidence of it (a tool's own confirmed "
    f"result, like Spotify's player-state check); for the media-key tools "
    f"specifically, say you sent the key rather than claiming to know the outcome. "
    f"VISION cannot click through an app's interface or read the screen to find "
    f"something to click — if a user's request would need that and no dedicated "
    f"tool covers it, say so honestly rather than attempting it."
)

TOOL_DECLARATIONS = [
    {
        "name": "open_app",
        "description": "Opens or brings focus to a macOS application.",
        "parameters": {"type": "OBJECT", "properties": {"app_name": {"type": "STRING"}}, "required": ["app_name"]},
    },
    {
        "name": "play_spotify_track",
        "description": (
            "Searches Spotify's catalog for a track and plays it in the Spotify app. "
            "Use this whenever asked to play a specific song/artist on Spotify. "
            "Verifies Spotify actually started playing before reporting success; if it "
            "returns success: false, the error explains exactly what went wrong "
            "(no track found, the command failed, or playback didn't actually start) "
            "so you can tell the user accurately."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {"query": {"type": "STRING", "description": "Song name and artist if known, e.g. 'Blinding Lights The Weeknd'"}},
            "required": ["query"],
        },
    },
    {
        "name": "pause_spotify",
        "description": "Pauses Spotify playback. Verifies it actually paused before reporting success.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "resume_spotify",
        "description": "Resumes/unpauses Spotify playback. Verifies it actually resumed before reporting success.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "next_spotify_track",
        "description": "Skips to the next track in Spotify. Verifies playback is in a real state afterward before reporting success.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "close_app",
        "description": (
            "Quits a macOS application. Call this WITHOUT confirmation_token first; "
            "if the result has status 'confirmation_required', ask the user to "
            "confirm out loud, then call this tool again passing the "
            "confirmation_token value you received."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {"app_name": {"type": "STRING"}, "confirmation_token": {"type": "STRING"}},
            "required": ["app_name"],
        },
    },
    {
        "name": "open_url",
        "description": "Opens a URL in the default browser.",
        "parameters": {"type": "OBJECT", "properties": {"url": {"type": "STRING"}}, "required": ["url"]},
    },
    {
        "name": "set_volume",
        "description": "Sets the system volume.",
        "parameters": {"type": "OBJECT", "properties": {"level": {"type": "INTEGER"}}, "required": ["level"]},
    },
    {
        "name": "run_shortcut",
        "description": (
            "Runs a macOS Shortcut by name. Call this WITHOUT confirmation_token first; "
            "if the result has status 'confirmation_required', ask the user to "
            "confirm out loud, then call this tool again passing the "
            "confirmation_token value you received."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {"shortcut_name": {"type": "STRING"}, "confirmation_token": {"type": "STRING"}},
            "required": ["shortcut_name"],
        },
    },
    {
        "name": "show_code",
        "description": (
            "Displays code in a panel on VISION's visual interface so the user "
            "can read and copy it. ALWAYS use this tool whenever asked to write, "
            "show, or provide code — do not speak the code itself aloud."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {"code": {"type": "STRING"}, "language": {"type": "STRING"}},
            "required": ["code"],
        },
    },
    {
        "name": "render_webpage",
        "description": (
            "Renders a complete HTML page full-screen so the user can see it live "
            "and interact with it — use this ONLY for a page YOU wrote yourself (a "
            "landing page, a demo, code you were asked to preview). NEVER use this "
            "to try to show a real external website (YouTube, a news site, anything "
            "with a real URL) — it can't load real external content. To open a real "
            "website, use open_url instead."
        ),
        "parameters": {"type": "OBJECT", "properties": {"html": {"type": "STRING"}}, "required": ["html"]},
    },
    {
        "name": "close_visual_panel",
        "description": (
            "Closes whatever visual panel or rendered page is currently showing "
            "(code, search results, execution result, or a rendered webpage) and "
            "returns to the normal display. Call this when the user says 'close "
            "search', 'close this', 'close the page', or similar."
        ),
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "view_screen",
        "description": "Starts capturing the user's screen so VISION can see what's on it.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "stop_viewing_screen",
        "description": "Stops capturing the screen. Call only after the user confirms.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "view_camera",
        "description": "Turns on the webcam so VISION can see what the user is showing it.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "stop_camera",
        "description": "Turns off the webcam. Call only when the user asks you to.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "remember_fact",
        "description": "Saves a durable fact about the user for long-term memory.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"key": {"type": "STRING"}, "value": {"type": "STRING"}},
            "required": ["key", "value"],
        },
    },
    {
        "name": "forget_fact",
        "description": "Removes a previously remembered fact.",
        "parameters": {"type": "OBJECT", "properties": {"key": {"type": "STRING"}}, "required": ["key"]},
    },
    {
        "name": "recall_memory",
        "description": "Searches the user's full conversation history by topic/meaning.",
        "parameters": {"type": "OBJECT", "properties": {"query": {"type": "STRING"}}, "required": ["query"]},
    },
    {
        "name": "get_calendar_events",
        "description": "Gets upcoming Calendar.app events. Call when the user asks about their schedule.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"hours_ahead": {"type": "INTEGER", "description": "How many hours ahead to check, default 24"}},
        },
    },
    {
        "name": "get_recent_emails",
        "description": "Gets recent unread emails from Mail.app.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"limit": {"type": "INTEGER", "description": "Max number of emails, default 5"}},
        },
    },
    {
        "name": "read_file",
        "description": "Reads the content of a text file at the given path.",
        "parameters": {"type": "OBJECT", "properties": {"path": {"type": "STRING"}}, "required": ["path"]},
    },
    {
        "name": "list_directory",
        "description": "Lists files in a directory.",
        "parameters": {"type": "OBJECT", "properties": {"path": {"type": "STRING"}}, "required": ["path"]},
    },
    {
        "name": "search_files",
        "description": "Searches for files matching a keyword within a directory.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"directory": {"type": "STRING"}, "keyword": {"type": "STRING"}},
            "required": ["directory", "keyword"],
        },
    },
    {
        "name": "web_search",
        "description": "Searches the web for real-time information — current events, live prices, anything outside your training data.",
        "parameters": {"type": "OBJECT", "properties": {"query": {"type": "STRING"}}, "required": ["query"]},
    },
    {
        "name": "execute_python",
        "description": (
            "Runs Python code and returns the actual output. Use this to verify code "
            "works correctly, or to solve math/data problems by actually computing them "
            "instead of guessing. Call this WITHOUT confirmation_token first; if the "
            "result has status 'confirmation_required', ask the user to confirm out "
            "loud, then call this tool again passing the confirmation_token value "
            "you received."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {"code": {"type": "STRING"}, "confirmation_token": {"type": "STRING"}},
            "required": ["code"],
        },
    },
    {
        "name": "send_slack_message",
        "description": (
            "Sends a Slack message (a reply to a DM or an @-mention). Call this "
            "WITHOUT confirmation_token first — this only mints a token, it never "
            "sends anything. Then speak your drafted reply to the user and ask "
            "them to confirm before calling this again with the confirmation_token "
            "you received. Never call this a second time without the user having "
            "actually said something affirmative in between."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "channel": {"type": "STRING", "description": "The Slack channel/DM ID to send to"},
                "text": {"type": "STRING", "description": "The message text to send"},
                "thread_ts": {"type": "STRING", "description": "Thread timestamp to reply in a thread, if applicable"},
                "confirmation_token": {"type": "STRING"},
            },
            "required": ["channel", "text"],
        },
    },
    {
        "name": "slack_catch_me_up",
        "description": (
            "Read-only Slack lookup: summarizes recent history for a channel or a "
            "person's DM out loud. Use for things like 'catch me up on #general', "
            "'what did Sarah and I talk about yesterday', 'summarize the design "
            "channel from this week'. Never sends or posts anything — separate from "
            "and unrelated to send_slack_message. If the result has "
            "needs_clarification: true, don't guess — ask the user which channel or "
            "person they meant, using the error message as a guide, then call this "
            "again with a more specific channel_or_person."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "channel_or_person": {
                    "type": "STRING",
                    "description": "A channel name (e.g. 'general', '#design') or a person's name (e.g. 'Sarah') to look up.",
                },
                "time_range": {
                    "type": "STRING",
                    "description": "How far back to look, loosely parsed: 'today', 'yesterday', 'this week'. Defaults to the last 24 hours if omitted or unrecognized.",
                },
            },
            "required": ["channel_or_person"],
        },
    },
    {
        "name": "shutdown_vision",
        "description": "Puts VISION to sleep until the user says its name again.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "toggle_media_playback",
        "description": (
            "Sends the hardware Play/Pause media key — works regardless of which "
            "app or website currently has focus (YouTube in a browser, Safari, Prime "
            "Video, etc.), the same way a physical keyboard's Play/Pause key would. "
            "Use this for generic 'pause this'/'play this'/'pause the video' requests "
            "where no more specific dedicated tool applies (e.g. prefer the Spotify "
            "tools for Spotify specifically). This does NOT click anything on screen "
            "and cannot verify whether it actually started or stopped playing — only "
            "that the key press was sent. Say so honestly rather than claiming to "
            "know the result."
        ),
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "next_media_track",
        "description": "Sends the hardware Next-track media key. Same focus-independent behavior and same honesty caveat as toggle_media_playback — can't verify the result.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "previous_media_track",
        "description": "Sends the hardware Previous-track media key. Same focus-independent behavior and same honesty caveat as toggle_media_playback — can't verify the result.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
]


def _log(msg: str):
    """Timestamped, line-flushed log. Used for everything that decides whether
    VISION can hear you — mute transitions and wake-word attempts — so that a
    "it stopped responding" report can be read straight off the terminal."""
    print(f"[{time.strftime('%H:%M:%S')}.{int(time.time() % 1 * 1000):03d}] [vision_live] {msg}", flush=True)


def _wav_level(path: str) -> float:
    """Peak-ish level of a recorded chunk, so the log can tell 'nobody spoke'
    apart from 'the mic handed us digital silence'."""
    try:
        with wave.open(path, "rb") as wf:
            return _rms_level(wf.readframes(wf.getnframes()))
    except Exception:
        return -1.0


def _rms_level(samples) -> float:
    """Perceptual 0..1 level for a chunk of int16 PCM, for the HUD's audio
    visualizer. Cheap enough to run inside the realtime audio callback."""
    try:
        arr = (np.frombuffer(samples, dtype=np.int16)
               if isinstance(samples, (bytes, bytearray, memoryview))
               else np.asarray(samples))
        if arr.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(np.square(arr.astype(np.float32)))))
        # int16 full scale is 32768; the sqrt curve lifts ordinary speech
        # into a range the bars can actually show instead of a flat nub.
        return min(1.0, (rms / 32768.0) ** 0.5 * 1.6)
    except Exception:
        return 0.0


def is_headphones_active() -> bool:
    """Checks the current default audio OUTPUT device name for headphone-like keywords."""
    try:
        device_info = sd.query_devices(kind="output")
        name = device_info.get("name", "")
        print(f"[vision_live] Detected output device: '{name}'")  # debug
        return any(kw in name.lower() for kw in HEADPHONE_KEYWORDS)
    except Exception as e:
        print(f"[vision_live] Could not detect output device: {e}")
        return False


class VisionLive:
    def __init__(self, ui_window=None):
        self.session = None
        self.audio_in_queue = None
        self.out_queue = None
        self._is_speaking = False
        self._active = False
        self._last_active_time = 0
        self._last_input_chunk_at = 0.0
        self._speaking_generation = 0
        # Guards every self.session.send_*() call -- the SDK holds no internal
        # lock, and concurrent writes to the same websocket from different
        # tasks (audio send, tool responses, live notification injection)
        # can corrupt the stream.
        self._session_send_lock = asyncio.Lock()
        # Created once, survives reconnects (unlike audio_in_queue/out_queue,
        # which are rebuilt per connection) -- a queued notification isn't
        # lost across a brief disconnect/reconnect.
        self._live_notification_queue = asyncio.Queue()
        self._tool_call_in_progress = False
        self._shutdown_event = None
        self.ui_window = ui_window
        self._muted = False           # manual mute (explicit "mute"/unmute button) -- only lifts on explicit unmute
        self._muted_auto = False      # auto-mute from inactivity -- lifts automatically on wake-word detection
        self._auto_mute_audio_buffer = bytearray()
        self._screen_active = False
        self._screen_task = None
        self._camera_active = False
        self._camera_task = None
        self._proactive_trigger = asyncio.Event()
        self._proactive_message = None
        self._notified_events = set()
        self._headphones_mode = False
        self._mic_level = 0.0
        self._agent_level = 0.0
        self._last_capture_at = 0.0     # last time the mic callback actually fired
        self._capture_stall_logged = False
        self._wake_attempts = 0
        self._silent_chunks = 0
        self._mic_drops = 0
        # HUD updates run on their own thread, never on the event loop.
        # pywebview's cocoa evaluate_js does AppHelper.callAfter(...) then
        # Semaphore.acquire() with NO timeout, so if the WKWebView never
        # runs its completion handler (web content process wedged or torn
        # down) the calling thread blocks forever. Calling that from the
        # event loop -- which is what every _set_ui_* site used to do --
        # freezes the whole voice pipeline: no audio in, no audio out, and
        # shutdown_vision can never run.
        #
        # Deliberately a plain daemon thread and not a ThreadPoolExecutor:
        # the executor registers an atexit hook that joins its workers, so
        # a wedged bridge thread would block interpreter shutdown -- the
        # "Cmd+Q does nothing, had to force-quit" symptom all over again.
        # A daemon thread is abandoned at exit instead.
        self._ui_queue = queue.Queue(maxsize=UI_QUEUE_LIMIT)
        self._ui_stalled = False
        self._ui_thread = threading.Thread(
            target=self._ui_bridge_loop, name="vision-ui-bridge", daemon=True
        )
        self._ui_thread.start()

    def _ui_bridge_loop(self):
        """Drains HUD updates, one at a time, off the event loop."""
        while True:
            script = self._ui_queue.get()
            started = time.monotonic()
            try:
                self.ui_window.evaluate_js(script)
            except Exception as e:
                print(f"[vision_live] HUD update failed: {e}")
            elapsed = time.monotonic() - started
            if elapsed > UI_EVAL_TIMEOUT:
                _log(f"HUD update took {elapsed:.1f}s (script: {script[:60]!r})")

    def _ui_eval(self, script: str):
        """Fire-and-forget HUD update. Returns immediately.

        Nothing in the voice pipeline reads a value back from the HUD, so
        no caller has any reason to wait on the webview. Backlog is
        bounded: if the bridge thread is wedged inside evaluate_js, updates
        are dropped and the stall is reported once, rather than queued
        without limit.
        """
        if not self.ui_window:
            return
        try:
            self._ui_queue.put_nowait(script)
            # Only call it recovered once the backlog has genuinely drained.
            # A single freed slot means the bridge dequeued one item and is
            # now wedged on that one instead -- reporting recovery there
            # would flap a stall line on every update.
            if self._ui_stalled and self._ui_queue.qsize() <= UI_QUEUE_LIMIT // 4:
                self._ui_stalled = False
                _log("HUD bridge recovered.")
        except queue.Full:
            if not self._ui_stalled:
                self._ui_stalled = True
                _log(f"HUD BRIDGE STALLED — {UI_QUEUE_LIMIT} updates queued and "
                     f"not draining; dropping further HUD updates. The webview "
                     f"is not answering evaluate_js.")
                vision_logging.dump_all_stacks("HUD bridge stalled")

    async def _session_send(self, what: str, coro_factory):
        """Every write to the Gemini Live websocket goes through here.

        Two unbounded waits used to live on this path and either one could
        wedge VISION permanently:

          * acquiring _session_send_lock, which is held for the whole
            duration of another task's send; and
          * the send itself -- a websocket write on a half-open TCP
            connection (sleep/wake, Wi-Fi handoff) never returns and never
            raises, because no FIN or RST is ever received.

        If a send stalls while holding the lock, _send_realtime,
        _live_notification_watcher and the tool-response path in
        _receive_audio all block on it forever. No task ever completes, so
        the asyncio.wait(FIRST_COMPLETED) in _run_session never returns and
        the reconnect never fires: the app stays running but is completely
        deaf and mute. Bounding both waits turns that permanent hang into a
        raised exception, which _run_session already handles by reconnecting.
        """
        try:
            await asyncio.wait_for(
                self._session_send_lock.acquire(), timeout=SESSION_SEND_LOCK_TIMEOUT
            )
        except asyncio.TimeoutError:
            _log(f"SESSION SEND LOCK TIMEOUT after {SESSION_SEND_LOCK_TIMEOUT}s "
                 f"waiting to send {what} — another send is wedged.")
            vision_logging.dump_all_stacks(f"session send lock timeout ({what})")
            raise
        try:
            await asyncio.wait_for(coro_factory(), timeout=SESSION_SEND_TIMEOUT)
        except asyncio.TimeoutError:
            _log(f"SESSION SEND TIMEOUT after {SESSION_SEND_TIMEOUT}s sending "
                 f"{what} — treating the Live connection as dead.")
            raise
        finally:
            self._session_send_lock.release()

    def _hearing_state(self) -> str:
        """One-line answer to 'could VISION have heard me just now?'"""
        idle_age = (time.time() - self._last_active_time) if self._last_active_time else None
        capture_age = (time.time() - self._last_capture_at) if self._last_capture_at else None
        return (
            f"active={self._active} muted={self._muted} auto_muted={self._muted_auto} "
            f"speaking={self._is_speaking} "
            f"idle_age={'n/a' if idle_age is None else f'{idle_age:.1f}s'} "
            f"last_capture={'never' if capture_age is None else f'{capture_age:.1f}s ago'} "
            f"wake_buffer={len(self._auto_mute_audio_buffer)}B"
        )

    def _set_ui_status(self, status: str):
        if self.ui_window:
            try:
                self._ui_eval(f"window.updateStatus && window.updateStatus('{status}')")
            except Exception:
                pass

    def _set_ui_transcript(self, role: str, text: str, final: bool = False):
        """Streams a turn into the HUD transcript. Called repeatedly with the
        growing partial while someone is still talking, then once more with
        final=True to close the bubble off."""
        if self.ui_window and text:
            try:
                self._ui_eval(
                    f"window.updateTranscript && window.updateTranscript("
                    f"{json.dumps(role)}, {json.dumps(text)}, {'true' if final else 'false'})"
                )
            except Exception:
                pass

    def _set_ui_levels(self, user_level: float, agent_level: float):
        """Both audio levels in a single evaluate_js — this runs ~15x/sec, so
        it stays one round-trip per frame."""
        if self.ui_window:
            try:
                self._ui_eval(
                    f"window.setAudioLevel && (window.setAudioLevel('user', {user_level:.3f}),"
                    f" window.setAudioLevel('agent', {agent_level:.3f}))"
                )
            except Exception:
                pass

    def _set_ui_tool(self, name: str, active: bool):
        if self.ui_window:
            try:
                self._ui_eval(
                    f"window.setToolActivity && window.setToolActivity("
                    f"{json.dumps(name)}, {'true' if active else 'false'})"
                )
            except Exception:
                pass

    def _set_ui_code(self, code: str, language: str = ""):
        if self.ui_window:
            safe_code = json.dumps(code)
            safe_lang = json.dumps(language)
            try:
                self._ui_eval(f"window.showCode && window.showCode({safe_code}, {safe_lang})")
            except Exception:
                pass

    def _set_ui_search_results(self, query: str, results: list, images: list = None):
        if self.ui_window:
            safe_query = json.dumps(query)
            safe_results = json.dumps(results)
            safe_images = json.dumps(images or [])
            try:
                self._ui_eval(
                    f"window.showSearchResults && window.showSearchResults({safe_query}, {safe_results}, {safe_images})"
                )
            except Exception:
                pass

    def _set_ui_execution_result(self, code: str, stdout: str, stderr: str, success: bool):
        if self.ui_window:
            safe_code = json.dumps(code)
            safe_stdout = json.dumps(stdout or "")
            safe_stderr = json.dumps(stderr or "")
            try:
                self._ui_eval(
                    f"window.showExecutionResult && window.showExecutionResult"
                    f"({safe_code}, {safe_stdout}, {safe_stderr}, {'true' if success else 'false'})"
                )
            except Exception:
                pass

    def _render_ui_webpage(self, html: str):
        if self.ui_window:
            safe_html = json.dumps(html)
            try:
                self._ui_eval(f"window.renderWebpage && window.renderWebpage({safe_html})")
            except Exception:
                pass

    def _close_ui_visual_panel(self):
        if self.ui_window:
            try:
                self._ui_eval("window.closeVisualPanel && window.closeVisualPanel()")
            except Exception:
                pass

    def _set_ui_screen_active(self, active: bool):
        if self.ui_window:
            try:
                self._ui_eval(f"window.setScreenActive && window.setScreenActive({'true' if active else 'false'})")
            except Exception:
                pass

    def _set_ui_camera_active(self, active: bool):
        if self.ui_window:
            try:
                self._ui_eval(f"window.setCameraActive && window.setCameraActive({'true' if active else 'false'})")
            except Exception:
                pass

    def set_muted(self, muted: bool):
        """Manual mute (explicit 'mute'/unmute button) -- fully independent of
        auto-mute. Engaging it supersedes and clears any active auto-mute;
        disengaging it resets the idle clock so it doesn't immediately
        re-auto-mute using a stale _last_active_time from before the mute."""
        self._muted = muted
        if muted:
            self._muted_auto = False
            self._auto_mute_audio_buffer.clear()
        else:
            self._last_active_time = time.time()
        _log(f"MANUAL {'MUTE' if muted else 'UNMUTE'} (HUD button) — {self._hearing_state()}")
        state = "muted" if muted else ("listening" if self._active else "idle")
        self._set_ui_status(state)

    def _play_system_sound(self, name: str):
        """Short native macOS system sounds for auto-mute events -- deliberately
        NOT the synthesized acoustic_cues tones, per an explicit ask for simple
        built-in system sounds here."""
        sound_files = {
            "auto_mute": "/System/Library/Sounds/Tink.aiff",
            "auto_unmute": "/System/Library/Sounds/Glass.aiff",
        }
        path = sound_files.get(name)
        if not path:
            return
        try:
            subprocess.run(["afplay", path], check=False, timeout=5)
        except Exception as e:
            print(f"[vision_live] Failed to play system sound '{name}': {e}")

    def _engage_auto_mute(self):
        self._muted_auto = True
        _log(f"AUTO-MUTE ENGAGED after {AUTO_MUTE_IDLE_SECONDS}s idle (connection stays open). "
             f"Only the local wake-word check can lift this — {self._hearing_state()}")
        self._set_ui_status("muted")
        asyncio.create_task(asyncio.to_thread(self._play_system_sound, "auto_mute"))

    def _disengage_auto_mute(self):
        self._muted_auto = False
        self._last_active_time = time.time()
        _log(f"AUTO-MUTE LIFTED — wake phrase heard while auto-muted. {self._hearing_state()}")
        self._set_ui_status("listening")
        asyncio.create_task(asyncio.to_thread(self._play_system_sound, "auto_unmute"))

    def _write_wav_chunk(self, chunk_bytes: bytes) -> str:
        """Writes raw int16 PCM (matching SEND_SAMPLE_RATE/CHANNELS, same
        format voice/stt.py's record_audio produces) to a temp WAV file so
        the existing transcribe() can be reused as-is."""
        temp_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        with wave.open(temp_path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SEND_SAMPLE_RATE)
            wf.writeframes(chunk_bytes)
        return temp_path

    def _build_config(self, headphones_mode: bool):
        # There is no real acoustic echo cancellation anywhere in this
        # pipeline -- the only existing mechanism (see _listen_audio_awake)
        # is gating the mic while self._is_speaking is True on speaker mode,
        # which only accounts for VISION'S OWN generated speech. It does
        # nothing about a separate app (Spotify, a browser, etc.) playing
        # through the same speakers -- that audio is picked up by the mic
        # and sent to Gemini like any other input, and Gemini's server-side
        # VAD has no way to know it isn't the user talking. HIGH/HIGH
        # sensitivity (tuned for snappy reaction) makes this worse: it
        # triggers on quieter/more ambiguous audio, including music/singing.
        # The SDK only exposes a binary HIGH/LOW choice here, no separate
        # music-vs-speech classifier or continuous confidence threshold.
        # Real proper AEC would mean replacing sounddevice/PortAudio's mic
        # capture with AVAudioEngine's voice-processing input node (Apple's
        # AUVoiceProcessingIO, which cancels against whatever the system is
        # currently outputting, not just this app's own audio) -- a real,
        # meaningfully larger change not made here.
        # As a real, available mitigation: headphone audio physically can't
        # leak into the mic, so keep the snappy HIGH/HIGH sensitivity there;
        # on speakers, where leakage is real, use LOW/LOW so ambient/media
        # audio needs a much clearer, unambiguous speech signal to register
        # as speech at all -- which also directly raises the bar for a
        # false barge-in/interruption, since the same sensitivity setting
        # governs both turn-start and mid-response interruption detection.
        sensitivity = (
            (types.StartSensitivity.START_SENSITIVITY_HIGH, types.EndSensitivity.END_SENSITIVITY_HIGH)
            if headphones_mode
            else (types.StartSensitivity.START_SENSITIVITY_LOW, types.EndSensitivity.END_SENSITIVITY_LOW)
        )
        kwargs = dict(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
            tools=[{"function_declarations": TOOL_DECLARATIONS}],
            input_audio_transcription={},
            output_audio_transcription={},
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    start_of_speech_sensitivity=sensitivity[0],
                    end_of_speech_sensitivity=sensitivity[1],
                    prefix_padding_ms=100,
                    silence_duration_ms=500,
                ),
            ),
        )
        try:
            return types.LiveConnectConfig(enable_affective_dialog=True, **kwargs)
        except Exception as e:
            print(f"[vision_live] Affective dialog not supported, continuing without it: {e}")
            return types.LiveConnectConfig(**kwargs)

    async def _capture_and_queue_screen_frame(self, zoom_region=None):
        """Grabs one screen frame and sends it to the model via the realtime
        video stream. Also records its pixel size (and crop region, if any)
        so computer_control can translate the model's click coordinates into
        real screen points. Shared by the ambient _screen_share_loop and
        computer_control's post-action "look" step.

        If zoom_region is given (logical screen rect from
        computer_control.compute_zoom_region), captures just that region
        instead of the full screen and upscales it for legibility — lets the
        model get a magnified close-up of a small target (e.g. a video
        player's Skip Ad button) instead of guessing from a full desktop
        screenshot."""
        with mss.mss() as sct:
            if zoom_region:
                region = {
                    "left": int(zoom_region["region_x"]), "top": int(zoom_region["region_y"]),
                    "width": int(zoom_region["region_w"]), "height": int(zoom_region["region_h"]),
                }
                screenshot = sct.grab(region)
            else:
                screenshot = sct.grab(sct.monitors[1])
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

        if zoom_region:
            scale = min(4, 1568 / max(img.width, img.height))
            if scale > 1:
                img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        else:
            img.thumbnail((1568, 1568))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        if self.out_queue:
            await self.out_queue.put({"data": buf.getvalue(), "mime_type": "image/jpeg"})

        if zoom_region:
            region_tuple = (zoom_region["region_x"], zoom_region["region_y"], zoom_region["region_w"], zoom_region["region_h"])
            computer_control.record_capture_size(img.width, img.height, region=region_tuple)
        else:
            computer_control.record_capture_size(img.width, img.height)

    async def _screen_share_loop(self):
        print("[vision_live] Screen sharing started.")
        try:
            while self._screen_active:
                await self._capture_and_queue_screen_frame()
                await asyncio.sleep(SCREEN_CAPTURE_INTERVAL)
        except Exception as e:
            print(f"[vision_live] Screen capture error: {e}")
        finally:
            print("[vision_live] Screen sharing stopped.")

    def _start_screen_share(self):
        if self._screen_active:
            return
        self._screen_active = True
        self._set_ui_screen_active(True)
        self._screen_task = asyncio.create_task(self._screen_share_loop())

    def _stop_screen_share(self):
        self._screen_active = False
        self._set_ui_screen_active(False)
        if self._screen_task:
            self._screen_task.cancel()
            self._screen_task = None

    async def _camera_loop(self):
        print("[vision_live] Camera started.")
        cap = None
        try:
            # Opening the capture device can block indefinitely if another
            # process holds the camera or the driver wedges.
            cap = await asyncio.wait_for(
                asyncio.to_thread(cv2.VideoCapture, 0), timeout=15
            )
            if not cap.isOpened():
                print("[vision_live] Could not open camera.")
                return
            while self._camera_active:
                ret, frame = await asyncio.wait_for(
                    asyncio.to_thread(cap.read), timeout=10
                )
                if not ret:
                    await asyncio.sleep(0.5)
                    continue
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                img.thumbnail((1024, 1024))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                if self.out_queue:
                    await self.out_queue.put({"data": buf.getvalue(), "mime_type": "image/jpeg"})
                await asyncio.sleep(CAMERA_CAPTURE_INTERVAL)
        except Exception as e:
            print(f"[vision_live] Camera error: {e}")
        finally:
            if cap:
                await asyncio.to_thread(cap.release)
            print("[vision_live] Camera stopped.")

    def _start_camera(self):
        if self._camera_active:
            return
        self._camera_active = True
        self._set_ui_camera_active(True)
        self._camera_task = asyncio.create_task(self._camera_loop())

    def _stop_camera(self):
        self._camera_active = False
        self._set_ui_camera_active(False)
        if self._camera_task:
            self._camera_task.cancel()
            self._camera_task = None

    async def _execute_tool(self, fc):
        name = fc.name
        args = dict(fc.args) if fc.args else {}

        if name == "shutdown_vision":
            print("[vision_live] Shutdown requested by user.")
            self._stop_screen_share()
            self._stop_camera()
            if self._shutdown_event:
                self._shutdown_event.set()
            return types.FunctionResponse(id=fc.id, name=name, response={"result": {"success": True}})

        if name == "show_code":
            self._set_ui_code(args.get("code", ""), args.get("language", ""))
            return types.FunctionResponse(id=fc.id, name=name, response={"result": {"success": True}})

        if name == "render_webpage":
            self._render_ui_webpage(args.get("html", ""))
            return types.FunctionResponse(id=fc.id, name=name, response={"result": {"success": True}})

        if name == "close_visual_panel":
            self._close_ui_visual_panel()
            return types.FunctionResponse(id=fc.id, name=name, response={"result": {"success": True}})

        if name == "view_screen":
            self._start_screen_share()
            return types.FunctionResponse(id=fc.id, name=name, response={"result": {"success": True}})

        if name == "stop_viewing_screen":
            self._stop_screen_share()
            return types.FunctionResponse(id=fc.id, name=name, response={"result": {"success": True}})

        if name == "view_camera":
            self._start_camera()
            return types.FunctionResponse(id=fc.id, name=name, response={"result": {"success": True}})

        if name == "stop_camera":
            self._stop_camera()
            return types.FunctionResponse(id=fc.id, name=name, response={"result": {"success": True}})

        if name == "remember_fact":
            save_fact(args.get("key", ""), args.get("value", ""))
            return types.FunctionResponse(id=fc.id, name=name, response={"result": {"success": True}})

        if name == "forget_fact":
            delete_fact(args.get("key", ""))
            return types.FunctionResponse(id=fc.id, name=name, response={"result": {"success": True}})

        if name == "recall_memory":
            # Embedding + vector query: off the loop like every other
            # blocking tool call.
            results = await asyncio.to_thread(
                semantic_search, args.get("query", ""), 5
            )
            return types.FunctionResponse(id=fc.id, name=name, response={"result": {"matches": results}})

        confirmation_token = args.pop("confirmation_token", None)
        print(f"[vision_live] Tool call: {name}({args})")

        if name == "execute_python" and confirmation_token:
            # Only on the confirmed call that actually runs code — the first
            # call (no token yet) just mints a confirmation request, which is
            # near-instant and has nothing "slow" to cover with this cue.
            asyncio.create_task(asyncio.to_thread(play_acoustic_cue, "waiting"))

        try:
            # NOTE: possessing a valid confirmation_token only proves this exact
            # pending action was offered by dispatch() moments ago — it does NOT
            # prove the user actually said "yes." That verification still depends
            # on the model faithfully following SYSTEM_PROMPT's confirm-before-
            # reuse instruction; the token closes off guessing/replay/naive
            # injection, not model misbehavior.
            # Threaded so a slow tool (AppleScript, execute_python, computer_control)
            # doesn't freeze the event loop — audio keeps flowing while it runs.
            if confirmation_token:
                result = await asyncio.to_thread(dispatcher.confirm, confirmation_token, approved=True)
            else:
                result = await asyncio.to_thread(dispatch, name, **args)
        except Exception as e:
            result = {"success": False, "error": str(e)}
            traceback.print_exc()
        print(f"[vision_live] Tool result: {result}")

        # Excludes computer_control/start_computer_use/end_computer_use: those
        # fire many times per GUI-automation task (or represent a state
        # transition, not a discrete checkmark-able result) — chiming on
        # every click/keystroke would be noise, not a helpful acknowledgment.
        if name not in ("computer_control", "start_computer_use", "end_computer_use") and isinstance(result, dict):
            if result.get("success") is True:
                asyncio.create_task(asyncio.to_thread(play_acoustic_cue, "success"))
            elif result.get("success") is False:
                asyncio.create_task(asyncio.to_thread(play_acoustic_cue, "error"))

        if name == "web_search" and result.get("success"):
            self._set_ui_search_results(args.get("query", ""), result.get("results", []), result.get("images", []))

        if name == "execute_python" and result.get("status") != "confirmation_required":
            self._set_ui_execution_result(
                args.get("code", ""),
                result.get("stdout"),
                result.get("stderr"),
                result.get("success", False),
            )

        if name == "computer_control":
            # Look-act-look: give the model a fresh view after every action,
            # regardless of whether the action itself succeeded, so it can
            # see what actually happened (or why it didn't). A short settle
            # delay first lets UI transitions/animations finish rather than
            # capturing a mid-transition frame. If this was a zoom_screenshot
            # request, capture that cropped region instead of the full screen.
            await asyncio.sleep(0.5)
            zoom_region = result.get("zoom_region") if isinstance(result, dict) else None
            await self._capture_and_queue_screen_frame(zoom_region=zoom_region)

        if name == "start_computer_use" and result.get("status") != "confirmation_required" and result.get("success"):
            # Give the model an immediate first look once a task is confirmed,
            # rather than making it call take_screenshot separately. Longer
            # settle delay since the target app may have just been opened
            # and could still be rendering its window.
            await asyncio.sleep(1.0)
            await self._capture_and_queue_screen_frame()

        return types.FunctionResponse(id=fc.id, name=name, response={"result": result})

    async def _check_mic_available(self) -> bool:
        try:
            # Bounded: opening a CoreAudio stream can block indefinitely
            # after a sleep/wake glitch, and this runs on every wake-word
            # cycle -- a hang here makes VISION permanently deaf.
            test_stream = await asyncio.wait_for(
                asyncio.to_thread(
                    sd.InputStream, samplerate=SEND_SAMPLE_RATE,
                    channels=CHANNELS, dtype="int16",
                ),
                timeout=10,
            )
            await asyncio.wait_for(asyncio.to_thread(test_stream.close), timeout=10)
            return True
        except asyncio.TimeoutError:
            _log("MIC CHECK TIMED OUT — CoreAudio did not answer within 10s.")
            return False
        except Exception as e:
            print(f"[vision_live] Mic unavailable: {e}")
            return False

    async def _auto_mute_idle_watcher(self):
        """Persistent background task (like _proactive_monitor_loop) --
        naturally no-ops while dormant/manually-muted/already-auto-muted/
        VISION-speaking via the guard below, so it doesn't need to be torn
        down and recreated on every reconnect the way the per-connection
        audio tasks do."""
        while True:
            await asyncio.sleep(1)

            # A live session whose mic callback has gone quiet is the one
            # condition that both triggers auto-mute AND starves the
            # wake-word check that is supposed to undo it.
            if self._active and not self._muted and self._last_capture_at:
                capture_age = time.time() - self._last_capture_at
                if capture_age > 5 and not self._capture_stall_logged:
                    self._capture_stall_logged = True
                    _log(f"WARNING: no mic callback for {capture_age:.1f}s — the capture stream "
                         f"looks stalled; VISION cannot hear anything in this state. "
                         f"{self._hearing_state()}")
                elif capture_age <= 5 and self._capture_stall_logged:
                    self._capture_stall_logged = False
                    _log(f"mic callback resumed after a stall. {self._hearing_state()}")

            if (
                self._active
                and not self._muted
                and not self._muted_auto
                and not self._is_speaking
                and (time.time() - self._last_active_time > AUTO_MUTE_IDLE_SECONDS)
            ):
                self._engage_auto_mute()

    async def _auto_mute_wake_word_watcher(self):
        """While auto-muted, periodically transcribes whatever's been
        buffered locally (see the callback in _listen_audio_awake) and
        checks for the wake phrase, exactly like the dormant wake-word loop
        but operating on audio from the already-open awake-session stream
        instead of a separate recording."""
        while True:
            await asyncio.sleep(WAKE_CHUNK_DURATION)

            if not self._muted_auto:
                self._auto_mute_audio_buffer.clear()
                continue

            chunk = bytes(self._auto_mute_audio_buffer)
            self._auto_mute_audio_buffer.clear()
            if not chunk:
                _log(f"wake-check (auto-muted): NO AUDIO BUFFERED in the last "
                     f"{WAKE_CHUNK_DURATION}s — nothing is reaching the wake-word check, so "
                     f"saying '{config.WAKE_WORD}' cannot lift auto-mute. {self._hearing_state()}")
                continue

            try:
                audio_path = await asyncio.to_thread(self._write_wav_chunk, chunk)
                level = await asyncio.to_thread(_wav_level, audio_path)
                text = await asyncio.to_thread(transcribe, audio_path)
            except Exception as e:
                _log(f"wake-check (auto-muted) FAILED: {e!r} — {self._hearing_state()}")
                continue

            hit = bool(text and WAKE_PHRASE in text.lower())
            _log(f"wake-check (auto-muted): {len(chunk)/2/SEND_SAMPLE_RATE:.1f}s buffered, "
                 f"level={level:.3f}, heard={text!r} -> {'WAKE' if hit else 'no match'}")
            if hit:
                self._disengage_auto_mute()

    async def _proactive_monitor_loop(self):
        while True:
            await asyncio.sleep(PROACTIVE_CHECK_INTERVAL)

            if self._active:
                continue

            try:
                # osascript subprocess: blocks for as long as Calendar takes
                # to answer, which must not be on the event loop.
                result = await asyncio.wait_for(
                    asyncio.to_thread(get_calendar_events, 1), timeout=30
                )
            except asyncio.TimeoutError:
                print("[vision_live] Proactive calendar check timed out")
                continue
            except Exception as e:
                print(f"[vision_live] Proactive check failed: {e}")
                continue

            if not result.get("success"):
                continue

            for event in result.get("events", []):
                event_id = f"{event['title']}-{event['start']}"
                if event_id in self._notified_events:
                    continue
                self._notified_events.add(event_id)
                self._proactive_message = (
                    f"You have an upcoming event: {event['title']} at {event['start']}."
                )
                print(f"[vision_live] Proactive trigger: {self._proactive_message}")
                self._proactive_trigger.set()
                break

    async def _wait_for_wake_phrase(self):
        _log(f"DORMANT — say '{config.WAKE_WORD}' to activate. {self._hearing_state()}")
        self._set_ui_status("muted" if self._muted else "idle")

        await asyncio.sleep(0.5)

        last_chunk_end = 0.0
        muted_notice = [False]

        async def listen_loop():
            nonlocal last_chunk_end
            while True:
                if self._muted:
                    # Manually muted while dormant: nothing is being recorded at
                    # all, so the wake word cannot work until it is unmuted.
                    if not muted_notice[0]:
                        muted_notice[0] = True
                        _log(f"dormant but MANUALLY MUTED — wake word is disabled until you "
                             f"unmute in the HUD. {self._hearing_state()}")
                    self._set_ui_status("muted")
                    await asyncio.sleep(0.5)
                    continue
                if muted_notice[0]:
                    muted_notice[0] = False
                    _log(f"unmuted — wake-word listening resumes. {self._hearing_state()}")

                if not await self._check_mic_available():
                    _log(f"wake-check #{self._wake_attempts + 1} SKIPPED: mic unavailable, "
                         f"retrying in 3s (deaf until then). {self._hearing_state()}")
                    self._set_ui_status("mic_unavailable")
                    await asyncio.sleep(3)
                    continue
                try:
                    self._wake_attempts += 1
                    gap = (time.time() - last_chunk_end) if last_chunk_end else 0.0
                    started = time.time()
                    audio_path = await asyncio.to_thread(record_audio, WAKE_CHUNK_DURATION)
                    last_chunk_end = time.time()
                    level = await asyncio.to_thread(_wav_level, audio_path)
                    text = await asyncio.to_thread(transcribe, audio_path)
                except Exception as e:
                    _log(f"wake-check #{self._wake_attempts} FAILED: {e!r} — "
                         f"retrying in 1s. {self._hearing_state()}")
                    self._set_ui_status("mic_unavailable")
                    await asyncio.sleep(1)
                    continue

                hit = bool(text and WAKE_PHRASE in text.lower())

                # Digital silence is not the same as a quiet room: a device that
                # has gone away often keeps handing back zeroes forever.
                if 0 <= level < 0.002:
                    self._silent_chunks += 1
                    if self._silent_chunks in (3, 10) or self._silent_chunks % 25 == 0:
                        _log(f"WARNING: {self._silent_chunks} consecutive near-silent chunks "
                             f"(level={level:.4f}) — the mic may be handing back digital silence. "
                             f"{self._hearing_state()}")
                else:
                    self._silent_chunks = 0

                _log(f"wake-check #{self._wake_attempts}: mic closed {gap*1000:.0f}ms before this "
                     f"chunk, captured {time.time()-started:.1f}s, level={level:.3f}, "
                     f"heard={text!r} -> {'WAKE' if hit else 'no match'}")

                if hit:
                    _log(f"WAKE PHRASE accepted — starting session. {self._hearing_state()}")
                    return "wake_phrase"

        listen_task = asyncio.create_task(listen_loop())
        proactive_task = asyncio.create_task(self._proactive_trigger.wait())

        done, pending = await asyncio.wait(
            [listen_task, proactive_task], return_when=asyncio.FIRST_COMPLETED
        )

        for t in pending:
            t.cancel()

        if listen_task in done and not listen_task.cancelled():
            exc = listen_task.exception()
            if exc is not None:
                # Without this the crash is invisible: the function falls
                # through and reports a wake phrase nobody said.
                _log(f"CRITICAL: the dormant wake-word listener crashed with {exc!r}. "
                     f"VISION was not listening. {self._hearing_state()}")
                traceback.print_exception(type(exc), exc, exc.__traceback__)

        if proactive_task in done:
            self._proactive_trigger.clear()
            self._set_ui_status("idle")
            return "proactive"

        self._set_ui_status("idle")
        return "wake_phrase"

    async def _listen_audio_awake(self):
        loop = asyncio.get_event_loop()

        def callback(indata, frames, time_info, status):
            # Just store the level — evaluate_js must never be called from
            # the realtime audio thread. _ui_level_pump forwards it.
            self._mic_level = 0.0 if self._muted else _rms_level(indata)
            self._last_capture_at = time.time()

            if self._muted:
                # Manual mute takes full precedence: no forwarding, and no
                # local wake-word scanning either -- manual mute only lifts
                # on an explicit "unmute", never automatically.
                return

            if self._muted_auto:
                # Not forwarding to Gemini, but still buffer locally so the
                # wake-word watcher can notice the wake phrase and resume.
                self._auto_mute_audio_buffer.extend(indata.tobytes())
                return

            # Headphones: always send audio, letting Gemini's native VAD
            # handle real interruption (barge-in). Speakers: gate while
            # VISION is talking to avoid it hearing its own voice.
            if self._headphones_mode or not self._is_speaking:
                self._last_active_time = time.time()
                # out_queue is bounded (maxsize=10). A bare put_nowait
                # raises QueueFull *inside* the event-loop callback, where
                # it is only ever reported through asyncio's exception
                # handler -- which went nowhere before file logging existed.
                # A persistently full queue is the first visible symptom of
                # a wedged _send_realtime, so say so loudly and drop the
                # frame instead of raising.
                loop.call_soon_threadsafe(
                    self._offer_mic_chunk, indata.tobytes()
                )

        try:
            with sd.InputStream(
                samplerate=SEND_SAMPLE_RATE, channels=CHANNELS, dtype="int16",
                blocksize=CHUNK_SIZE, callback=callback,
            ):
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[vision_live] Mic unavailable during session: {e}")
            self._set_ui_status("mic_unavailable")
            raise

    def _offer_mic_chunk(self, pcm: bytes):
        """Queues a mic chunk for upload, dropping it if the uplink is
        backed up. Runs on the event loop via call_soon_threadsafe."""
        try:
            self.out_queue.put_nowait({"data": pcm, "mime_type": "audio/pcm"})
            if self._mic_drops:
                _log(f"Mic uplink recovered after dropping {self._mic_drops} chunks.")
                self._mic_drops = 0
        except asyncio.QueueFull:
            self._mic_drops += 1
            # Powers of two so a sustained stall keeps reporting without
            # writing a line for every 32ms chunk.
            if self._mic_drops & (self._mic_drops - 1) == 0:
                _log(f"MIC UPLINK BACKED UP — dropped {self._mic_drops} chunk(s); "
                     f"send_realtime is not draining out_queue. "
                     f"{self._hearing_state()}")
        except Exception as e:
            print(f"[vision_live] Mic chunk enqueue failed: {e}")

    async def _send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self._session_send(
                "realtime audio",
                lambda m=msg: self.session.send_realtime_input(media=m),
            )

    async def _finish_speaking(self, generation):
        while self.audio_in_queue and self.audio_in_queue.qsize() > 0:
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.6)
        if generation != self._speaking_generation:
            # A newer turn has already started speaking while this one was
            # winding down — don't stomp its "speaking" state back to False.
            return
        self._is_speaking = False
        self._last_active_time = time.time()
        self._set_ui_status("muted" if self._muted else "listening")

    async def _handle_interruption(self, out_buf=None):
        """Called when Gemini's server reports the user barged in mid-response."""
        print("[vision_live] User interrupted — stopping playback immediately.")
        # Drain any queued audio so playback stops right away
        while self.audio_in_queue and not self.audio_in_queue.empty():
            try:
                self.audio_in_queue.get_nowait()
            except Exception:
                break

        if out_buf:
            partial = " ".join(out_buf).strip()
            out_buf.clear()
            if partial:
                self._set_ui_transcript("agent", partial, final=True)
                print(f"VISION (interrupted): {partial}")
                save_message("assistant", f"[Interrupted by user] {partial}")

        self._is_speaking = False
        self._last_active_time = time.time()
        self._set_ui_status("listening")

    async def _maybe_signal_processing(self, chunk_time):
        """Debounced "user has likely finished talking" detector. There's no
        explicit SDK event for this — input_transcription streams in
        progressively while the user talks, and the only other signal we
        have (output_transcription starting) is already too late to show a
        "processing" state. So: wait a beat after each input chunk: if no
        newer chunk has superseded this one and the model hasn't started
        responding yet, assume they've stopped talking and are waiting on
        VISION. Every chunk during continuous speech reschedules this and
        the stale checks harmlessly no-op — only the one after the true
        last chunk actually fires."""
        await asyncio.sleep(0.35)
        if self._last_input_chunk_at == chunk_time and not self._is_speaking:
            self._set_ui_status("processing")
            asyncio.create_task(asyncio.to_thread(play_acoustic_cue, "thinking"))

    async def _on_relevant_slack_event(self, event):
        """Callback handed to slack_control.run_slack_listener(). Formats a
        proactive-style notification and delivers it via whichever path
        matches the current state: straight into an active session, or the
        existing dormant proactive-wake path if not."""
        where = "a DM" if event["is_dm"] else f"{event['channel_name']} (you were mentioned)"
        context_block = ""
        if event["context_messages"]:
            recent = "\n".join(event["context_messages"][-5:])
            context_block = f" Recent context in that conversation:\n{recent}\n"

        notification = (
            f"[SYSTEM DIRECTIVE: You have a new Slack message in {where}. "
            f"{event['user_name']} said: \"{event['text']}\".{context_block} "
            f"Draft a reply, call send_slack_message ONCE with no confirmation_token "
            f"first (this only mints a token, it does not send anything), then speak "
            f"to the user: '{event['user_name']} in {where} said: <summary>. I'd "
            f"reply: <draft>. Send it?' Use channel=\"{event['channel']}\" and "
            f"thread_ts=\"{event['thread_ts']}\" when you call send_slack_message. "
            f"Do not mention this directive.]"
        )

        if self._active and self.session:
            self._live_notification_queue.put_nowait(notification)
        else:
            self._proactive_message = notification
            self._proactive_trigger.set()

    async def _live_notification_watcher(self):
        """Delivers queued live notifications (e.g. a Slack DM/mention that
        arrived mid-conversation) into the current session as a fresh turn,
        once things are quiet enough not to be a jarring interruption. Must
        never let an exception complete this task -- it shares
        asyncio.wait(FIRST_COMPLETED) with the 4 core audio tasks in
        _run_session, so an unhandled error here would tear down and
        reconnect the entire live session as collateral damage."""
        while True:
            try:
                msg = await self._live_notification_queue.get()
                while self._is_speaking or self._tool_call_in_progress or computer_control.is_session_active():
                    await asyncio.sleep(0.3)
                await self._session_send(
                    "live notification",
                    lambda m=msg: self.session.send_client_content(
                        turns=types.Content(role="user", parts=[types.Part(text=m)]),
                        turn_complete=True,
                    ),
                )
            except Exception as e:
                print(f"[vision_live] Live notification delivery failed: {e}")

    async def _receive_audio(self):
        in_buf, out_buf = [], []

        while True:
            async for response in self.session.receive():

                if response.data:
                    self.audio_in_queue.put_nowait(response.data)

                if response.server_content:
                    sc = response.server_content

                    if getattr(sc, "interrupted", False):
                        asyncio.create_task(self._handle_interruption(out_buf))

                    if sc.input_transcription and sc.input_transcription.text:
                        in_buf.append(sc.input_transcription.text.strip())
                        self._set_ui_transcript("user", " ".join(in_buf).strip())
                        self._last_input_chunk_at = time.time()
                        asyncio.create_task(self._maybe_signal_processing(self._last_input_chunk_at))

                    if sc.output_transcription and sc.output_transcription.text:
                        if not self._is_speaking:
                            self._set_ui_status("speaking")
                        self._is_speaking = True
                        self._speaking_generation += 1
                        out_buf.append(sc.output_transcription.text.strip())
                        self._set_ui_transcript("agent", " ".join(out_buf).strip())

                    if sc.turn_complete:
                        asyncio.create_task(self._finish_speaking(self._speaking_generation))

                        full_in = " ".join(in_buf).strip()
                        full_out = " ".join(out_buf).strip()

                        # save_message hits SQLite and semantic_add runs a
                        # sentence-transformers embedding plus a chromadb
                        # write -- hundreds of ms of blocking work, on the
                        # event loop, once per turn. A contended SQLite
                        # lock here stalls audio in both directions.
                        if full_in:
                            self._set_ui_transcript("user", full_in, final=True)
                            print(f"You: {full_in}")
                            asyncio.create_task(
                                asyncio.to_thread(self._persist_turn, "user", full_in)
                            )
                        if full_out:
                            print(f"VISION: {full_out}")
                            asyncio.create_task(
                                asyncio.to_thread(self._persist_turn, "assistant", full_out)
                            )
                            self._set_ui_transcript("agent", full_out, final=True)

                        in_buf, out_buf = [], []

                if response.tool_call:
                    self._tool_call_in_progress = True
                    try:
                        fn_responses = []
                        for fc in response.tool_call.function_calls:
                            self._set_ui_tool(fc.name, True)
                            try:
                                fn_responses.append(await self._execute_tool(fc))
                            finally:
                                self._set_ui_tool(fc.name, False)
                    finally:
                        self._tool_call_in_progress = False
                    await self._session_send(
                        "tool response",
                        lambda r=fn_responses: self.session.send_tool_response(
                            function_responses=r
                        ),
                    )

    def _persist_turn(self, role: str, text: str):
        """Blocking history write, always called via asyncio.to_thread."""
        try:
            save_message(role, text)
        except Exception as e:
            print(f"[vision_live] save_message failed: {e}")
        try:
            semantic_add(role, text)
        except Exception as e:
            print(f"[vision_live] semantic_add failed: {e}")

    async def _play_audio(self):
        stream = sd.RawOutputStream(
            samplerate=RECEIVE_SAMPLE_RATE, channels=CHANNELS, dtype="int16", blocksize=CHUNK_SIZE,
        )
        stream.start()
        try:
            while True:
                chunk = await self.audio_in_queue.get()
                self._agent_level = _rms_level(chunk)
                # RawOutputStream.write blocks until the device drains the
                # buffer. If the output device goes away mid-session
                # (headphones yanked, Bluetooth drop) that can block for
                # good, permanently consuming a thread-pool worker and
                # silently ending playback. Bound it so the task raises and
                # _run_session reopens the stream on reconnect.
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(stream.write, chunk),
                        timeout=AUDIO_WRITE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    _log(f"AUDIO OUTPUT STALLED — write blocked for "
                         f"{AUDIO_WRITE_TIMEOUT}s. Output device likely gone.")
                    raise
        finally:
            stream.stop()
            stream.close()

    async def _ui_level_pump(self):
        """Forwards mic/output audio levels to the HUD visualizer at ~15fps.
        Levels decay between pushes so the bars fall back to rest on their own
        once a stream stops producing chunks, and silence isn't re-sent."""
        was_quiet = False
        while True:
            await asyncio.sleep(1 / 15)
            if not self.ui_window:
                return

            user, agent = self._mic_level, self._agent_level
            quiet = user < 0.005 and agent < 0.005
            if not (quiet and was_quiet):
                # Direct call: _set_ui_levels only enqueues now, so the
                # extra thread hop this used to need is pure overhead at
                # 15 dispatches a second.
                self._set_ui_levels(user, agent)
            was_quiet = quiet

            self._mic_level *= 0.5
            self._agent_level *= 0.5

    async def _run_session(self, proactive_message: str = None):
        """Stays active for as long as the process is online. A crash or
        dropped connection in any of the core tasks reconnects automatically
        (showing "reconnecting") rather than falling back to dormant/idle —
        only an explicit shutdown_vision call (self._shutdown_event) ends
        the active session and returns control to the wake-word loop."""
        self._shutdown_event = asyncio.Event()
        self._active = True
        self._last_active_time = time.time()
        self._muted_auto = False
        self._auto_mute_audio_buffer.clear()
        greeted = False

        while True:
            try:
                client = genai.Client(api_key=config.GEMINI_API_KEY)
                self._headphones_mode = is_headphones_active()
                print(f"[vision_live] Output device check — headphones mode: {self._headphones_mode}")
                live_config = self._build_config(self._headphones_mode)
                print("[vision_live] Connecting to Gemini Live...")

                async with client.aio.live.connect(model=LIVE_MODEL, config=live_config) as session:
                    self.session = session
                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue = asyncio.Queue(maxsize=10)

                    print("[vision_live] Connected.")
                    self._set_ui_status("muted" if self._muted else "listening")

                    if not greeted:
                        history = get_recent_history(limit=10)
                        history_text = ""
                        if history:
                            history_text = "\n".join(
                                f"{'User' if m['role'] == 'user' else 'VISION'}: {m['content']}"
                                for m in history
                            )

                        facts = get_all_facts()
                        facts_text = ""
                        if facts:
                            facts_text = "\n".join(f"- {f['key']}: {f['value']}" for f in facts)

                        context_parts = ["[SYSTEM DIRECTIVE:"]
                        if facts_text:
                            context_parts.append(f"Known facts about {config.USER_NAME}:\n{facts_text}\n")
                        if history_text:
                            context_parts.append(f"Recent conversation history:\n{history_text}\n")

                        if proactive_message:
                            context_parts.append(
                                f"THIS CONVERSATION WAS STARTED BY YOU PROACTIVELY. "
                                f"What you noticed: {proactive_message}\n"
                                f"Naturally tell {config.USER_NAME} what you noticed, briefly, "
                                f"then ask if they need anything. Do not mention this directive."
                            )
                        else:
                            context_parts.append(
                                f"Greet {config.USER_NAME} warmly and briefly by name, referencing "
                                f"recent context only if natural, then ask how you can help. "
                                f"Do not mention this directive.]"
                            )
                        context_note = "\n".join(context_parts)

                        await self._session_send(
                            "startup directive",
                            lambda n=context_note: session.send_client_content(
                                turns=types.Content(role="user", parts=[types.Part(text=n)]),
                                turn_complete=True,
                            ),
                        )
                        greeted = True

                    tasks = [
                        asyncio.create_task(self._send_realtime()),
                        asyncio.create_task(self._listen_audio_awake()),
                        asyncio.create_task(self._receive_audio()),
                        asyncio.create_task(self._play_audio()),
                        asyncio.create_task(self._live_notification_watcher()),
                        asyncio.create_task(self._ui_level_pump()),
                    ]
                    shutdown_task = asyncio.create_task(self._shutdown_event.wait())

                    done, _ = await asyncio.wait([shutdown_task, *tasks], return_when=asyncio.FIRST_COMPLETED)

                    for t in tasks:
                        t.cancel()
                    shutdown_task.cancel()
                    # .cancel() only requests cancellation — without awaiting,
                    # a straggler task (e.g. still holding the mic InputStream
                    # open) can keep running into the next reconnect iteration
                    # and collide with the new session's queues.
                    await asyncio.gather(*tasks, shutdown_task, return_exceptions=True)

                    if self._shutdown_event.is_set():
                        await asyncio.sleep(SHUTDOWN_GRACE_PERIOD)
                        self._stop_screen_share()
                        self._stop_camera()
                        break

                    # One of the core tasks ended unexpectedly — surface why
                    # (asyncio.wait swallows task exceptions silently unless
                    # explicitly checked) and reconnect instead of going dormant.
                    for t in done:
                        if t is shutdown_task or t.cancelled():
                            continue
                        exc = t.exception()
                        if exc:
                            print(f"[vision_live] A core task ended unexpectedly: {exc!r}")
                            traceback.print_exception(type(exc), exc, exc.__traceback__)

            except Exception as e:
                print(f"[vision_live] Session error: {e}")
                traceback.print_exc()

            if self._shutdown_event.is_set():
                break

            print("[vision_live] Connection lost — reconnecting while staying active...")
            self._set_ui_status("reconnecting")
            await asyncio.sleep(2)

        self._active = False
        self.session = None
        self._set_ui_status("muted" if self._muted else "idle")
        print("[vision_live] Session ended.\n")

    async def run(self):
        # asyncio reports some failures only through its exception handler
        # -- notably anything raised inside a call_soon_threadsafe callback
        # -- and by default those go to a stderr that does not exist in a
        # packaged .app.
        vision_logging.install_asyncio_exception_handler(asyncio.get_running_loop())
        # Feeds the freeze watchdog. If this stops ticking, the event loop
        # is blocked, and the watchdog dumps every thread's stack to the
        # log -- the only evidence a hang ever leaves behind.
        asyncio.create_task(vision_logging.heartbeat_loop())

        asyncio.create_task(self._proactive_monitor_loop())
        asyncio.create_task(slack_control.run_slack_listener(self._on_relevant_slack_event))
        asyncio.create_task(self._auto_mute_idle_watcher())
        asyncio.create_task(self._auto_mute_wake_word_watcher())

        while True:
            try:
                trigger = await self._wait_for_wake_phrase()
                if trigger == "proactive":
                    msg = self._proactive_message
                    self._proactive_message = None
                    await self._run_session(proactive_message=msg)
                else:
                    await self._run_session()
            except Exception as e:
                print(f"[vision_live] Error: {e}")
                traceback.print_exc()
                self._set_ui_status("reconnecting")
                await asyncio.sleep(2)


def main():
    vision_logging.setup("vision_live")
    init_db()
    live = VisionLive()
    try:
        asyncio.run(live.run())
    except KeyboardInterrupt:
        print("\n[vision_live] Shutting down.")


if __name__ == "__main__":
    main()