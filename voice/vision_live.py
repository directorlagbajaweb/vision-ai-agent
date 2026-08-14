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
import tempfile
import traceback
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import sounddevice as sd
import mss
import cv2
from PIL import Image
from google import genai
from google.genai import types

import config
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

    def _set_ui_status(self, status: str):
        if self.ui_window:
            try:
                self.ui_window.evaluate_js(f"window.updateStatus && window.updateStatus('{status}')")
            except Exception:
                pass

    def _set_ui_response(self, text: str):
        if self.ui_window:
            safe = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")
            try:
                self.ui_window.evaluate_js(f"window.updateResponse && window.updateResponse('{safe}')")
            except Exception:
                pass

    def _set_ui_code(self, code: str, language: str = ""):
        if self.ui_window:
            safe_code = json.dumps(code)
            safe_lang = json.dumps(language)
            try:
                self.ui_window.evaluate_js(f"window.showCode && window.showCode({safe_code}, {safe_lang})")
            except Exception:
                pass

    def _set_ui_search_results(self, query: str, results: list, images: list = None):
        if self.ui_window:
            safe_query = json.dumps(query)
            safe_results = json.dumps(results)
            safe_images = json.dumps(images or [])
            try:
                self.ui_window.evaluate_js(
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
                self.ui_window.evaluate_js(
                    f"window.showExecutionResult && window.showExecutionResult"
                    f"({safe_code}, {safe_stdout}, {safe_stderr}, {'true' if success else 'false'})"
                )
            except Exception:
                pass

    def _render_ui_webpage(self, html: str):
        if self.ui_window:
            safe_html = json.dumps(html)
            try:
                self.ui_window.evaluate_js(f"window.renderWebpage && window.renderWebpage({safe_html})")
            except Exception:
                pass

    def _close_ui_visual_panel(self):
        if self.ui_window:
            try:
                self.ui_window.evaluate_js("window.closeVisualPanel && window.closeVisualPanel()")
            except Exception:
                pass

    def _set_ui_screen_active(self, active: bool):
        if self.ui_window:
            try:
                self.ui_window.evaluate_js(f"window.setScreenActive && window.setScreenActive({'true' if active else 'false'})")
            except Exception:
                pass

    def _set_ui_camera_active(self, active: bool):
        if self.ui_window:
            try:
                self.ui_window.evaluate_js(f"window.setCameraActive && window.setCameraActive({'true' if active else 'false'})")
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
        print(f"[vision_live] Mute set to: {muted}")
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
        print(f"[vision_live] Auto-muted after {AUTO_MUTE_IDLE_SECONDS}s of inactivity (connection stays open).")
        self._set_ui_status("muted")
        asyncio.create_task(asyncio.to_thread(self._play_system_sound, "auto_mute"))

    def _disengage_auto_mute(self):
        self._muted_auto = False
        self._last_active_time = time.time()
        print("[vision_live] Wake phrase heard while auto-muted -- resuming.")
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

    def _build_config(self):
        kwargs = dict(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
            tools=[{"function_declarations": TOOL_DECLARATIONS}],
            input_audio_transcription={},
            output_audio_transcription={},
            # Tuned for a snappier feel: react to speech starting/stopping
            # faster than the SDK defaults, while keeping enough padding/
            # silence margin to avoid clipping the first phoneme or cutting
            # someone off on a mid-sentence breath.
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
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
            cap = await asyncio.to_thread(cv2.VideoCapture, 0)
            if not cap.isOpened():
                print("[vision_live] Could not open camera.")
                return
            while self._camera_active:
                ret, frame = await asyncio.to_thread(cap.read)
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
            results = semantic_search(args.get("query", ""), n_results=5)
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
            test_stream = await asyncio.to_thread(
                sd.InputStream, samplerate=SEND_SAMPLE_RATE, channels=CHANNELS, dtype="int16"
            )
            await asyncio.to_thread(test_stream.close)
            return True
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
                continue

            try:
                audio_path = await asyncio.to_thread(self._write_wav_chunk, chunk)
                text = await asyncio.to_thread(transcribe, audio_path)
            except Exception as e:
                print(f"[vision_live] Auto-mute wake-word check failed: {e}")
                continue

            if text and WAKE_PHRASE in text.lower():
                self._disengage_auto_mute()

    async def _proactive_monitor_loop(self):
        while True:
            await asyncio.sleep(PROACTIVE_CHECK_INTERVAL)

            if self._active:
                continue

            try:
                result = get_calendar_events(hours_ahead=1)
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
        print(f"[vision_live] Dormant. Say '{config.WAKE_WORD}' to activate...")
        self._set_ui_status("muted" if self._muted else "idle")

        await asyncio.sleep(0.5)

        async def listen_loop():
            while True:
                if self._muted:
                    self._set_ui_status("muted")
                    await asyncio.sleep(0.5)
                    continue

                if not await self._check_mic_available():
                    self._set_ui_status("mic_unavailable")
                    await asyncio.sleep(3)
                    continue
                try:
                    audio_path = await asyncio.to_thread(record_audio, WAKE_CHUNK_DURATION)
                    text = await asyncio.to_thread(transcribe, audio_path)
                except Exception as e:
                    print(f"[vision_live] Mic error while dormant: {e}")
                    self._set_ui_status("mic_unavailable")
                    await asyncio.sleep(1)
                    continue

                print(f"[vision_live] Heard: '{text}'")

                if text and WAKE_PHRASE in text.lower():
                    print(f"[vision_live] Wake phrase heard in: '{text}' — waking up.")
                    return "wake_phrase"

        listen_task = asyncio.create_task(listen_loop())
        proactive_task = asyncio.create_task(self._proactive_trigger.wait())

        done, pending = await asyncio.wait(
            [listen_task, proactive_task], return_when=asyncio.FIRST_COMPLETED
        )

        for t in pending:
            t.cancel()

        if proactive_task in done:
            self._proactive_trigger.clear()
            self._set_ui_status("idle")
            return "proactive"

        self._set_ui_status("idle")
        return "wake_phrase"

    async def _listen_audio_awake(self):
        loop = asyncio.get_event_loop()

        def callback(indata, frames, time_info, status):
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
                loop.call_soon_threadsafe(
                    self.out_queue.put_nowait,
                    {"data": indata.tobytes(), "mime_type": "audio/pcm"},
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

    async def _send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            async with self._session_send_lock:
                await self.session.send_realtime_input(media=msg)

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
                async with self._session_send_lock:
                    await self.session.send_client_content(
                        turns=types.Content(role="user", parts=[types.Part(text=msg)]),
                        turn_complete=True,
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
                        self._last_input_chunk_at = time.time()
                        asyncio.create_task(self._maybe_signal_processing(self._last_input_chunk_at))

                    if sc.output_transcription and sc.output_transcription.text:
                        if not self._is_speaking:
                            self._set_ui_status("speaking")
                        self._is_speaking = True
                        self._speaking_generation += 1
                        out_buf.append(sc.output_transcription.text.strip())

                    if sc.turn_complete:
                        asyncio.create_task(self._finish_speaking(self._speaking_generation))

                        full_in = " ".join(in_buf).strip()
                        full_out = " ".join(out_buf).strip()

                        if full_in:
                            print(f"You: {full_in}")
                            save_message("user", full_in)
                            semantic_add("user", full_in)
                        if full_out:
                            print(f"VISION: {full_out}")
                            save_message("assistant", full_out)
                            semantic_add("assistant", full_out)
                            self._set_ui_response(full_out)

                        in_buf, out_buf = [], []

                if response.tool_call:
                    self._tool_call_in_progress = True
                    try:
                        fn_responses = [await self._execute_tool(fc) for fc in response.tool_call.function_calls]
                    finally:
                        self._tool_call_in_progress = False
                    async with self._session_send_lock:
                        await self.session.send_tool_response(function_responses=fn_responses)

    async def _play_audio(self):
        stream = sd.RawOutputStream(
            samplerate=RECEIVE_SAMPLE_RATE, channels=CHANNELS, dtype="int16", blocksize=CHUNK_SIZE,
        )
        stream.start()
        try:
            while True:
                chunk = await self.audio_in_queue.get()
                await asyncio.to_thread(stream.write, chunk)
        finally:
            stream.stop()
            stream.close()

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
                live_config = self._build_config()

                self._headphones_mode = is_headphones_active()
                print(f"[vision_live] Output device check — headphones mode: {self._headphones_mode}")
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

                        async with self._session_send_lock:
                            await session.send_client_content(
                                turns=types.Content(role="user", parts=[types.Part(text=context_note)]),
                                turn_complete=True,
                            )
                        greeted = True

                    tasks = [
                        asyncio.create_task(self._send_realtime()),
                        asyncio.create_task(self._listen_audio_awake()),
                        asyncio.create_task(self._receive_audio()),
                        asyncio.create_task(self._play_audio()),
                        asyncio.create_task(self._live_notification_watcher()),
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
    init_db()
    live = VisionLive()
    try:
        asyncio.run(live.run())
    except KeyboardInterrupt:
        print("\n[vision_live] Shutting down.")


if __name__ == "__main__":
    main()