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
import asyncio
import traceback
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
from system_control.dispatcher import dispatch
from system_control.mac_actions import get_calendar_events
from voice.stt import record_audio, transcribe

LIVE_MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

WAKE_PHRASE = config.WAKE_WORD.lower()
WAKE_CHUNK_DURATION = 3.0
ACTIVE_TIMEOUT = 30
SHUTDOWN_GRACE_PERIOD = 2.0
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
    f"Whenever you write or provide code, ALWAYS call the show_code tool with the "
    f"code instead of speaking it aloud — just give a brief spoken summary of what "
    f"it does. "
    f"When asked to build a webpage, landing page, or website, write complete HTML "
    f"with inline CSS/JS and call show_code with language 'html'. If the user then "
    f"asks you to run, show, or preview it, call render_webpage with that same full "
    f"HTML content — it will render live and full-screen for them. "
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
    f"If the user says something like 'shut down', 'go to sleep', or 'goodbye', "
    f"say a brief warm goodbye and call the shutdown_vision tool. You will wake up "
    f"again the next time the user says your name. "
    f"When the user asks you to complete a task by directly using their screen — "
    f"opening an app, clicking something, searching within an app, filling in a "
    f"field, playing a song, and similar multi-step actions — call start_computer_use "
    f"with a short description of the goal. Once confirmed, you'll get a screenshot "
    f"automatically; from then on, repeat: look at the most recent screenshot, decide "
    f"one next action, call computer_control with exactly one action (mouse_move, "
    f"left_click, double_click, type_text, press_key, or take_screenshot), then look "
    f"at the new screenshot you're sent before deciding the next one. Keep going on "
    f"your own without asking the user again after each step — only speak up if you're "
    f"stuck, need information only they have, or the task looks done. Call "
    f"take_screenshot if you're ever unsure what's currently on screen. When the goal "
    f"is complete, or you're told you've hit the step limit, call end_computer_use and "
    f"briefly tell the user what happened."
)

TOOL_DECLARATIONS = [
    {
        "name": "open_app",
        "description": "Opens or brings focus to a macOS application.",
        "parameters": {"type": "OBJECT", "properties": {"app_name": {"type": "STRING"}}, "required": ["app_name"]},
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
            "and interact with it — use this when the user asks to run, show, "
            "preview, or open a webpage/landing page you wrote."
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
        "name": "shutdown_vision",
        "description": "Puts VISION to sleep until the user says its name again.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "start_computer_use",
        "description": (
            "Starts an autonomous GUI-automation task where VISION directly controls "
            "the mouse and keyboard to complete a multi-step goal (e.g. 'open Spotify, "
            "search for a song, and play it'). Call this WITHOUT confirmation_token first; "
            "if the result has status 'confirmation_required', ask the user to confirm "
            "out loud, then call this tool again passing the confirmation_token value "
            "you received. Once confirmed, use the computer_control tool to act."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "goal": {"type": "STRING", "description": "Short description of what you're trying to accomplish"},
                "confirmation_token": {"type": "STRING"},
            },
            "required": ["goal"],
        },
    },
    {
        "name": "computer_control",
        "description": (
            "Performs one GUI action as part of an active computer-use task (see "
            "start_computer_use) — moving/clicking the mouse, typing text, pressing a "
            "key, or taking a screenshot. After every call, a fresh screenshot is sent "
            "to you automatically — look at it before deciding your next action. "
            "Coordinates are pixel positions within that most recent screenshot. Keep "
            "calling this one action at a time, autonomously, until the goal is done or "
            "you're told the step limit was reached."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "One of: mouse_move, left_click, double_click, type_text, press_key, take_screenshot",
                },
                "x": {"type": "INTEGER", "description": "X coordinate in the last screenshot, for mouse_move/left_click/double_click"},
                "y": {"type": "INTEGER", "description": "Y coordinate in the last screenshot, for mouse_move/left_click/double_click"},
                "text": {"type": "STRING", "description": "Text to type, for type_text"},
                "key": {"type": "STRING", "description": "Key to press, for press_key, e.g. 'enter', 'tab', 'cmd+a'"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "end_computer_use",
        "description": "Ends the current computer-use task. Call this once the goal is complete (or you need to give up and explain why) before speaking to the user again.",
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
        self._shutdown_event = None
        self.ui_window = ui_window
        self._muted = False
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
        self._muted = muted
        print(f"[vision_live] Mute set to: {muted}")
        if self.ui_window:
            try:
                state = "muted" if muted else ("listening" if self._active else "idle")
                self.ui_window.evaluate_js(f"window.updateStatus && window.updateStatus('{state}')")
            except Exception:
                pass

    def _build_config(self):
        kwargs = dict(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
            tools=[{"function_declarations": TOOL_DECLARATIONS}],
            input_audio_transcription={},
            output_audio_transcription={},
        )
        try:
            return types.LiveConnectConfig(enable_affective_dialog=True, **kwargs)
        except Exception as e:
            print(f"[vision_live] Affective dialog not supported, continuing without it: {e}")
            return types.LiveConnectConfig(**kwargs)

    async def _capture_and_queue_screen_frame(self):
        """Grabs one screen frame and sends it to the model via the realtime
        video stream. Also records its pixel size so computer_control can
        translate the model's click coordinates into real screen points.
        Shared by the ambient _screen_share_loop and computer_control's
        post-action "look" step."""
        with mss.mss() as sct:
            screenshot = sct.grab(sct.monitors[1])
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        img.thumbnail((1280, 1280))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        if self.out_queue:
            await self.out_queue.put({"data": buf.getvalue(), "mime_type": "image/jpeg"})
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
        try:
            # NOTE: possessing a valid confirmation_token only proves this exact
            # pending action was offered by dispatch() moments ago — it does NOT
            # prove the user actually said "yes." That verification still depends
            # on the model faithfully following SYSTEM_PROMPT's confirm-before-
            # reuse instruction; the token closes off guessing/replay/naive
            # injection, not model misbehavior.
            if confirmation_token:
                result = dispatcher.confirm(confirmation_token, approved=True)
            else:
                result = dispatch(name, **args)
        except Exception as e:
            result = {"success": False, "error": str(e)}
            traceback.print_exc()
        print(f"[vision_live] Tool result: {result}")

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
            # see what actually happened (or why it didn't).
            await self._capture_and_queue_screen_frame()

        if name == "start_computer_use" and result.get("status") != "confirmation_required" and result.get("success"):
            # Give the model an immediate first look once a task is confirmed,
            # rather than making it call take_screenshot separately.
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
            await self.session.send_realtime_input(media=msg)

    async def _finish_speaking(self):
        while self.audio_in_queue and self.audio_in_queue.qsize() > 0:
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.6)
        self._is_speaking = False
        self._last_active_time = time.time()
        self._set_ui_status("muted" if self._muted else "listening")

    async def _handle_interruption(self):
        """Called when Gemini's server reports the user barged in mid-response."""
        print("[vision_live] User interrupted — stopping playback immediately.")
        # Drain any queued audio so playback stops right away
        while self.audio_in_queue and not self.audio_in_queue.empty():
            try:
                self.audio_in_queue.get_nowait()
            except Exception:
                break
        self._is_speaking = False
        self._last_active_time = time.time()
        self._set_ui_status("listening")

    async def _receive_audio(self):
        in_buf, out_buf = [], []

        while True:
            async for response in self.session.receive():

                if response.data:
                    self.audio_in_queue.put_nowait(response.data)

                if response.server_content:
                    sc = response.server_content

                    if getattr(sc, "interrupted", False):
                        asyncio.create_task(self._handle_interruption())

                    if sc.input_transcription and sc.input_transcription.text:
                        in_buf.append(sc.input_transcription.text.strip())

                    if sc.output_transcription and sc.output_transcription.text:
                        if not self._is_speaking:
                            self._set_ui_status("speaking")
                        self._is_speaking = True
                        out_buf.append(sc.output_transcription.text.strip())

                    if sc.turn_complete:
                        asyncio.create_task(self._finish_speaking())

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
                    fn_responses = [await self._execute_tool(fc) for fc in response.tool_call.function_calls]
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

    async def _watch_timeout(self):
        while True:
            await asyncio.sleep(1)
            if self._active and not self._is_speaking and (time.time() - self._last_active_time > ACTIVE_TIMEOUT):
                print("[vision_live] Silence timeout — going dormant.")
                self._shutdown_event.set()
                return

    async def _run_session(self, proactive_message: str = None):
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        live_config = self._build_config()

        self._shutdown_event = asyncio.Event()
        self._active = True
        self._last_active_time = time.time()

        self._headphones_mode = is_headphones_active()
        print(f"[vision_live] Output device check — headphones mode: {self._headphones_mode}")

        print("[vision_live] Connecting to Gemini Live...")

        async with client.aio.live.connect(model=LIVE_MODEL, config=live_config) as session:
            self.session = session
            self.audio_in_queue = asyncio.Queue()
            self.out_queue = asyncio.Queue(maxsize=10)

            print("[vision_live] Connected.")
            self._set_ui_status("muted" if self._muted else "listening")

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

            await session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text=context_note)]),
                turn_complete=True,
            )

            tasks = [
                asyncio.create_task(self._send_realtime()),
                asyncio.create_task(self._listen_audio_awake()),
                asyncio.create_task(self._receive_audio()),
                asyncio.create_task(self._play_audio()),
                asyncio.create_task(self._watch_timeout()),
            ]
            shutdown_task = asyncio.create_task(self._shutdown_event.wait())

            await asyncio.wait([shutdown_task, *tasks], return_when=asyncio.FIRST_COMPLETED)
            await asyncio.sleep(SHUTDOWN_GRACE_PERIOD)

            self._stop_screen_share()
            self._stop_camera()

            for t in tasks:
                t.cancel()
            shutdown_task.cancel()

        self._active = False
        self.session = None
        self._set_ui_status("muted" if self._muted else "idle")
        print("[vision_live] Session ended.\n")

    async def run(self):
        asyncio.create_task(self._proactive_monitor_loop())

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