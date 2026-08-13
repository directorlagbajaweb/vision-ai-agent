"""
system_control/computer_control.py
GUI automation primitives for VISION's "Computer Use" feature: synthetic
mouse/keyboard input via Quartz, plus the task-level session/step-limit
guard that bounds an autonomous look-act-look loop.

Screenshots themselves are captured and pushed to the model by
voice/vision_live.py (which owns the realtime video stream) — this module
only needs to know the pixel size of whatever was last sent, via
record_capture_size(), so mouse coordinates the model gives us (in that
screenshot's pixel space) can be converted to real logical screen points.
"""

import ctypes.util
import time
import Quartz

MAX_STEPS = 15

# Apps for which start_computer_use skips the confirmation token entirely —
# an explicit, accepted risk tradeoff: for these apps a task begins with NO
# checkpoint at all, not even the one that exists for everything else. Kept
# as a plain editable set. Checked against whatever app is frontmost RIGHT
# NOW, not parsed from the goal text — so the natural open_app-then-
# start_computer_use flow only goes frictionless once the whitelisted app
# is actually already focused, not just requested by name.
BENIGN_APPS = {"Spotify", "Google Chrome", "Safari", "Music", "TV"}

# A real HID-state event source, rather than None, makes synthetic events
# look like genuine hardware input — some apps (Electron-based ones like
# Spotify especially) silently ignore clicks/keystrokes posted with no
# source or an obviously synthetic one.
_event_source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)

# Small delays between move->down->up mimic real click timing; some apps
# ignore clicks fired back-to-back with zero gap as likely-synthetic.
CLICK_SETTLE_SECONDS = 0.05

_active_session = None
_last_capture = None

# Standard macOS virtual keycodes (kVK_* from Carbon/HIToolbox), enough to
# cover typing, navigation, and common shortcuts.
KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "9": 25, "7": 26, "8": 28, "0": 29,
    "o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38, "k": 40, "n": 45, "m": 46,
    "equal": 24, "minus": 27, "rightbracket": 30, "leftbracket": 33,
    "quote": 39, "semicolon": 41, "backslash": 42, "comma": 43, "slash": 44, "period": 47, "grave": 50,
    "return": 36, "enter": 36, "tab": 48, "space": 49, "delete": 51, "backspace": 51,
    "forwarddelete": 117, "escape": 53, "esc": 53, "help": 114,
    "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
    "left": 123, "right": 124, "down": 125, "up": 126,
    "leftarrow": 123, "rightarrow": 124, "downarrow": 125, "uparrow": 126,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98,
    "f8": 100, "f9": 101, "f10": 109, "f11": 103, "f12": 111,
}

MODIFIER_FLAGS = {
    "cmd": Quartz.kCGEventFlagMaskCommand,
    "command": Quartz.kCGEventFlagMaskCommand,
    "shift": Quartz.kCGEventFlagMaskShift,
    "option": Quartz.kCGEventFlagMaskAlternate,
    "alt": Quartz.kCGEventFlagMaskAlternate,
    "ctrl": Quartz.kCGEventFlagMaskControl,
    "control": Quartz.kCGEventFlagMaskControl,
}


def _accessibility_trusted() -> bool:
    # pyobjc doesn't expose AXIsProcessTrusted on this system (no
    # pyobjc-framework-ApplicationServices installed) — call it directly via
    # ctypes against the system framework instead, which needs no extra
    # dependency. Without this check, a missing permission fails SILENTLY:
    # macOS just drops synthetic input from an untrusted process with no
    # error, so a click looks like it succeeded but nothing happens on screen.
    try:
        lib_path = ctypes.util.find_library("ApplicationServices")
        if lib_path:
            lib = ctypes.CDLL(lib_path)
            return bool(lib.AXIsProcessTrusted())
    except Exception as e:
        print(f"[computer_control] Accessibility check via ctypes failed: {e}")

    print("[computer_control] Could not check Accessibility trust status; proceeding without the check.")
    return True


def record_capture_size(img_w: int, img_h: int, region=None) -> None:
    """Called by vision_live.py after it sends a screenshot to the model, so
    subsequent click coordinates (given in that image's pixel space) can be
    converted to real screen points. `region`, if given, is the logical
    screen rect (x, y, w, h) the image was cropped from — e.g. for a
    zoom_screenshot close-up — so coordinates the model gives us afterward
    are understood as relative to that crop, not the full screen."""
    global _last_capture
    if region is None:
        bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        region = (bounds.origin.x, bounds.origin.y, bounds.size.width, bounds.size.height)
    region_x, region_y, region_w, region_h = region
    _last_capture = {
        "img_w": img_w, "img_h": img_h,
        "region_x": region_x, "region_y": region_y, "region_w": region_w, "region_h": region_h,
    }


def _to_logical_coords(x, y):
    if _last_capture is None:
        raise RuntimeError("No screenshot has been captured yet — call take_screenshot first.")
    c = _last_capture
    logical_x = c["region_x"] + x * (c["region_w"] / c["img_w"])
    logical_y = c["region_y"] + y * (c["region_h"] / c["img_h"])
    # Clamp to the actual screen (not just the capture region) — a coordinate
    # right at the image edge can convert to a point exactly on/past a boundary.
    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    logical_x = min(max(logical_x, bounds.origin.x), bounds.origin.x + bounds.size.width - 1)
    logical_y = min(max(logical_y, bounds.origin.y), bounds.origin.y + bounds.size.height - 1)
    return logical_x, logical_y


def compute_zoom_region(x, y, radius=200):
    """Converts a point in the current screenshot's coordinate space into a
    logical-screen crop region centered on it, clamped to stay on-screen."""
    cx, cy = _to_logical_coords(x, y)
    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    half = max(20, radius)
    region_x = min(max(cx - half, bounds.origin.x), bounds.origin.x + bounds.size.width - 2 * half)
    region_y = min(max(cy - half, bounds.origin.y), bounds.origin.y + bounds.size.height - 2 * half)
    region_x = max(region_x, bounds.origin.x)
    region_y = max(region_y, bounds.origin.y)
    region_w = min(2 * half, bounds.size.width)
    region_h = min(2 * half, bounds.size.height)
    return {"region_x": region_x, "region_y": region_y, "region_w": region_w, "region_h": region_h}


def _post_mouse_event(event_type, x, y, click_count=None):
    event = Quartz.CGEventCreateMouseEvent(_event_source, event_type, (x, y), Quartz.kCGMouseButtonLeft)
    if click_count is not None:
        Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, click_count)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def _current_cursor_location():
    # A no-op event still carries the real, current pointer location — a
    # standard Quartz trick for reading cursor position without a dedicated API.
    return Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))


def _animated_move(x, y, duration=0.25, steps=15):
    """Glides the cursor from wherever it currently is to (x, y) in small
    steps rather than teleporting — so movement is visible on screen instead
    of an instant jump, per the user's request to be able to see it happen."""
    start = _current_cursor_location()
    for i in range(1, steps + 1):
        t = i / steps
        ix = start.x + (x - start.x) * t
        iy = start.y + (y - start.y) * t
        _post_mouse_event(Quartz.kCGEventMouseMoved, ix, iy)
        time.sleep(duration / steps)


def mouse_move(x, y) -> dict:
    lx, ly = _to_logical_coords(x, y)
    _animated_move(lx, ly)
    return {"success": True}


def left_click(x, y) -> dict:
    lx, ly = _to_logical_coords(x, y)
    _animated_move(lx, ly)
    time.sleep(CLICK_SETTLE_SECONDS)
    _post_mouse_event(Quartz.kCGEventLeftMouseDown, lx, ly, click_count=1)
    time.sleep(CLICK_SETTLE_SECONDS)
    _post_mouse_event(Quartz.kCGEventLeftMouseUp, lx, ly, click_count=1)
    return {"success": True}


def double_click(x, y) -> dict:
    lx, ly = _to_logical_coords(x, y)
    _animated_move(lx, ly)
    time.sleep(CLICK_SETTLE_SECONDS)
    _post_mouse_event(Quartz.kCGEventLeftMouseDown, lx, ly, click_count=1)
    time.sleep(CLICK_SETTLE_SECONDS)
    _post_mouse_event(Quartz.kCGEventLeftMouseUp, lx, ly, click_count=1)
    time.sleep(CLICK_SETTLE_SECONDS)
    _post_mouse_event(Quartz.kCGEventLeftMouseDown, lx, ly, click_count=2)
    time.sleep(CLICK_SETTLE_SECONDS)
    _post_mouse_event(Quartz.kCGEventLeftMouseUp, lx, ly, click_count=2)
    return {"success": True}


def right_click(x, y) -> dict:
    lx, ly = _to_logical_coords(x, y)
    _animated_move(lx, ly)
    time.sleep(CLICK_SETTLE_SECONDS)
    event = Quartz.CGEventCreateMouseEvent(_event_source, Quartz.kCGEventRightMouseDown, (lx, ly), Quartz.kCGMouseButtonRight)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
    time.sleep(CLICK_SETTLE_SECONDS)
    event = Quartz.CGEventCreateMouseEvent(_event_source, Quartz.kCGEventRightMouseUp, (lx, ly), Quartz.kCGMouseButtonRight)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
    return {"success": True}


def scroll(direction: str, amount: int = 3) -> dict:
    vertical = {"up": amount, "down": -amount}.get(direction, 0)
    horizontal = {"left": amount, "right": -amount}.get(direction, 0)
    if vertical == 0 and horizontal == 0:
        return {"success": False, "error": f"Unknown scroll direction: {direction}"}
    event = Quartz.CGEventCreateScrollWheelEvent(_event_source, Quartz.kCGScrollEventUnitLine, 2, vertical, horizontal)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
    return {"success": True}


def type_text(text: str) -> dict:
    for down in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(_event_source, 0, down)
        Quartz.CGEventKeyboardSetUnicodeString(event, len(text), text)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        time.sleep(CLICK_SETTLE_SECONDS)
    return {"success": True}


def press_key(key: str) -> dict:
    parts = key.lower().replace(" ", "").split("+")
    main_key, modifiers = parts[-1], parts[:-1]
    if main_key not in KEYCODES:
        return {"success": False, "error": f"Unknown key: {main_key}"}

    flags = 0
    for mod in modifiers:
        if mod not in MODIFIER_FLAGS:
            return {"success": False, "error": f"Unknown modifier: {mod}"}
        flags |= MODIFIER_FLAGS[mod]

    keycode = KEYCODES[main_key]
    down = Quartz.CGEventCreateKeyboardEvent(_event_source, keycode, True)
    up = Quartz.CGEventCreateKeyboardEvent(_event_source, keycode, False)
    if flags:
        Quartz.CGEventSetFlags(down, flags)
        Quartz.CGEventSetFlags(up, flags)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
    time.sleep(CLICK_SETTLE_SECONDS)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)
    return {"success": True}


def _frontmost_app_name():
    try:
        from AppKit import NSWorkspace
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        return app.localizedName() if app else None
    except Exception:
        return None


def needs_confirmation_for_start(**kwargs) -> bool:
    """Whether start_computer_use requires a confirmation token. False only
    when the current frontmost app is on BENIGN_APPS. An undetectable
    frontmost app (None) fails safe — treated as NOT whitelisted."""
    app = _frontmost_app_name()
    return app not in BENIGN_APPS


def start_session(goal: str) -> dict:
    global _active_session
    _active_session = {"goal": goal, "steps_used": 0, "max_steps": MAX_STEPS}
    print(f"[computer_control] Session started: {goal}")
    return {"success": True, "goal": goal, "max_steps": MAX_STEPS}


def end_session() -> dict:
    global _active_session
    had_session = _active_session is not None
    if had_session:
        print(f"[computer_control] Session ended: {_active_session['goal']}")
    _active_session = None
    return {"success": True, "ended": had_session}


def is_session_active() -> bool:
    """Lets other modules (e.g. vision_live.py's live-notification gating)
    check whether a computer-use task is in progress without reaching into
    this module's internal state directly."""
    return _active_session is not None


def execute(action: str, x=None, y=None, text: str = None, key: str = None, direction: str = None, amount: int = None) -> dict:
    """Single entry point for all computer_control tool calls."""
    if _active_session is None:
        return {"success": False, "error": "No active computer-use session. Call start_computer_use first."}

    if _active_session["steps_used"] >= _active_session["max_steps"]:
        return {
            "success": False,
            "error": (
                f"Reached the {_active_session['max_steps']}-step limit for this task. "
                "Call end_computer_use and check in with the user before continuing."
            ),
        }

    if action not in ("take_screenshot", "zoom_screenshot") and not _accessibility_trusted():
        return {
            "success": False,
            "error": (
                "VISION doesn't have Accessibility permission yet. Ask the user to grant it "
                "in System Settings > Privacy & Security > Accessibility, then try again."
            ),
        }

    try:
        if action == "take_screenshot":
            result = {"success": True}
        elif action == "zoom_screenshot":
            if x is None or y is None:
                return {"success": False, "error": "zoom_screenshot requires x and y"}
            result = {"success": True, "zoom_region": compute_zoom_region(x, y, amount if amount else 200)}
        elif action == "mouse_move":
            if x is None or y is None:
                return {"success": False, "error": "mouse_move requires x and y"}
            result = mouse_move(int(round(x)), int(round(y)))
        elif action == "left_click":
            if x is None or y is None:
                return {"success": False, "error": "left_click requires x and y"}
            result = left_click(int(round(x)), int(round(y)))
        elif action == "double_click":
            if x is None or y is None:
                return {"success": False, "error": "double_click requires x and y"}
            result = double_click(int(round(x)), int(round(y)))
        elif action == "right_click":
            if x is None or y is None:
                return {"success": False, "error": "right_click requires x and y"}
            result = right_click(int(round(x)), int(round(y)))
        elif action == "scroll":
            result = scroll(direction or "", amount if amount is not None else 3)
        elif action == "type_text":
            result = type_text(text or "")
        elif action == "press_key":
            result = press_key(key or "")
        else:
            return {"success": False, "error": f"Unknown computer_control action: {action}"}
    except RuntimeError as e:
        return {"success": False, "error": str(e)}

    _active_session["steps_used"] += 1
    result["steps_used"] = _active_session["steps_used"]
    result["steps_remaining"] = _active_session["max_steps"] - _active_session["steps_used"]
    return result
