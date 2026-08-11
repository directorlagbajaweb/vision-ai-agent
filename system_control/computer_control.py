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

import Quartz

MAX_STEPS = 15

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
    try:
        from Quartz import AXIsProcessTrusted
        return bool(AXIsProcessTrusted())
    except ImportError:
        try:
            from ApplicationServices import AXIsProcessTrusted
            return bool(AXIsProcessTrusted())
        except ImportError:
            print("[computer_control] Could not check Accessibility trust status; proceeding without the check.")
            return True


def record_capture_size(img_w: int, img_h: int) -> None:
    """Called by vision_live.py after it sends a screenshot to the model, so
    subsequent click coordinates (given in that image's pixel space) can be
    converted to real screen points."""
    global _last_capture
    _last_capture = {"img_w": img_w, "img_h": img_h}


def _to_logical_coords(x, y):
    if _last_capture is None:
        raise RuntimeError("No screenshot has been captured yet — call take_screenshot first.")
    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    screen_w = bounds.size.width
    screen_h = bounds.size.height
    logical_x = x * (screen_w / _last_capture["img_w"])
    logical_y = y * (screen_h / _last_capture["img_h"])
    return logical_x, logical_y


def _post_mouse_event(event_type, x, y, click_count=None):
    event = Quartz.CGEventCreateMouseEvent(None, event_type, (x, y), Quartz.kCGMouseButtonLeft)
    if click_count is not None:
        Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventClickState, click_count)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def mouse_move(x, y) -> dict:
    lx, ly = _to_logical_coords(x, y)
    _post_mouse_event(Quartz.kCGEventMouseMoved, lx, ly)
    return {"success": True}


def left_click(x, y) -> dict:
    lx, ly = _to_logical_coords(x, y)
    _post_mouse_event(Quartz.kCGEventMouseMoved, lx, ly)
    _post_mouse_event(Quartz.kCGEventLeftMouseDown, lx, ly, click_count=1)
    _post_mouse_event(Quartz.kCGEventLeftMouseUp, lx, ly, click_count=1)
    return {"success": True}


def double_click(x, y) -> dict:
    lx, ly = _to_logical_coords(x, y)
    _post_mouse_event(Quartz.kCGEventMouseMoved, lx, ly)
    _post_mouse_event(Quartz.kCGEventLeftMouseDown, lx, ly, click_count=1)
    _post_mouse_event(Quartz.kCGEventLeftMouseUp, lx, ly, click_count=1)
    _post_mouse_event(Quartz.kCGEventLeftMouseDown, lx, ly, click_count=2)
    _post_mouse_event(Quartz.kCGEventLeftMouseUp, lx, ly, click_count=2)
    return {"success": True}


def type_text(text: str) -> dict:
    for down in (True, False):
        event = Quartz.CGEventCreateKeyboardEvent(None, 0, down)
        Quartz.CGEventKeyboardSetUnicodeString(event, len(text), text)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
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
    down = Quartz.CGEventCreateKeyboardEvent(None, keycode, True)
    up = Quartz.CGEventCreateKeyboardEvent(None, keycode, False)
    if flags:
        Quartz.CGEventSetFlags(down, flags)
        Quartz.CGEventSetFlags(up, flags)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)
    return {"success": True}


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


def execute(action: str, x=None, y=None, text: str = None, key: str = None) -> dict:
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

    if action != "take_screenshot" and not _accessibility_trusted():
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
