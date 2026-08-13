"""
system_control/dispatcher.py
Gatekeeper for all system actions. Decides which actions require
user confirmation before executing.
"""

import sys
import secrets
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from system_control import mac_actions
from system_control import file_tools
from system_control import web_search as web_search_module
from system_control import code_exec
from system_control import computer_control
from system_control import spotify_control

CONFIRMATION_TTL_SECONDS = 120

# Server-side store of one-time confirmation tokens, keyed by token.
# Redeeming a token (via confirm()) always pops it, making it single-use.
_pending_confirmations: dict[str, dict] = {}


ACTION_REGISTRY = {
    "open_app": {"function": mac_actions.open_app, "requires_confirmation": False},
    "open_url": {"function": mac_actions.open_url, "requires_confirmation": False},
    "send_notification": {"function": mac_actions.send_notification, "requires_confirmation": False},
    "get_volume": {"function": mac_actions.get_volume, "requires_confirmation": False},
    "set_volume": {"function": mac_actions.set_volume, "requires_confirmation": False},
    "close_app": {"function": mac_actions.close_app, "requires_confirmation": True},
    "run_shortcut": {"function": mac_actions.run_shortcut, "requires_confirmation": True},
    "get_calendar_events": {"function": mac_actions.get_calendar_events, "requires_confirmation": False},
    "get_recent_emails": {"function": mac_actions.get_recent_emails, "requires_confirmation": False},
    "read_file": {"function": file_tools.read_file, "requires_confirmation": False},
    "list_directory": {"function": file_tools.list_directory, "requires_confirmation": False},
    "search_files": {"function": file_tools.search_files, "requires_confirmation": False},
    "web_search": {"function": web_search_module.web_search, "requires_confirmation": False},
    "execute_python": {"function": code_exec.execute_python, "requires_confirmation": True},
    "start_computer_use": {"function": computer_control.start_session, "requires_confirmation": computer_control.needs_confirmation_for_start},
    "computer_control": {"function": computer_control.execute, "requires_confirmation": False},
    "end_computer_use": {"function": computer_control.end_session, "requires_confirmation": False},
    "play_spotify_track": {"function": spotify_control.play_spotify_track, "requires_confirmation": False},
    "pause_spotify": {"function": spotify_control.pause_spotify, "requires_confirmation": False},
    "resume_spotify": {"function": spotify_control.resume_spotify, "requires_confirmation": False},
    "next_spotify_track": {"function": spotify_control.next_spotify_track, "requires_confirmation": False},
}


def describe_action(action_name: str, **kwargs) -> str:
    if action_name == "close_app":
        return f"Close {kwargs.get('app_name', 'this app')}?"
    if action_name == "run_shortcut":
        return f"Run the Shortcut '{kwargs.get('shortcut_name', 'unknown')}'?"
    if action_name == "execute_python":
        return "Run this code on your Mac?"
    if action_name == "start_computer_use":
        return f"Let VISION control your screen to: {kwargs.get('goal', 'do something')}?"
    return f"Proceed with {action_name}?"


def dispatch(action_name: str, **kwargs) -> dict:
    if action_name not in ACTION_REGISTRY:
        return {"success": False, "error": f"Unknown action: {action_name}"}

    entry = ACTION_REGISTRY[action_name]
    requires_confirmation = entry["requires_confirmation"]
    if callable(requires_confirmation):
        requires_confirmation = requires_confirmation(**kwargs)

    if requires_confirmation:
        token = secrets.token_urlsafe(16)
        _pending_confirmations[token] = {
            "action": action_name,
            "kwargs": kwargs,
            "expires_at": time.time() + CONFIRMATION_TTL_SECONDS,
        }
        return {
            "status": "confirmation_required",
            "action": action_name,
            "confirmation_token": token,
            "prompt": describe_action(action_name, **kwargs),
        }

    return entry["function"](**kwargs)


def confirm(token: str, approved: bool = True) -> dict:
    """Redeems a one-time confirmation token issued by dispatch(). Single-use:
    the token is popped regardless of outcome, so replaying it always fails."""
    entry = _pending_confirmations.pop(token, None)
    if entry is None:
        return {"success": False, "error": "Invalid or already-used confirmation token."}
    if time.time() > entry["expires_at"]:
        return {"success": False, "error": "Confirmation token expired. Please ask again."}
    if not approved:
        return {"success": False, "cancelled": True}

    action_entry = ACTION_REGISTRY[entry["action"]]
    return action_entry["function"](**entry["kwargs"])