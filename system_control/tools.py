"""
system_control/tools.py
Defines the tool schemas the LLM sees, and translates a model's
tool call into a dispatcher.dispatch() call.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from system_control import dispatcher

# OpenAI-style function schemas — this is what the LLM reads to decide
# whether and how to call a system action.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Open or bring focus to a macOS application.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "description": "Name of the app, e.g. 'Safari'"}
                },
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_app",
            "description": "Quit a macOS application. Requires user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {"type": "string", "description": "Name of the app, e.g. 'Safari'"}
                },
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Open a URL in the default browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL, e.g. 'https://google.com'"}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_volume",
            "description": "Set the system volume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "description": "Volume level 0-100"}
                },
                "required": ["level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shortcut",
            "description": "Run a macOS Shortcut by name. Requires user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "shortcut_name": {"type": "string", "description": "Exact name of the Shortcut"}
                },
                "required": ["shortcut_name"],
            },
        },
    },
]


def execute_tool_call(name: str, arguments: dict) -> dict:
    """Bridges a model's tool call into dispatcher.dispatch()."""
    return dispatcher.dispatch(name, **arguments)