"""
system_control/slack_control.py
Slack integration for VISION (Phase 1: DMs + being @-mentioned in a
channel/group thread). Listens in real time via Socket Mode (no polling).

send_slack_message is the ONLY function that actually posts to Slack.
It must be reached exclusively through system_control/dispatcher.py's
confirmation-token system (registered there with requires_confirmation:
True) -- never call it directly from the listener or anywhere else. This
is what makes "never auto-send without confirmation" a real code-level
guarantee rather than something the model is just asked nicely to respect.
"""

import asyncio
import ssl

import certifi
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.socket_mode.aiohttp import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web.async_client import AsyncWebClient

import config

# Two separate clients, matching each call site's actual execution context:
# the async one is used from the Socket Mode event loop directly (no
# threading involved); the sync one is used by send_slack_message, which
# dispatcher.dispatch() calls synchronously via asyncio.to_thread from the
# live pipeline -- matches every other registered action's calling convention.
_async_web_client = None
_sync_web_client = None

_user_name_cache = {}
_channel_name_cache = {}

# slack_sdk's clients don't use requests/certifi like the rest of this
# codebase, so they can't always find the system's CA bundle on macOS --
# pass certifi's explicitly rather than relying on a manual system fix.
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# Holds the callback passed to run_slack_listener so the Socket Mode
# listener function (registered once, can't easily take extra args) can
# reach it.
_event_callback = {"fn": None}


def _get_async_web_client():
    global _async_web_client
    if _async_web_client is None:
        _async_web_client = AsyncWebClient(token=config.SLACK_USER_TOKEN, ssl=_SSL_CONTEXT)
    return _async_web_client


def _get_sync_web_client():
    global _sync_web_client
    if _sync_web_client is None:
        _sync_web_client = WebClient(token=config.SLACK_USER_TOKEN, ssl=_SSL_CONTEXT)
    return _sync_web_client


async def _resolve_user_name(user_id):
    if not user_id:
        return "someone"
    if user_id in _user_name_cache:
        return _user_name_cache[user_id]
    try:
        response = await _get_async_web_client().users_info(user=user_id)
        profile = response["user"]
        name = profile.get("real_name") or profile.get("name") or user_id
    except Exception:
        name = user_id
    _user_name_cache[user_id] = name
    return name


async def _resolve_channel_name(channel_id):
    if channel_id in _channel_name_cache:
        return _channel_name_cache[channel_id]
    try:
        response = await _get_async_web_client().conversations_info(channel=channel_id)
        info = response["channel"]
        name = f"#{info['name']}" if info.get("name") else channel_id
    except Exception:
        name = channel_id
    _channel_name_cache[channel_id] = name
    return name


async def _fetch_context(channel_id, thread_ts):
    """Best-effort recent context so the message doesn't read in isolation."""
    try:
        if thread_ts:
            response = await _get_async_web_client().conversations_replies(channel=channel_id, ts=thread_ts, limit=10)
            messages = response.get("messages", [])
        else:
            response = await _get_async_web_client().conversations_history(channel=channel_id, limit=10)
            messages = list(reversed(response.get("messages", [])))
        return [m.get("text", "") for m in messages if m.get("text")]
    except Exception as e:
        print(f"[slack_control] Failed to fetch context: {e}")
        return []


def _is_relevant(event) -> bool:
    if event.get("subtype"):
        # Edits, joins, bot messages, etc. -- not a new message to react to.
        return False
    if event.get("channel_type") == "im":
        return True
    text = event.get("text", "") or ""
    return f"<@{config.SLACK_USER_ID}>" in text


async def _handle_request(client, req: SocketModeRequest):
    # Ack immediately regardless of relevance -- Slack requires this promptly.
    await client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))

    if req.type != "events_api":
        return

    event = req.payload.get("event", {})
    if event.get("type") != "message" or not _is_relevant(event):
        return

    channel_id = event.get("channel")
    is_dm = event.get("channel_type") == "im"
    thread_ts = event.get("thread_ts")

    user_name = await _resolve_user_name(event.get("user"))
    channel_name = None if is_dm else await _resolve_channel_name(channel_id)
    context_messages = await _fetch_context(channel_id, thread_ts)

    callback = _event_callback["fn"]
    if callback is None:
        return

    await callback({
        "channel": channel_id,
        "thread_ts": thread_ts or event.get("ts"),
        "user_name": user_name,
        "text": event.get("text", ""),
        "is_dm": is_dm,
        "channel_name": channel_name,
        "context_messages": context_messages,
    })


async def run_slack_listener(on_relevant_event):
    """Long-running task: connects via Socket Mode and calls
    on_relevant_event(event_dict) for every relevant DM/mention. Never
    returns on its own -- meant to be run as a background asyncio task for
    the lifetime of the app, independent of VISION's awake/dormant state."""
    if not (config.SLACK_APP_TOKEN and config.SLACK_USER_TOKEN and config.SLACK_USER_ID):
        print("[slack_control] Missing SLACK_APP_TOKEN/SLACK_USER_TOKEN/SLACK_USER_ID in .env -- Slack integration disabled.")
        return

    _event_callback["fn"] = on_relevant_event

    client = SocketModeClient(app_token=config.SLACK_APP_TOKEN, web_client=_get_async_web_client())
    client.socket_mode_request_listeners.append(_handle_request)

    await client.connect()
    print("[slack_control] Connected via Socket Mode -- listening for DMs and mentions.")
    await asyncio.Event().wait()


def send_slack_message(channel: str, text: str, thread_ts: str = None) -> dict:
    """The only function that actually posts to Slack. Plain sync function
    (uses the sync WebClient) since dispatcher.dispatch() calls action
    functions synchronously via asyncio.to_thread."""
    try:
        response = _get_sync_web_client().chat_postMessage(channel=channel, text=text, thread_ts=thread_ts)
    except SlackApiError as e:
        reason = e.response.get("error", "unknown error") if e.response else str(e)
        return {"success": False, "error": f"Slack API rejected the message: {reason}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to send Slack message: {e}"}

    if not response.get("ok"):
        return {"success": False, "error": f"Slack API returned ok=false: {response.get('error', 'unknown error')}"}

    return {"success": True, "channel": channel, "ts": response.get("ts")}
