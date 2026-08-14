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

slack_catch_me_up is a separate, on-demand, strictly READ-ONLY feature
(history/summary lookup) -- unrelated to the automatic DM/mention alert
path above. It must never call chat_postMessage or any other write
endpoint; it only calls conversations.list/history/replies and
users.list/info.
"""

import asyncio
import ssl
from datetime import datetime, timedelta

import certifi
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.socket_mode.aiohttp import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.web.async_client import AsyncWebClient

import config
from brain.model_fallback import get_response

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


# ── slack_catch_me_up: on-demand, read-only history/summary ─────────────
# Everything below this point only ever calls read endpoints
# (conversations.list/history/replies, users.list/info). Nothing here may
# call chat_postMessage or any other write endpoint.

_CATCH_UP_MESSAGE_CAP = 150  # keeps the summarization prompt bounded for busy channels


def _list_all_users(client) -> list:
    users = []
    cursor = None
    for _ in range(20):
        response = client.users_list(limit=200, cursor=cursor)
        users.extend(response.get("members", []))
        cursor = response.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return users


def _match_users_by_name(users: list, name: str) -> list:
    name = name.strip().lower()
    matches = []
    seen_ids = set()
    for u in users:
        if u.get("deleted") or u.get("is_bot") or u.get("id") == "USLACKBOT":
            continue
        profile = u.get("profile", {})
        candidates = [profile.get("real_name", ""), profile.get("display_name", ""), u.get("name", "")]
        candidates = [c.lower() for c in candidates if c]
        if any(name == c or name in c for c in candidates) and u["id"] not in seen_ids:
            matches.append(u)
            seen_ids.add(u["id"])
    return matches


def _find_channel_by_name(client, name: str):
    """Matches a public/private channel by exact name (case-insensitive, '#' optional).
    Returns (channel_id_or_None, label_or_None, scope_error_or_None) -- a missing-scope
    error here (channels:read/groups:read not granted to the token) is reported back
    rather than raised, so the caller can still fall through to person/DM matching."""
    target = name.lstrip("#").strip().lower()
    cursor = None
    try:
        for _ in range(20):
            response = client.conversations_list(types="public_channel,private_channel", limit=200, cursor=cursor)
            for ch in response.get("channels", []):
                if ch.get("name", "").lower() == target:
                    return ch["id"], f"#{ch['name']}", None
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
    except SlackApiError as e:
        reason = e.response.get("error", "unknown error") if e.response else str(e)
        return None, None, reason
    return None, None, None


def _find_dm_channel(client, name: str):
    """Matches a person by name and finds the existing DM channel with them.
    Returns (channel_id_or_None, display_name_or_None, matched_users)."""
    matched_users = _match_users_by_name(_list_all_users(client), name)
    if len(matched_users) != 1:
        return None, None, matched_users

    user = matched_users[0]
    profile = user.get("profile", {})
    display_name = profile.get("real_name") or profile.get("display_name") or user.get("name")

    cursor = None
    for _ in range(20):
        response = client.conversations_list(types="im", limit=200, cursor=cursor)
        for im in response.get("channels", []):
            if im.get("user") == user["id"]:
                return im["id"], display_name, matched_users
        cursor = response.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return None, display_name, matched_users


def _resolve_catch_up_target(client, channel_or_person: str):
    """Returns (channel_id, label, error). error is None on success, otherwise
    a short human-readable reason to relay back and ask for clarification."""
    raw = channel_or_person.strip()

    channel_id, label, channel_scope_error = _find_channel_by_name(client, raw)
    if channel_id:
        return channel_id, label, None

    dm_channel_id, display_name, matched_users = _find_dm_channel(client, raw)
    if dm_channel_id:
        return dm_channel_id, f"your DM with {display_name}", None
    if len(matched_users) > 1:
        names = ", ".join(
            (u.get("profile", {}).get("real_name") or u.get("name")) for u in matched_users
        )
        return None, None, f"\"{raw}\" matches multiple people ({names}) -- ask which one they meant."
    if len(matched_users) == 1:
        return None, None, f"Found {display_name}, but there's no existing DM with them to summarize."

    if channel_scope_error == "missing_scope":
        return None, None, (
            f"Couldn't find a person matching \"{raw}\", and channel-name lookup isn't "
            f"available yet (the Slack app needs the channels:read and groups:read scopes "
            f"added and reauthorized) -- ask for the exact person's name, or add those "
            f"scopes to look up channels by name."
        )
    return None, None, f"Couldn't find a channel or person matching \"{raw}\" -- ask for the exact name."


def _parse_time_range(time_range: str) -> float:
    """Loosely parses a time_range phrase into a unix timestamp to fetch from.
    Defaults to the last 24 hours if the phrase is ambiguous/unrecognized."""
    tr = (time_range or "").strip().lower()
    now = datetime.now()
    if "yesterday" in tr:
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif "week" in tr:
        start = now - timedelta(days=7)
    elif "today" in tr or not tr:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        start = now - timedelta(hours=24)
    return start.timestamp()


def _fetch_catch_up_messages(client, channel_id: str, oldest_ts: float):
    """Fetches history (paginated) plus replies for the busiest few threads,
    capped at _CATCH_UP_MESSAGE_CAP total messages. Returns (messages, hit_cap)."""
    messages = []
    cursor = None
    hit_cap = False
    for _ in range(10):
        response = client.conversations_history(channel=channel_id, oldest=str(oldest_ts), limit=200, cursor=cursor)
        messages.extend(response.get("messages", []))
        if len(messages) >= _CATCH_UP_MESSAGE_CAP:
            hit_cap = True
            break
        cursor = response.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    messages = messages[:_CATCH_UP_MESSAGE_CAP]

    threaded_parents = sorted(
        (m for m in messages if m.get("reply_count")), key=lambda m: m["reply_count"], reverse=True
    )
    for parent in threaded_parents[:5]:
        if len(messages) >= _CATCH_UP_MESSAGE_CAP:
            hit_cap = True
            break
        try:
            response = client.conversations_replies(channel=channel_id, ts=parent["ts"], limit=50)
            replies = [r for r in response.get("messages", []) if r.get("ts") != parent["ts"]]
        except SlackApiError:
            continue
        room = _CATCH_UP_MESSAGE_CAP - len(messages)
        messages.extend(replies[:room])

    return messages, hit_cap


def _resolve_message_author_sync(client, user_id) -> str:
    if not user_id:
        return "Someone"
    if user_id in _user_name_cache:
        return _user_name_cache[user_id]
    try:
        response = client.users_info(user=user_id)
        profile = response["user"]
        name = profile.get("real_name") or profile.get("name") or user_id
    except Exception:
        name = user_id
    _user_name_cache[user_id] = name
    return name


def _format_catch_up_messages(client, messages: list) -> list:
    messages = sorted(messages, key=lambda m: float(m.get("ts", 0)))
    lines = []
    for m in messages:
        if m.get("subtype") and m.get("subtype") != "thread_broadcast":
            continue  # skip joins/leaves/channel-topic-change/etc., not conversation content
        text = m.get("text", "")
        if not text:
            continue
        if m.get("user"):
            name = _resolve_message_author_sync(client, m["user"])
        else:
            name = m.get("username") or "A bot"
        ts_label = datetime.fromtimestamp(float(m["ts"])).strftime("%a %H:%M")
        lines.append(f"[{ts_label}] {name}: {text}")
    return lines


def _summarize_catch_up(label: str, lines: list, hit_cap: bool):
    if not lines:
        return f"No messages found in {label} for that time range."

    transcript = "\n".join(lines)
    truncation_note = (
        "\n\n(Note: this is only the most recent portion of a busier conversation -- "
        "not the full history for that time range.)"
        if hit_cap
        else ""
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You summarize Slack conversation history for someone who wasn't present. "
                "Be concise and conversational -- this will be read aloud, not displayed as "
                "text. Cover the key points and who said what, and clearly call out anything "
                "that seems to need the listener's attention: a direct question to them, a "
                "decision that was made, a deadline, an action item. If nothing needs their "
                "attention, say so plainly instead of inventing something."
            ),
        },
        {
            "role": "user",
            "content": f"Slack history for {label}:\n\n{transcript}{truncation_note}",
        },
    ]
    result = get_response(messages, config.EVERYDAY_MODEL_CHAIN)
    if not result["success"]:
        return None
    return result["message"].content


def slack_catch_me_up(channel_or_person: str, time_range: str = "today") -> dict:
    """Read-only: resolves a channel or person, fetches recent history, and
    returns a spoken-friendly summary. Never posts anything to Slack -- only
    calls conversations.list/history/replies and users.list/info. Asks for
    clarification (success: False, needs_clarification: True) rather than
    guessing when the target can't be confidently resolved."""
    if not (config.SLACK_USER_TOKEN and config.SLACK_USER_ID):
        return {"success": False, "error": "Slack isn't configured (missing SLACK_USER_TOKEN)."}

    client = _get_sync_web_client()

    try:
        channel_id, label, resolve_error = _resolve_catch_up_target(client, channel_or_person)
    except SlackApiError as e:
        reason = e.response.get("error", "unknown error") if e.response else str(e)
        return {"success": False, "error": f"Slack API error while resolving \"{channel_or_person}\": {reason}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to resolve \"{channel_or_person}\": {e}"}

    if not channel_id:
        return {"success": False, "needs_clarification": True, "error": resolve_error}

    oldest_ts = _parse_time_range(time_range)

    try:
        raw_messages, hit_cap = _fetch_catch_up_messages(client, channel_id, oldest_ts)
    except SlackApiError as e:
        reason = e.response.get("error", "unknown error") if e.response else str(e)
        return {"success": False, "error": f"Slack API error fetching history for {label}: {reason}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to fetch history for {label}: {e}"}

    lines = _format_catch_up_messages(client, raw_messages)
    summary = _summarize_catch_up(label, lines, hit_cap)
    if summary is None:
        return {"success": False, "error": "Fetched the Slack history, but the summarization model failed to respond."}

    return {
        "success": True,
        "channel": label,
        "summary": summary,
        "messages_summarized": len(lines),
        "truncated": hit_cap,
    }
