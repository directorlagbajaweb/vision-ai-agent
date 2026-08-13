"""
system_control/spotify_control.py
Dedicated Spotify control for VISION — replaces generic computer-use
(screen-reading + clicking) for Spotify actions, which was unreliable.

Auth: Client Credentials Flow only (app-level, no user login/redirect).
This is sufficient because search is a public catalog endpoint; actual
playback goes through local AppleScript to the already-authenticated
desktop app, not through Spotify's Web API playback endpoints. The
access token is cached and refreshed on expiry, not re-fetched per call.

Every action here verifies real player state via AppleScript after
issuing a command, rather than assuming the command worked — failures
report a specific reason (no track found / AppleScript command failed /
state didn't change) so the model can say something accurate.
"""

import time

import requests

import config
from system_control.mac_actions import run_applescript, _escape_applescript_string

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"

_token_cache = {"access_token": None, "expires_at": 0}


def _get_access_token():
    """Returns a cached Client Credentials token, fetching/refreshing one
    only when the cached token is missing or expired (~1hr lifetime)."""
    if _token_cache["access_token"] and time.time() < _token_cache["expires_at"]:
        return _token_cache["access_token"]

    if not config.SPOTIFY_CLIENT_ID or not config.SPOTIFY_CLIENT_SECRET:
        print("[spotify_control] Missing SPOTIFY_CLIENT_ID/SPOTIFY_CLIENT_SECRET in .env")
        return None

    try:
        response = requests.post(
            SPOTIFY_TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(config.SPOTIFY_CLIENT_ID, config.SPOTIFY_CLIENT_SECRET),
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        # Refresh a bit early rather than right at the expiry instant.
        _token_cache["access_token"] = data["access_token"]
        _token_cache["expires_at"] = time.time() + data.get("expires_in", 3600) - 60
        return _token_cache["access_token"]
    except Exception as e:
        print(f"[spotify_control] Token fetch failed: {e}")
        return None


def _search_track(query: str):
    """Returns (uri, label) on success, or (None, reason) on failure."""
    token = _get_access_token()
    if not token:
        return None, "Could not authenticate with Spotify's API — check SPOTIFY_CLIENT_ID/SPOTIFY_CLIENT_SECRET in .env"

    try:
        response = requests.get(
            f"{SPOTIFY_API_BASE}/search",
            headers={"Authorization": f"Bearer {token}"},
            params={"q": query, "type": "track", "limit": 1},
            timeout=10,
        )
        response.raise_for_status()
        items = response.json().get("tracks", {}).get("items", [])
        if not items:
            return None, f"No track found matching '{query}'"
        track = items[0]
        label = f"{track['name']} by {', '.join(a['name'] for a in track['artists'])}"
        return track["uri"], label
    except Exception as e:
        return None, f"Spotify search failed: {e}"


def _get_player_state():
    """Returns Spotify's current player state string ('playing', 'paused',
    'stopped'), or None if the AppleScript query itself failed."""
    result = run_applescript('tell application "Spotify" to get player state')
    if not result["success"]:
        return None
    return result["output"].strip()


def _verify_state(expected_states, attempts=3, delay=0.6):
    """Polls player state a few times rather than checking once immediately —
    gives Spotify room for a cold app launch or a brief settle delay."""
    state = None
    for _ in range(attempts):
        time.sleep(delay)
        state = _get_player_state()
        if state in expected_states:
            return state
    return state


def play_spotify_track(query: str) -> dict:
    uri, info = _search_track(query)
    if uri is None:
        return {"success": False, "error": info}

    script = f'tell application "Spotify" to play track "{_escape_applescript_string(uri)}"'
    result = run_applescript(script)
    if not result["success"]:
        return {"success": False, "error": f"AppleScript command failed: {result['error']}"}

    state = _verify_state({"playing"})
    if state != "playing":
        return {
            "success": False,
            "error": f"Told Spotify to play '{info}', but its player state is still '{state or 'unknown'}', not playing",
        }
    return {"success": True, "track": info, "player_state": state}


def pause_spotify() -> dict:
    result = run_applescript('tell application "Spotify" to pause')
    if not result["success"]:
        return {"success": False, "error": f"AppleScript command failed: {result['error']}"}

    state = _verify_state({"paused"})
    if state != "paused":
        return {"success": False, "error": f"Told Spotify to pause, but its player state is '{state or 'unknown'}', not paused"}
    return {"success": True, "player_state": state}


def resume_spotify() -> dict:
    result = run_applescript('tell application "Spotify" to play')
    if not result["success"]:
        return {"success": False, "error": f"AppleScript command failed: {result['error']}"}

    state = _verify_state({"playing"})
    if state != "playing":
        return {"success": False, "error": f"Told Spotify to resume, but its player state is '{state or 'unknown'}', not playing"}
    return {"success": True, "player_state": state}


def next_spotify_track() -> dict:
    result = run_applescript('tell application "Spotify" to next track')
    if not result["success"]:
        return {"success": False, "error": f"AppleScript command failed: {result['error']}"}

    state = _verify_state({"playing", "paused"})
    if state not in ("playing", "paused"):
        return {"success": False, "error": f"Told Spotify to skip to the next track, but its player state is '{state or 'unknown'}'"}
    return {"success": True, "player_state": state}
