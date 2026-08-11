"""
system_control/mac_actions.py
Executes macOS system actions on VISION's behalf: opening apps,
running Shortcuts, sending notifications, calendar/email access,
and basic AppleScript control.
"""

import subprocess


def _escape_applescript_string(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def run_applescript(script: str) -> dict:
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return {"success": True, "output": result.stdout.strip()}
        else:
            return {"success": False, "error": result.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "AppleScript timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def open_app(app_name: str) -> dict:
    script = f'tell application "{_escape_applescript_string(app_name)}" to activate'
    result = run_applescript(script)
    if result["success"]:
        print(f"[system_control] Opened {app_name}")
    else:
        print(f"[system_control] Failed to open {app_name}: {result['error']}")
    return result


def close_app(app_name: str) -> dict:
    script = f'tell application "{_escape_applescript_string(app_name)}" to quit'
    result = run_applescript(script)
    if result["success"]:
        print(f"[system_control] Closed {app_name}")
    else:
        print(f"[system_control] Failed to close {app_name}: {result['error']}")
    return result


def open_url(url: str) -> dict:
    if not (url.startswith("http://") or url.startswith("https://")):
        print(f"[system_control] Rejected URL with disallowed scheme: {url}")
        return {"success": False, "error": "Only http:// and https:// URLs are allowed"}
    try:
        subprocess.run(["open", url], check=True, timeout=10)
        print(f"[system_control] Opened URL: {url}")
        return {"success": True}
    except Exception as e:
        print(f"[system_control] Failed to open URL: {e}")
        return {"success": False, "error": str(e)}


def send_notification(title: str, message: str) -> dict:
    script = (
        f'display notification "{_escape_applescript_string(message)}" '
        f'with title "{_escape_applescript_string(title)}"'
    )
    return run_applescript(script)


def run_shortcut(shortcut_name: str, input_text: str = None) -> dict:
    try:
        cmd = ["shortcuts", "run", shortcut_name]
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            print(f"[system_control] Ran Shortcut: {shortcut_name}")
            return {"success": True, "output": result.stdout.strip()}
        else:
            print(f"[system_control] Shortcut failed: {result.stderr.strip()}")
            return {"success": False, "error": result.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Shortcut timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def list_shortcuts() -> list[str]:
    try:
        result = subprocess.run(
            ["shortcuts", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception as e:
        print(f"[system_control] Failed to list shortcuts: {e}")
        return []


def get_volume() -> int:
    result = run_applescript("output volume of (get volume settings)")
    try:
        return int(result["output"])
    except (KeyError, ValueError):
        return -1


def set_volume(level: int) -> dict:
    level = max(0, min(100, level))
    script = f"set volume output volume {level}"
    return run_applescript(script)


def get_calendar_events(hours_ahead: int = 24) -> dict:
    """Gets upcoming Calendar.app events within the next N hours."""
    try:
        hours_ahead = int(hours_ahead)
    except (TypeError, ValueError):
        return {"success": False, "error": "hours_ahead must be an integer"}
    script = f'''
    set outputList to {{}}
    tell application "Calendar"
        set nowDate to current date
        set endDate to nowDate + ({hours_ahead} * hours)
        repeat with cal in calendars
            try
                set theEvents to (every event of cal whose start date ≥ nowDate and start date ≤ endDate)
                repeat with evt in theEvents
                    set eventInfo to (summary of evt) & "|" & ((start date of evt) as string)
                    set end of outputList to eventInfo
                end repeat
            end try
        end repeat
    end tell
    set AppleScript's text item delimiters to "###"
    return outputList as string
    '''
    result = run_applescript(script)
    if not result["success"]:
        return {"success": False, "error": result.get("error")}

    raw = result.get("output", "")
    events = []
    if raw:
        for item in raw.split("###"):
            if "|" in item:
                title, start = item.split("|", 1)
                events.append({"title": title.strip(), "start": start.strip()})
    return {"success": True, "events": events}


def get_recent_emails(limit: int = 5) -> dict:
    """Gets the most recent unread emails from Mail.app's inbox."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return {"success": False, "error": "limit must be an integer"}
    script = f'''
    set outputList to {{}}
    tell application "Mail"
        set unreadMessages to (messages of inbox whose read status is false)
        set msgCount to count of unreadMessages
        if msgCount > {limit} then set msgCount to {limit}
        repeat with i from 1 to msgCount
            set msg to item i of unreadMessages
            set msgInfo to (subject of msg) & "|" & (sender of msg)
            set end of outputList to msgInfo
        end repeat
    end tell
    set AppleScript's text item delimiters to "###"
    return outputList as string
    '''
    result = run_applescript(script)
    if not result["success"]:
        return {"success": False, "error": result.get("error")}

    raw = result.get("output", "")
    emails = []
    if raw:
        for item in raw.split("###"):
            if "|" in item:
                subject, sender = item.split("|", 1)
                emails.append({"subject": subject.strip(), "sender": sender.strip()})
    return {"success": True, "emails": emails}


if __name__ == "__main__":
    print("Available Shortcuts on this Mac:")
    for s in list_shortcuts():
        print(f"  - {s}")

    print("\nSending a test notification...")
    send_notification("VISION", "System control is online.")

    print("\nCurrent volume:", get_volume())