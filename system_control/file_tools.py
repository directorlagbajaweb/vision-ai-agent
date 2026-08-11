"""
system_control/file_tools.py
File and document access for VISION: read, list, and search local files.
"""

from pathlib import Path

HOME = Path.home().resolve()
BLOCKED_DIRS = {HOME / ".ssh", HOME / ".aws", HOME / ".gnupg", HOME / "Library", HOME / ".config"}


def _resolve_safe(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not (p == HOME or HOME in p.parents):
        raise PermissionError(f"Access denied: '{path}' is outside the allowed directory")
    for blocked in BLOCKED_DIRS:
        if p == blocked or blocked in p.parents:
            raise PermissionError(f"Access denied: '{path}' is in a restricted location")
    return p


def read_file(path: str, max_chars: int = 5000) -> dict:
    try:
        p = _resolve_safe(path)
        if not p.exists():
            return {"success": False, "error": "File not found"}
        content = p.read_text(errors="ignore")
        truncated = len(content) > max_chars
        return {
            "success": True,
            "content": content[:max_chars],
            "truncated": truncated,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def list_directory(path: str) -> dict:
    try:
        p = _resolve_safe(path)
        if not p.exists() or not p.is_dir():
            return {"success": False, "error": "Directory not found"}
        items = [item.name for item in p.iterdir()]
        return {"success": True, "items": items}
    except Exception as e:
        return {"success": False, "error": str(e)}


def search_files(directory: str, keyword: str, max_results: int = 15) -> dict:
    try:
        p = _resolve_safe(directory)
        if not p.exists():
            return {"success": False, "error": "Directory not found"}
        matches = []
        for item in p.rglob(f"*{keyword}*"):
            matches.append(str(item))
            if len(matches) >= max_results:
                break
        return {"success": True, "matches": matches}
    except Exception as e:
        return {"success": False, "error": str(e)}