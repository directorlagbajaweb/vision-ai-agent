"""
brain/router.py
Classifies incoming requests as "coding" or "everyday", delegates to
the appropriate specialized handler, and manages conversation memory.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from memory.db import init_db, save_message, get_recent_history
from brain import coding, everyday

CODING_KEYWORDS = [
    "code", "coding", "function", "script", "debug", "bug",
    "error", "python", "javascript", "swift", "html", "css",
    "class", "variable", "compile", "refactor", "api", "sql",
    "terminal", "git", "repo", "install package", "write a program",
    "fix this", "syntax", "algorithm",
]


def classify(user_input: str) -> str:
    lowered = user_input.lower()
    for keyword in CODING_KEYWORDS:
        if keyword in lowered:
            return "coding"
    return "everyday"


def route(user_input: str, history_limit: int = 10) -> dict:
    task_type = classify(user_input)
    conversation_history = get_recent_history(limit=history_limit)

    print(f"[router] Classified as: {task_type}")

    handler = coding if task_type == "coding" else everyday
    result = handler.handle(user_input, conversation_history)
    result["task_type"] = task_type

    save_message("user", user_input, task_type=task_type)
    if not result.get("pending_confirmation"):
        save_message("assistant", result["text"], task_type=task_type, model_used=result.get("model_used"))

    return result


def confirm_pending_action(token: str, approved: bool) -> dict:
    from system_control import dispatcher
    result = dispatcher.confirm(token, approved=approved)
    if not approved or result.get("cancelled"):
        return {"text": "Okay, I won't do that.", "success": True}
    return {"text": "Done." if result.get("success") else f"That didn't work: {result.get('error')}", "success": True}


if __name__ == "__main__":
    init_db()
    print("VISION router test (specialized handlers). Type 'quit' to exit.\n")

    while True:
        text = input("You: ")
        if text.lower() in ("quit", "exit"):
            break

        result = route(text)
        print(f"\nVISION ({result.get('task_type')}): {result['text']}\n")

        if result.get("pending_confirmation"):
            answer = input("(y/n) > ").strip().lower()
            confirm_result = confirm_pending_action(
                result["pending_confirmation"]["confirmation_token"],
                approved=(answer == "y"),
            )
            print(f"VISION: {confirm_result['text']}\n")