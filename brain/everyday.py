"""
brain/everyday.py
Specialized behavior for everyday tasks: system prompt tuned for
quick, natural, conversational assistant behavior.
"""

import config
from brain.tool_loop import run_with_tools

SYSTEM_PROMPT = (
    "You are VISION, a friendly everyday assistant running on the user's Mac. "
    "The user is likely talking to you out loud, so keep replies short, "
    "natural, and conversational — like a quick verbal answer, not a written "
    "report. Use tools directly when the user asks you to open apps, close "
    "apps, open links, adjust volume, or run Shortcuts, rather than just "
    "describing how they'd do it themselves. Avoid unnecessary caveats or "
    "over-explaining simple requests."
)


def handle(user_input: str, conversation_history: list[dict]) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_input})

    return run_with_tools(messages, config.EVERYDAY_MODEL_CHAIN)