"""
brain/coding.py
Specialized behavior for coding tasks: system prompt tuned for
technical accuracy and voice-friendly explanations of code.
"""

import config
from brain.tool_loop import run_with_tools

SYSTEM_PROMPT = (
    "You are VISION, an expert coding assistant running on the user's Mac. "
    "The user may be listening to your response spoken aloud, so: "
    "briefly explain what the code does in plain language BEFORE showing it, "
    "keep code blocks minimal and focused on exactly what was asked, "
    "avoid over-explaining basic syntax unless asked, "
    "and prefer working, runnable code over pseudocode. "
    "If the task is complex (multi-file, needs debugging across a project), "
    "say so clearly and suggest using Claude Code for that instead of guessing."
)


def handle(user_input: str, conversation_history: list[dict]) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_input})

    return run_with_tools(messages, config.CODING_MODEL_CHAIN)