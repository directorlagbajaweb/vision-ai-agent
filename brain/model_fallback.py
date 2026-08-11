"""
brain/model_fallback.py
Walks a chain of models (free-first, paid-fallback) and returns
the first successful response. Used by both everyday.py and coding.py.
"""

import sys
from pathlib import Path

# Allow imports from project root when run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
import config

client = OpenAI(
    base_url=config.OPENROUTER_BASE_URL,
    api_key=config.OPENROUTER_API_KEY,
)


def get_response(messages: list[dict], model_chain: list[str], tools: list[dict] = None) -> dict:
    """
    Try each model in model_chain, in order, until one succeeds.
    Now supports optional tool calling.
    """
    last_error = None

    for attempt, model_id in enumerate(model_chain, start=1):
        try:
            print(f"[model_fallback] Attempt {attempt}: trying {model_id}")

            kwargs = {
                "model": model_id,
                "messages": messages,
                "timeout": config.REQUEST_TIMEOUT_SECONDS,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            print(f"[model_fallback] Success with {model_id}")
            return {
                "message": message,   # full message object (may include tool_calls)
                "model_used": model_id,
                "success": True,
            }

        except Exception as e:
            print(f"[model_fallback] {model_id} failed: {e}")
            last_error = e
            continue

    return {
        "message": None,
        "model_used": None,
        "success": False,
        "error": str(last_error),
    }