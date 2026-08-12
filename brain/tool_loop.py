"""
brain/tool_loop.py
Shared conversation engine: sends messages to a model chain, handles
any tool calls (including the confirm-before-execute flow), and
returns a final natural-language response. Used by both coding.py
and everyday.py so the tool-handling logic lives in exactly one place.
"""

import json
from brain.model_fallback import get_response
from system_control.tools import TOOL_SCHEMAS, execute_tool_call


def run_with_tools(messages: list[dict], model_chain: list[str]) -> dict:
    """
    Sends messages to model_chain with tools enabled. If the model
    calls a tool, executes it (or pauses for confirmation) and asks
    the model for a final natural-language reply.

    Returns:
        {
            "text": str,
            "model_used": str,
            "success": bool,
            "pending_confirmation": dict | None
        }
    """
    result = get_response(messages, model_chain, tools=TOOL_SCHEMAS)

    if not result["success"]:
        return {
            "text": "Sorry, I couldn't reach any model right now.",
            "model_used": None,
            "success": False,
            "pending_confirmation": None,
        }

    message = result["message"]

    if not message.tool_calls:
        return {
            "text": message.content,
            "model_used": result["model_used"],
            "success": True,
            "pending_confirmation": None,
        }

    # ── Model wants to call one or more tools ──
    messages.append(message)

    pending_confirmation = None

    for tool_call in message.tool_calls:
        name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        dispatch_result = execute_tool_call(name, arguments)

        if dispatch_result.get("status") == "confirmation_required":
            if pending_confirmation is None:
                # Only one confirmation can be surfaced to the caller per
                # turn — the first one found wins. A second gated call still
                # runs through dispatch() (minting its own token, which just
                # expires unused via the existing TTL) and its real prompt is
                # still recorded in message history, but re-asking is needed
                # to actually confirm it.
                pending_confirmation = {
                    "action": dispatch_result["action"],
                    "confirmation_token": dispatch_result["confirmation_token"],
                    "prompt": dispatch_result["prompt"],
                }
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({
                    "success": False,
                    "status": "confirmation_required",
                    "prompt": dispatch_result["prompt"],
                }),
            })
            continue

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(dispatch_result),
        })

    if pending_confirmation is not None:
        return {
            "text": pending_confirmation["prompt"],
            "model_used": result["model_used"],
            "success": True,
            "pending_confirmation": pending_confirmation,
        }

    final_result = get_response(messages, model_chain)
    final_text = final_result["message"].content if final_result["success"] else "Done."

    return {
        "text": final_text,
        "model_used": result["model_used"],
        "success": True,
        "pending_confirmation": None,
    }