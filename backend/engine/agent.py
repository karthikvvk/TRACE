"""
agent.py — Async agent loop, refactored from model_flow.py.

Exposes `agent_stream()` — an async generator that runs a full agent turn
(with tool-calling) and yields structured events the caller can stream
as SSE or consume directly.

Event shapes:
    {"type": "tool_call",   "name": str, "args": dict}
    {"type": "tool_result", "name": str, "result": str}
    {"type": "text",        "content": str}
    {"type": "error",       "content": str}
    {"type": "done"}

The generator runs the Gemini multi-turn loop in asyncio.to_thread() calls
so it doesn't block the FastAPI event loop.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, AsyncGenerator

from backend.engine.router import ToolRouter

logger = logging.getLogger(__name__)


# ── Schema helpers (ported from model_flow.py) ────────────────────────────────

def _clean_schema(schema: dict, for_gemini: bool = False) -> dict:
    """Strip unsupported keys and optionally uppercase type values for Gemini."""
    if not isinstance(schema, dict):
        return schema
    allowed = {"type", "format", "description", "enum", "properties", "required", "items"}
    cleaned = {}
    for k, v in schema.items():
        if k not in allowed:
            continue
        if k == "type" and isinstance(v, str):
            cleaned[k] = v.upper() if for_gemini else v
        elif k == "properties":
            cleaned[k] = {pk: _clean_schema(pv, for_gemini) for pk, pv in v.items()}
        elif k == "items":
            cleaned[k] = _clean_schema(v, for_gemini)
        else:
            cleaned[k] = v
    return cleaned


def _to_gemini_tool(tool_dict: dict) -> dict:
    raw = tool_dict.get("input_schema", {})
    if raw.get("type") in ("object", "OBJECT"):
        params = _clean_schema(raw, for_gemini=True)
    else:
        params = _clean_schema({"type": "OBJECT", "properties": raw}, for_gemini=True)
    return {
        "name": tool_dict["name"],
        "description": tool_dict["description"],
        "parameters": params,
    }


# ── Core agent loop ───────────────────────────────────────────────────────────

async def agent_stream(
    message: str,
    history: list[dict],
    tool_router: ToolRouter,
    model: str | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Run a single agent turn against Gemini, yielding events.

    Args:
        message:      The user's latest message.
        history:      Prior conversation turns as {role, content} dicts.
        tool_router:  The server's ToolRouter (already has browser tools registered).
        model:        Gemini model name. Falls back to settings.gemini_model.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        yield {"type": "error", "content": "google-genai is not installed."}
        yield {"type": "done"}
        return

    from backend.config import settings

    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        yield {"type": "error", "content": "GEMINI_API_KEY is not set."}
        yield {"type": "done"}
        return

    chosen_model = model or settings.gemini_model

    # Build Gemini tool declarations
    func_declarations = [_to_gemini_tool(t) for t in tool_router.list_tools()]
    gemini_tools = [{"function_declarations": func_declarations}]

    # Build initial history for the chat
    gemini_history = []
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        gemini_history.append(
            types.Content(role=role, parts=[types.Part(text=content)])
        )

    client = genai.Client(api_key=api_key)

    # Create async chat session
    chat = client.aio.chats.create(
        model=chosen_model,
        config=types.GenerateContentConfig(
            tools=gemini_tools,
            temperature=0.7,
            system_instruction=(
                "You are Friday, a local-first AI assistant with access to the user's "
                "terminal, tasks, memory, and browser. You can read and interact with "
                "any open browser tab, navigate URLs, click elements, and fill forms. "
                "Always prefer the least invasive tool first (read before write). "
                "When using browser tools, check is_connected before assuming the "
                "browser is available."
            ),
        ),
        history=gemini_history,
    )

    try:
        response = await chat.send_message(message)

        # Tool-calling loop
        while response.function_calls:
            tool_responses = []

            for fc in response.function_calls:
                name = fc.name
                args = {k: v for k, v in fc.args.items()}

                yield {"type": "tool_call", "name": name, "args": args}

                try:
                    result = await tool_router.dispatch(name, args)
                    result_str = json.dumps(result) if not isinstance(result, str) else result
                except Exception as exc:
                    result_str = f"Tool error: {exc}"
                    logger.warning("[Agent] Tool %r failed: %s", name, exc)

                yield {"type": "tool_result", "name": name, "result": result_str}

                tool_responses.append(
                    types.Part.from_function_response(
                        name=name,
                        response={"result": result_str},
                    )
                )

            response = await chat.send_message(tool_responses)

        # Final text response
        text = response.text or ""
        if text:
            yield {"type": "text", "content": text}

    except Exception as exc:
        logger.exception("[Agent] Unhandled error in agent_stream")
        yield {"type": "error", "content": str(exc)}

    yield {"type": "done"}
