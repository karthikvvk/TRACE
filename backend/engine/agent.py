"""
agent.py — Async agent loop.

Exposes `agent_stream()` — an async generator that runs a full agent turn
(with tool-calling) and yields structured events the caller can stream
as SSE or consume directly.

Event shapes:
    {"type": "tool_call",   "name": str, "args": dict}
    {"type": "tool_result", "name": str, "result": str}
    {"type": "text",        "content": str}
    {"type": "error",       "content": str}
    {"type": "done"}

Routing logic (checked in order):
    1. If settings.use_llm is True  → Ollama (OpenAI-compatible /v1 endpoint)
    2. Otherwise                    → Gemini (google-genai)

Configure in .env:
    USE_LLM=true               # enable local Ollama
    OLLAMA_BASE_URL=http://localhost:11434   # default
    OLLAMA_MODEL=llama3                     # default
"""

import json
import logging
import os
from typing import AsyncGenerator

from backend.engine.router import ToolRouter

logger = logging.getLogger(__name__)


# ── Schema helpers ────────────────────────────────────────────────────────────

def _clean_schema(schema: dict, for_gemini: bool = False) -> dict:
    """Strip unsupported keys; optionally uppercase type values for Gemini."""
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


def _to_openai_tool(tool_dict: dict) -> dict:
    """Convert Friday's ToolSchema to full OpenAI function-calling format."""
    raw = tool_dict.get("input_schema", {})
    if raw.get("type") in ("object", "OBJECT"):
        params = _clean_schema(raw, for_gemini=False)
    else:
        params = _clean_schema({"type": "object", "properties": raw}, for_gemini=False)
    return {
        "type": "function",
        "function": {
            "name": tool_dict["name"],
            "description": tool_dict["description"],
            "parameters": params,
        },
    }


def _to_slim_openai_tool(tool_dict: dict) -> dict:
    """
    Minimal OpenAI tool schema — strips all parameter-level descriptions,
    keeping only type/enum/required. Cuts token usage by ~70%, which is
    critical for small local models with limited context windows (e.g. 4096).
    """
    def _strip_descriptions(schema: dict) -> dict:
        """Recursively remove 'description' keys from parameter schemas."""
        if not isinstance(schema, dict):
            return schema
        return {
            k: (_strip_descriptions(v) if isinstance(v, dict) else
                {pk: _strip_descriptions(pv) for pk, pv in v.items()} if k == "properties" else
                [_strip_descriptions(i) for i in v] if k == "items" and isinstance(v, list) else
                _strip_descriptions(v) if k == "items" else v)
            for k, v in schema.items()
            if k != "description"  # strip descriptions from parameter properties
        }

    raw = tool_dict.get("input_schema", {})
    if raw.get("type") in ("object", "OBJECT"):
        params = _strip_descriptions(_clean_schema(raw, for_gemini=False))
    else:
        params = _strip_descriptions(_clean_schema({"type": "object", "properties": raw}, for_gemini=False))

    # Keep only the first sentence of the tool description to save tokens
    full_desc = tool_dict.get("description", "")
    short_desc = full_desc.split(".")[0].strip() + "." if "." in full_desc else full_desc

    return {
        "type": "function",
        "function": {
            "name": tool_dict["name"],
            "description": short_desc,
            "parameters": params,
        },
    }



# ── Ollama agent (OpenAI-compatible) ─────────────────────────────────────────

async def _ollama_stream(
    message: str,
    history: list[dict],
    tool_router: ToolRouter,
    base_url: str,
    model: str,
) -> AsyncGenerator[dict, None]:
    # Run a full tool-calling agent loop against any OpenAI-compatible local
    # LLM server (LM Studio, Ollama, llama.cpp, etc.) using the OpenAI SDK.

    # The `base_url` should already include the /v1 path if required,
    # e.g. "http://localhost:1234/v1" for LM Studio.

    try:
        from openai import AsyncOpenAI
    except ImportError:
        yield {"type": "error", "content": "openai package is not installed. Run: pip install openai"}
        yield {"type": "done"}
        return

    client = AsyncOpenAI(
        base_url=base_url.rstrip("/"),  # use exactly as configured — e.g. http://localhost:1234/v1
        api_key="lm-studio",            # any non-empty string; LM Studio / Ollama ignore it
    )

    # Use slim schemas for small local models to avoid context overflow.
    # Full schemas can be 5000+ tokens with 24 tools; slim cuts that by ~70%.
    tools = [_to_slim_openai_tool(t) for t in tool_router.list_tools()]

    # Build message list — keep system prompt short to preserve context budget
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are Friday, an AI assistant on the user's Linux machine. "
                "Call tools when needed. Confirm before destructive actions."
            ),
        }
    ]
    for turn in history:
        role = turn.get("role", "user")
        # Gemini uses "model" role; OpenAI uses "assistant"
        if role == "model":
            role = "assistant"
        messages.append({"role": role, "content": turn.get("content", "")})
    messages.append({"role": "user", "content": message})

    try:
        while True:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
            )

            choice = response.choices[0]
            msg = choice.message

            # Append assistant message to conversation
            messages.append(msg.model_dump(exclude_none=True))

            if msg.tool_calls:
                # Execute each tool call
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    yield {"type": "tool_call", "name": name, "args": args}

                    try:
                        result = await tool_router.dispatch(name, args)
                        result_str = json.dumps(result) if not isinstance(result, str) else result
                    except Exception as exc:
                        result_str = f"Tool error: {exc}"
                        logger.warning("[Agent/Ollama] Tool %r failed: %s", name, exc)

                    yield {"type": "tool_result", "name": name, "result": result_str}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })
            else:
                # No more tool calls → final text response
                text = msg.content or ""
                if text:
                    yield {"type": "text", "content": text}
                break

    except Exception as exc:
        logger.exception("[Agent/Ollama] Unhandled error")
        yield {"type": "error", "content": str(exc)}

    yield {"type": "done"}


# ── Gemini agent ──────────────────────────────────────────────────────────────

async def _gemini_stream(
    message: str,
    history: list[dict],
    tool_router: ToolRouter,
    api_key: str,
    model: str,
) -> AsyncGenerator[dict, None]:
    """Run a full agent turn against Gemini, yielding events."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        yield {"type": "error", "content": "google-genai is not installed."}
        yield {"type": "done"}
        return

    func_declarations = [_to_gemini_tool(t) for t in tool_router.list_tools()]
    gemini_tools = [{"function_declarations": func_declarations}]

    gemini_history = []
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        gemini_history.append(
            types.Content(role=role, parts=[types.Part(text=content)])
        )

    client = genai.Client(api_key=api_key)
    chat = client.aio.chats.create(
        model=model,
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
                    logger.warning("[Agent/Gemini] Tool %r failed: %s", name, exc)

                yield {"type": "tool_result", "name": name, "result": result_str}

                tool_responses.append(
                    types.Part.from_function_response(
                        name=name,
                        response={"result": result_str},
                    )
                )

            response = await chat.send_message(tool_responses)

        text = response.text or ""
        if text:
            yield {"type": "text", "content": text}

    except Exception as exc:
        logger.exception("[Agent/Gemini] Unhandled error in agent_stream")
        yield {"type": "error", "content": str(exc)}

    yield {"type": "done"}


# ── Public API ────────────────────────────────────────────────────────────────

async def agent_stream(
    message: str,
    history: list[dict],
    tool_router: ToolRouter,
    model: str | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Route a single agent turn to either Ollama (local) or Gemini (cloud).

    Routing (checked in order):
        1. settings.use_llm == True  → Ollama at settings.ollama_base_url
        2. Otherwise                 → Gemini with settings.gemini_api_key

    Args:
        message:     The user's latest message.
        history:     Prior turns as [{role, content}, ...].
        tool_router: The server's ToolRouter.
        model:       Optional override for the model name.
    """
    from backend.config import settings

    if settings.use_llm:
        chosen_model = model or settings.ollama_model
        logger.info("[Agent] Routing to local Ollama: %s @ %s", chosen_model, settings.ollama_base_url)
        async for event in _ollama_stream(
            message, history, tool_router,
            base_url=settings.ollama_base_url,
            model=chosen_model,
        ):
            yield event
    else:
        api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            yield {"type": "error", "content": "GEMINI_API_KEY is not set."}
            yield {"type": "done"}
            return
        chosen_model = model or settings.gemini_model
        logger.info("[Agent] Routing to Gemini: %s", chosen_model)
        async for event in _gemini_stream(
            message, history, tool_router,
            api_key=api_key,
            model=chosen_model,
        ):
            yield event
