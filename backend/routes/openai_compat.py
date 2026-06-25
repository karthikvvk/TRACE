"""
openai_compat.py — OpenAI-compatible API surface for Big-AGI integration.

Provides two endpoints that mirror the OpenAI REST API so that Big-AGI's
LocalAI vendor can talk directly to the TRACE / Friday backend:

    GET  /v1/models               → list available "models"
    POST /v1/chat/completions     → streaming chat (SSE in OpenAI delta format)

The chat endpoint bridges Big-AGI's standard OpenAI streaming protocol to
Friday's internal agent_stream() generator.  Tool-call events are NOT exposed
in the OpenAI delta stream (Big-AGI drives tools via its own UI); instead every
event that carries text is concatenated and streamed as assistant content.

CORS note: Big-AGI runs at http://localhost:3000 and hits this backend directly
(client-side-fetch mode).  The CORS middleware in main.py already allows "*";
no extra headers are needed here.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from pydantic import BaseModel

from backend.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai-compat"])


# ── /v1/models ────────────────────────────────────────────────────────────────

class _ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "friday"


@router.get("/models", summary="List available TRACE / Friday models")
async def list_models() -> dict:
    """
    Return an OpenAI-format model list for Big-AGI.

    When USE_LLM=true: proxies the actual model list from LM Studio/Ollama so Big-AGI
    shows the user's real local models. All chat requests still go through the TRACE
    agent wrapper which injects full tool-calling support.

    When USE_LLM=false: returns the configured Gemini model.
    """
    import httpx as _httpx

    now = int(time.time())

    # Route through Colab brain if connected and colab_mode is True
    if settings.colab_mode:
        from backend.bridge.colab_bridge import get_colab_bridge
        bridge = get_colab_bridge()
        if bridge.is_connected:
            return {"object": "list", "data": [{
                "id": "colab-brain",
                "object": "model",
                "created": now,
                "owned_by": "friday-colab",
            }]}

    if settings.use_llm:
        # Proxy model list from the local LLM server so Big-AGI sees real model names
        try:
            async with _httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{settings.ollama_base_url.rstrip('/')}/models")
                resp.raise_for_status()
                upstream = resp.json()
                # Stamp all upstream models as coming through Friday
                models = []
                for m in upstream.get("data", []):
                    models.append({
                        "id": m.get("id", m.get("name", "unknown")),
                        "object": "model",
                        "created": m.get("created", now),
                        "owned_by": "friday-local",
                    })
                if models:
                    return {"object": "list", "data": models}
        except Exception:
            pass  # LM Studio offline — fall back to configured name

        # Fallback: just return the configured model name
        return {"object": "list", "data": [{
            "id": settings.ollama_model,
            "object": "model",
            "created": now,
            "owned_by": "friday-local",
        }]}
    else:
        # Gemini mode
        return {"object": "list", "data": [{
            "id": settings.gemini_model,
            "object": "model",
            "created": now,
            "owned_by": "friday-gemini",
        }]}


# ── /v1/chat/completions ──────────────────────────────────────────────────────

def _get_content_text(content: Any) -> str:
    """Normalize OpenAI rich/multimodal content payload into a plain string."""
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


class _Message(BaseModel):
    role: str
    content: Any = None


class _ChatRequest(BaseModel):
    model: str | None = None
    messages: list[_Message]
    stream: bool = True
    # The rest of the OpenAI fields are accepted but silently ignored
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None


def _make_chunk(
    chunk_id: str,
    delta_content: str | None = None,
    finish_reason: str | None = None,
    model: str = "friday",
) -> str:
    """Encode one Server-Sent Event chunk in OpenAI delta format."""
    delta: dict = {}
    if delta_content is not None:
        delta["content"] = delta_content
    if not delta and finish_reason is None:
        delta["role"] = "assistant"

    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"


async def _stream_generator(
    request: _ChatRequest,
    tool_router,
) -> AsyncGenerator[str, None]:
    """Bridge Friday's agent_stream events to OpenAI streaming delta format."""
    from backend.engine.agent import agent_stream

    # Convert OpenAI message history to Friday's {role, content} format.
    # Big-AGI sends the full conversation including the latest user message as
    # the last entry; we split the last user message off as `message`.
    messages = [m for m in request.messages if m.role in ("user", "assistant", "system")]

    # Separate the latest user turn from history
    last_user_idx = max(
        (i for i, m in enumerate(messages) if m.role == "user"),
        default=None,
    )
    if last_user_idx is None:
        # Nothing to do
        return

    message = _get_content_text(messages[last_user_idx].content)
    history: list[dict] = []
    for m in messages[:last_user_idx]:
        role = "model" if m.role == "assistant" else m.role
        history.append({"role": role, "content": _get_content_text(m.content)})

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Route through Colab brain if connected and colab_mode is True
    use_colab = False
    if settings.colab_mode:
        from backend.bridge.colab_bridge import get_colab_bridge
        bridge = get_colab_bridge()
        if bridge.is_connected:
            use_colab = True

    if use_colab:
        model_name = request.model or "colab-brain"
        stream = bridge.run_turn(message, history, tool_router)
    else:
        model_name = request.model or (settings.ollama_model if settings.use_llm else settings.gemini_model)
        stream = agent_stream(message, history, tool_router, request.model)

    # Opening chunk (role announcement)
    yield _make_chunk(chunk_id, delta_content=None, model=model_name)

    accumulated_text = ""
    try:
        async for event in stream:
            etype = event.get("type")
            if etype == "text":
                content = event.get("content", "")
                accumulated_text += content
                yield _make_chunk(chunk_id, delta_content=content, model=model_name)
            elif etype == "tool_call":
                # Signal tool activity as a small comment so the user sees progress
                tool_name = event.get("name", "tool")
                notice = f"\n*🔧 Calling `{tool_name}`…*\n"
                yield _make_chunk(chunk_id, delta_content=notice, model=model_name)
            elif etype == "tool_result":
                tool_name = event.get("name", "tool")
                notice = f"*✓ `{tool_name}` done*\n\n"
                yield _make_chunk(chunk_id, delta_content=notice, model=model_name)
            elif etype == "error":
                error_text = f"\n⚠️ **Error:** {event.get('content', 'Unknown error')}\n"
                yield _make_chunk(chunk_id, delta_content=error_text, model=model_name)
            elif etype == "done":
                break
    except Exception as exc:
        logger.exception("[OpenAI compat] Unhandled error in agent stream")
        yield _make_chunk(chunk_id, delta_content=f"\n⚠️ **Server error:** {exc}\n", model=model_name)

    # Terminating chunk
    yield _make_chunk(chunk_id, finish_reason="stop", model=model_name)
    yield "data: [DONE]\n\n"


@router.post("/chat/completions", summary="OpenAI-compatible streaming chat for Big-AGI")
async def chat_completions(body: _ChatRequest, request: Request) -> StreamingResponse:
    """
    Stream a Friday agent turn using OpenAI chat-completion delta format.

    Big-AGI (LocalAI vendor, client-side-fetch) calls this endpoint directly
    from the browser, so no server-side proxy is needed.
    """
    tool_router = request.app.state.tool_router

    if body.stream:
        return StreamingResponse(
            _stream_generator(body, tool_router),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
            },
        )

    # Non-streaming fallback (collect everything and return once)
    parts: list[str] = []
    async for chunk in _stream_generator(body, tool_router):
        if chunk.startswith("data: ") and not chunk.strip() == "data: [DONE]":
            raw = chunk.removeprefix("data: ").strip()
            try:
                obj = json.loads(raw)
                content = obj["choices"][0]["delta"].get("content")
                if content:
                    parts.append(content)
            except Exception:
                pass

    full_text = "".join(parts)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model or "friday",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
