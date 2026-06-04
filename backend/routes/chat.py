"""
chat.py — POST /chat endpoint with Server-Sent Events streaming.

The side panel (or any HTTP client) sends:
    POST /chat
    { "message": "what tabs do I have open?", "history": [] }

The server streams back SSE events until the agent turn is complete:
    data: {"type": "tool_call",   "name": "browser_get_tabs", "args": {}}
    data: {"type": "tool_result", "name": "browser_get_tabs", "result": "[...]"}
    data: {"type": "text",        "content": "You have 5 tabs open: ..."}
    data: {"type": "done"}

History format: list of {role: "user"|"model", content: str}
"""

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.engine.agent import agent_stream

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


class HistoryTurn(BaseModel):
    role: str       # "user" or "model"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[HistoryTurn] = []
    model: str | None = None    # optional model override


async def _sse_generator(
    message: str,
    history: list[dict],
    tool_router,
    model: str | None,
) -> AsyncGenerator[str, None]:
    """Wrap agent events as SSE data lines. Routes through Colab brain if connected."""
    # ── COLAB BRIDGE (remove this block to always use local agent) ────────────
    from backend.config import settings
    if settings.colab_mode:
        from backend.bridge.colab_bridge import get_colab_bridge
        bridge = get_colab_bridge()
        if bridge.is_connected:
            async for event in bridge.run_turn(message, history, tool_router):
                yield f"data: {json.dumps(event)}\n\n"
            return
    # ── END COLAB BRIDGE ──────────────────────────────────────────────────────
    async for event in agent_stream(message, history, tool_router, model):
        yield f"data: {json.dumps(event)}\n\n"


@router.post("/chat", summary="Run an agent turn and stream the response via SSE")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    """
    Stream a full agent turn as Server-Sent Events.

    Each event is a JSON object on a `data:` line.
    The final event always has `{"type": "done"}`.

    If the browser extension is connected, the agent can call browser_* tools
    inline during the same turn.
    """
    tool_router = request.app.state.tool_router
    history = [t.model_dump() for t in body.history]

    return StreamingResponse(
        _sse_generator(body.message, history, tool_router, body.model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind a proxy
        },
    )


@router.get("/chat/status", summary="Check agent and browser channel status")
async def chat_status(request: Request) -> dict:
    """
    Return the current readiness state of the agent and browser channel.
    Useful for the extension popup / status widget.
    """
    from backend.ws.manager import get_channel
    from backend.config import settings

    channel = get_channel()

    # ── COLAB BRIDGE (remove this block if removing the feature) ─────────────
    from backend.bridge.colab_bridge import get_colab_bridge
    colab = get_colab_bridge()
    colab_connected = colab.is_connected
    # ── END COLAB BRIDGE ─────────────────────────────────────────────────────

    return {
        "agent": "ready",
        "browser_connected": channel.is_connected,
        "colab_connected": colab_connected,       # ← COLAB BRIDGE
        "colab_mode": settings.colab_mode,        # ← COLAB BRIDGE
        "model": settings.gemini_model,
        "script_execution_allowed": settings.allow_script_execution,
    }
