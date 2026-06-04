"""
colab_bridge.py — ColabBridge server-side singleton.

HOW IT WORKS
============
1. The Colab notebook opens a WebSocket to ws://localhost:8000/ws/colab
2. This server sends the full tool schema list to Colab ("tool_schemas" msg)
3. Colab runs the Gemini thinking loop and sends "tool_call" messages here
4. We dispatch those calls through the local ToolRouter and send back "tool_result"
5. Colab streams text chunks back; we forward them as SSE to the browser

MESSAGE PROTOCOL (JSON over WebSocket)
=======================================
Local → Colab
  {"type": "tool_schemas", "tools": [...]}          — sent once on connect
  {"type": "user_message",  "id": str,
   "message": str, "history": [...]}                — one per /chat request

Colab → Local
  {"type": "tool_call",   "id": str,
   "name": str, "args": dict}                       — Gemini wants a tool
  {"type": "tool_result_ack"}                       — (unused, for future use)
  {"type": "text_chunk",  "id": str,
   "content": str}                                  — streaming text
  {"type": "done",        "id": str}                — turn complete
  {"type": "error",       "id": str,
   "content": str}                                  — Gemini error

TO REMOVE THIS FEATURE
=======================
  1. Delete backend/bridge/
  2. Delete backend/routes/colab_ws.py
  3. Remove the 3 marked lines in backend/main.py
  4. Remove the 1 marked block in backend/routes/chat.py
  5. Remove colab_secret / colab_mode from config.py
"""

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncGenerator

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ColabBridge:
    """
    Manages the single persistent WebSocket connection from the Colab notebook.
    Also manages in-flight /chat requests: each gets a unique turn_id so
    concurrent requests (if any) can be matched to the right SSE stream.
    """

    def __init__(self) -> None:
        self._ws: WebSocket | None = None
        # turn_id → asyncio.Queue of events from Colab
        self._queues: dict[str, asyncio.Queue] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._ws is not None

    async def run_turn(
        self,
        message: str,
        history: list[dict],
        tool_router,
    ) -> AsyncGenerator[dict, None]:
        """
        Send a user message to Colab, execute any tool calls that come back,
        and yield agent events (tool_call, tool_result, text, done, error).
        Must only be called when is_connected is True.
        """
        turn_id = str(uuid.uuid4())
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[turn_id] = queue

        try:
            await self._send({
                "type": "user_message",
                "id": turn_id,
                "message": message,
                "history": history,
            })

            while True:
                event: dict = await asyncio.wait_for(queue.get(), timeout=120.0)

                if event["type"] == "tool_call":
                    name = event["name"]
                    args = event.get("args", {})
                    yield {"type": "tool_call", "name": name, "args": args}

                    try:
                        result = await tool_router.dispatch(name, args)
                        result_str = (
                            json.dumps(result) if not isinstance(result, str) else result
                        )
                    except Exception as exc:
                        result_str = f"Tool error: {exc}"
                        logger.warning("[ColabBridge] Tool %r failed: %s", name, exc)

                    yield {"type": "tool_result", "name": name, "result": result_str}

                    # Send result back to Colab so it can continue the LLM loop
                    await self._send({
                        "type": "tool_result",
                        "id": turn_id,
                        "name": name,
                        "result": result_str,
                    })

                elif event["type"] == "text_chunk":
                    yield {"type": "text", "content": event.get("content", "")}

                elif event["type"] == "done":
                    yield {"type": "done"}
                    break

                elif event["type"] == "error":
                    yield {"type": "error", "content": event.get("content", "Unknown error")}
                    yield {"type": "done"}
                    break

        except asyncio.TimeoutError:
            yield {"type": "error", "content": "Colab brain timed out (120 s)."}
            yield {"type": "done"}
        finally:
            self._queues.pop(turn_id, None)

    # ── WebSocket lifecycle (called by the route) ─────────────────────────────

    async def connect(self, ws: WebSocket, tool_router) -> None:
        """
        Accept the Colab WebSocket, send tool schemas, and run the receive loop.
        Blocks until the connection closes.
        """
        if self._ws is not None:
            try:
                await self._ws.close(code=4000)
            except Exception:
                pass

        self._ws = ws
        logger.info("🧠 Colab brain connected.")

        # Send full tool schemas so Colab can build Gemini function declarations
        await self._send({
            "type": "tool_schemas",
            "tools": tool_router.list_tools(),
        })

        try:
            await self._receive_loop()
        except WebSocketDisconnect:
            logger.info("Colab brain disconnected.")
        except Exception as exc:
            logger.exception("Colab bridge error: %s", exc)
        finally:
            self._ws = None
            # Unblock all waiting turn queues
            for q in self._queues.values():
                await q.put({"type": "error", "content": "Colab disconnected mid-turn."})
            self._queues.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _send(self, payload: dict) -> None:
        if self._ws:
            await self._ws.send_json(payload)

    async def _receive_loop(self) -> None:
        """Route incoming Colab messages to the correct turn queue."""
        while True:
            data = await self._ws.receive_json()
            turn_id: str = data.get("id", "")
            queue = self._queues.get(turn_id)

            if queue is None:
                logger.debug("[ColabBridge] No queue for turn_id=%s", turn_id)
                continue

            await queue.put(data)


# ── Module-level singleton ────────────────────────────────────────────────────

_bridge: ColabBridge | None = None


def get_colab_bridge() -> ColabBridge:
    """Return the process-wide ColabBridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = ColabBridge()
    return _bridge
