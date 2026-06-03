"""
manager.py — BrowserChannelManager

Maintains the single persistent WebSocket connection from the Chrome extension.
Provides a clean async request/response interface over that connection using
correlation IDs so concurrent tool calls never mix up their responses.

Usage (from a tool):
    from backend.ws.manager import get_channel

    channel = get_channel()
    if not channel.is_connected:
        return {"error": "Browser extension not connected."}
    result = await channel.request("get_dom", {}, timeout=10.0)
"""

import asyncio
import logging
import uuid
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class BrowserChannelManager:
    """
    Singleton that owns the extension WebSocket and exposes
    a correlated async request/response API to the MCP tool layer.
    """

    def __init__(self) -> None:
        self._ws: WebSocket | None = None
        # Pending futures keyed by correlation ID
        self._pending: dict[str, asyncio.Future] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._ws is not None

    async def request(
        self,
        action: str,
        params: dict,
        timeout: float = 10.0,
    ) -> Any:
        """
        Send a command to the extension and await its response.

        Raises:
            RuntimeError: if the extension is not connected.
            asyncio.TimeoutError: if the extension doesn't respond within `timeout`.
        """
        if self._ws is None:
            raise RuntimeError(
                "Browser extension is not connected. "
                "Make sure Friday is running and the extension is loaded."
            )

        req_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[req_id] = future

        try:
            await self._ws.send_json(
                {"id": req_id, "action": action, "params": params}
            )
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "[BrowserChannel] Timeout waiting for '%s' (id=%s)", action, req_id
            )
            raise
        finally:
            self._pending.pop(req_id, None)

    # ── WebSocket lifecycle (called by the route) ─────────────────────────────

    async def connect(self, ws: WebSocket) -> None:
        """
        Accept the extension's WebSocket and run the receive loop.
        Blocks until the connection closes.
        """
        if self._ws is not None:
            # Only one extension connection at a time — reject the old one first.
            try:
                await self._ws.close(code=4000)
            except Exception:
                pass

        self._ws = ws
        logger.info("🔌 Browser channel connected.")

        try:
            await self._receive_loop()
        except WebSocketDisconnect:
            logger.info("Browser channel disconnected.")
        except Exception as exc:
            logger.exception("Browser channel error: %s", exc)
        finally:
            self._ws = None
            # Reject all pending futures so tool calls fail fast
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(
                        RuntimeError("Browser channel disconnected mid-request.")
                    )
            self._pending.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        """Read messages from the extension and resolve pending futures."""
        while True:
            data = await self._ws.receive_json()
            req_id: str = data.get("id", "")
            future = self._pending.get(req_id)

            if future is None or future.done():
                logger.debug(
                    "[BrowserChannel] Received response for unknown/expired id=%s", req_id
                )
                continue

            if "error" in data:
                future.set_exception(RuntimeError(data["error"]))
            else:
                future.set_result(data.get("result"))


# ── Module-level singleton ────────────────────────────────────────────────────

_channel: BrowserChannelManager | None = None


def get_channel() -> BrowserChannelManager:
    """Return the process-wide BrowserChannelManager instance."""
    global _channel
    if _channel is None:
        _channel = BrowserChannelManager()
    return _channel
