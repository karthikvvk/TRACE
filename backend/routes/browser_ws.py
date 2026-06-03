"""
browser_ws.py — WebSocket route for the Chrome extension bridge.

The extension connects once on startup to:
    ws://localhost:8000/ws/browser?secret=<EXTENSION_SECRET>

The secret is validated on handshake. Any connection without the correct
secret is rejected with close code 4001 so other local processes cannot
hijack the browser channel.
"""

import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from backend.config import settings
from backend.ws.manager import get_channel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["browser"])


@router.websocket("/ws/browser")
async def browser_ws_endpoint(
    ws: WebSocket,
    secret: str = Query(..., description="Shared secret from EXTENSION_SECRET env var"),
) -> None:
    """
    Persistent WebSocket channel between Friday's backend and the Chrome extension.

    On connect:
      1. Validates the shared secret.
      2. Hands the socket to BrowserChannelManager which runs the receive loop.

    The channel manager resolves pending tool-call futures as responses arrive.
    """
    if secret != settings.extension_secret:
        logger.warning(
            "[BrowserWS] Rejected connection — bad secret (got %r)", secret[:8] + "…"
        )
        await ws.accept()  # must accept before we can close with a code
        await ws.close(code=4001)
        return

    await ws.accept()
    channel = get_channel()
    await channel.connect(ws)
