"""
colab_ws.py — WebSocket route for the Colab Brain bridge.

The Colab notebook connects once to:
    ws://localhost:8000/ws/colab?secret=<COLAB_SECRET>

Secret is validated on handshake (same pattern as browser_ws.py).

TO REMOVE: Delete this file and the 3 marked lines in main.py.
"""

import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from backend.config import settings
from backend.bridge.colab_bridge import get_colab_bridge

logger = logging.getLogger(__name__)

router = APIRouter(tags=["colab-bridge"])


@router.websocket("/ws/colab")
async def colab_ws_endpoint(
    ws: WebSocket,
    secret: str = Query(..., description="Shared secret from COLAB_SECRET env var"),
) -> None:
    """
    Persistent WebSocket channel between Friday's backend and the Colab brain.

    On connect:
      1. Validates the shared secret.
      2. Sends full tool schemas to Colab.
      3. Runs the receive loop (blocking until Colab disconnects).
    """
    if secret != settings.colab_secret:
        logger.warning("[ColabWS] Rejected connection — bad secret")
        await ws.accept()
        await ws.close(code=4001)
        return

    await ws.accept()

    from fastapi import Request  # import here to avoid circular at module level
    # tool_router lives on app.state — get it via the ws scope
    tool_router = ws.app.state.tool_router

    bridge = get_colab_bridge()
    await bridge.connect(ws, tool_router)
