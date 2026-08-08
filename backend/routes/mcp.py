"""
mcp.py — MCP Streamable HTTP transport endpoint.

Implements the Model Context Protocol (spec: 2024-11-05) over HTTP so that
any MCP-aware client (Claude Desktop, Continue.dev, Cursor, other agents)
can discover and invoke TRACE's tools without any bespoke integration.

Transport:
    POST /mcp  — JSON-RPC 2.0 request/response (all methods)
    GET  /mcp  — SSE stream (server-initiated messages / keepalive ping)

Supported JSON-RPC methods:
    initialize                — handshake; returns serverInfo + capabilities
    notifications/initialized — client ACK (no-op)
    tools/list                — returns all ToolRouter tools in MCP schema
    tools/call                — dispatches to ToolRouter.dispatch()

Auth:
    If settings.extension_secret != "change-me-in-production" (i.e. a real
    secret is configured), requests must carry:
        Authorization: Bearer <extension_secret>
    In dev (default secret), auth is skipped entirely for easy local testing.
"""

import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from backend.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["mcp"])

# ── MCP protocol constants ────────────────────────────────────────────────────

MCP_PROTOCOL_VERSION = "2024-11-05"
_DEV_SECRET = "change-me-in-production"


# ── Auth helper ───────────────────────────────────────────────────────────────

def _check_auth(request: Request) -> None:
    """
    If a non-default extension_secret is configured, enforce Bearer auth.
    Raises HTTP 401 on failure. Skips check entirely in dev mode.
    """
    if settings.extension_secret == _DEV_SECRET:
        return  # dev mode — skip auth

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization: Bearer <token> header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = auth_header.removeprefix("Bearer ").strip()
    if token != settings.extension_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Schema conversion ─────────────────────────────────────────────────────────

def _to_mcp_tool(tool_dict: dict) -> dict:
    """
    Convert a Friday ToolSchema dict → MCP Tool object.

    Friday schema:
        { name, description, input_schema, output_schema, risk_level }

    MCP Tool:
        { name, description, inputSchema }
    """
    raw_schema = tool_dict.get("input_schema", {})

    # Ensure the input schema is a proper JSON Schema object
    if raw_schema.get("type") in ("object", "OBJECT"):
        input_schema = raw_schema
    else:
        # Wrap bare property dict in an object schema
        input_schema = {"type": "object", "properties": raw_schema}

    # Normalise type to lowercase (MCP expects standard JSON Schema casing)
    def _lower_types(schema: Any) -> Any:
        if not isinstance(schema, dict):
            return schema
        result = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, str):
                result[k] = v.lower()
            elif k == "properties" and isinstance(v, dict):
                result[k] = {pk: _lower_types(pv) for pk, pv in v.items()}
            elif k == "items":
                result[k] = _lower_types(v)
            else:
                result[k] = v
        return result

    return {
        "name": tool_dict["name"],
        "description": tool_dict.get("description", ""),
        "inputSchema": _lower_types(input_schema),
    }


# ── JSON-RPC helpers ──────────────────────────────────────────────────────────

def _ok(id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _err(id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


# ── Method handlers ───────────────────────────────────────────────────────────

def _handle_initialize(params: dict, req_id: Any) -> dict:
    """MCP handshake — return serverInfo and capabilities."""
    return _ok(req_id, {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "serverInfo": {
            "name": "TRACE",
            "version": "0.1.0",
        },
        "capabilities": {
            "tools": {},          # we support tools/list + tools/call
        },
    })


def _handle_notifications_initialized(params: dict, req_id: Any) -> dict | None:
    """Client ACK after initialize — no response needed (notification)."""
    return None  # notifications have no id; we return nothing


async def _handle_tools_list(params: dict, req_id: Any, tool_router) -> dict:
    """Return all registered tools in MCP schema format."""
    raw_tools = tool_router.list_tools()
    mcp_tools = [_to_mcp_tool(t) for t in raw_tools]
    return _ok(req_id, {"tools": mcp_tools})


async def _handle_tools_call(params: dict, req_id: Any, tool_router) -> dict:
    """
    Dispatch a tool call and return a CallToolResult.

    MCP CallToolResult:
        { content: [{ type: "text", text: str }], isError?: bool }
    """
    tool_name: str = params.get("name", "")
    arguments: dict = params.get("arguments", {})

    if not tool_name:
        return _err(req_id, -32602, "Missing required param: name")

    try:
        result = await tool_router.dispatch(tool_name, arguments)
        result_text = json.dumps(result) if not isinstance(result, str) else result
        return _ok(req_id, {
            "content": [{"type": "text", "text": result_text}],
            "isError": False,
        })
    except ValueError as exc:
        # Unknown tool name
        logger.warning("[MCP] Unknown tool %r: %s", tool_name, exc)
        return _ok(req_id, {
            "content": [{"type": "text", "text": str(exc)}],
            "isError": True,
        })
    except Exception as exc:
        logger.exception("[MCP] Tool %r execution error", tool_name)
        return _ok(req_id, {
            "content": [{"type": "text", "text": f"Tool error: {exc}"}],
            "isError": True,
        })


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/mcp")
async def mcp_post(request: Request):
    """
    MCP Streamable HTTP — POST handler.

    Accepts a single JSON-RPC 2.0 object or a JSON array of objects (batch).
    Returns a single JSON-RPC response or an array of responses.
    """
    if not settings.mcp_enabled:
        raise HTTPException(status_code=404, detail="MCP endpoint is disabled")

    _check_auth(request)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content=_err(None, -32700, "Parse error: invalid JSON"),
        )

    tool_router = request.app.state.tool_router

    # Support both single request and batch array
    is_batch = isinstance(body, list)
    requests_list: list[dict] = body if is_batch else [body]

    responses = []
    for rpc in requests_list:
        method: str = rpc.get("method", "")
        params: dict = rpc.get("params") or {}
        req_id = rpc.get("id")  # None for notifications

        logger.debug("[MCP] method=%r id=%r", method, req_id)

        try:
            if method == "initialize":
                resp = _handle_initialize(params, req_id)

            elif method == "notifications/initialized":
                resp = _handle_notifications_initialized(params, req_id)

            elif method == "tools/list":
                resp = await _handle_tools_list(params, req_id, tool_router)

            elif method == "tools/call":
                resp = await _handle_tools_call(params, req_id, tool_router)

            else:
                resp = _err(req_id, -32601, f"Method not found: {method!r}")

        except Exception as exc:
            logger.exception("[MCP] Internal error handling method %r", method)
            resp = _err(req_id, -32603, f"Internal error: {exc}")

        # Notifications produce no response (resp is None); skip them
        if resp is not None:
            responses.append(resp)

    if not responses:
        # All were notifications → 204 No Content
        return JSONResponse(status_code=204, content=None)

    payload = responses if is_batch else responses[0]
    return JSONResponse(content=payload)


@router.get("/mcp")
async def mcp_sse(request: Request):
    """
    MCP Streamable HTTP — GET handler (SSE stream).

    The spec requires servers to accept GET /mcp so clients can open an
    SSE channel for server-initiated messages. TRACE has no server-initiated
    MCP messages yet, so this returns a minimal SSE stream with a periodic
    keepalive ping. Clients that only use POST/response will never hit this.
    """
    if not settings.mcp_enabled:
        raise HTTPException(status_code=404, detail="MCP endpoint is disabled")

    _check_auth(request)

    async def _event_stream():
        # Send an initial endpoint event so clients know the channel is live
        session_id = str(uuid.uuid4())
        yield f"event: endpoint\ndata: /mcp?session={session_id}\n\n"

        # Keepalive ping every 15 s (clients disconnect on their own)
        try:
            while True:
                await asyncio.sleep(15)
                if await request.is_disconnected():
                    break
                yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable Nginx buffering if behind a proxy
        },
    )
