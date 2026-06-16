"""
test_browser_tools.py — Programmatic tests for browser tools + WebSocket bridge.

Architecture:
  ┌─────────────┐  WebSocket   ┌──────────────────────────────┐
  │ FakeExtension│ ──────────► │  Friday FastAPI (random port)│
  │ (test-local) │ ◄────────── │  /ws/browser                 │
  └─────────────┘              └──────────────────────────────┘
                                            │ calls
                                   ┌────────▼────────┐
                                   │  BrowserXxxTool  │
                                   └─────────────────┘

Key design decisions:
  • No agent / no LLM — tool.execute() is called directly.
  • A real uvicorn instance starts on a RANDOM free port per test
    (avoids collision with the dev server on 8000/8765).
  • A FakeExtension coroutine simulates Chrome: reads server→extension
    requests and sends back pre-configured mock payloads.
  • All "Connected" tests use function-scoped fixtures to stay inside
    a single asyncio event loop (compatible with asyncio_mode=auto).
"""

import asyncio
import json
import socket

import pytest
import pytest_asyncio
from websockets.asyncio.client import connect as ws_connect

from backend.ws.manager import get_channel
from backend.config import settings
from backend.tools.browser_tool import (
    BrowserGetActiveTabTool,
    BrowserGetPageTitleTool,
    BrowserNavigateTool,
    BrowserOpenTabTool,
    BrowserCloseTabTool,
    BrowserClickTool,
    BrowserFillInputTool,
    BrowserExecuteScriptTool,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

SECRET = settings.extension_secret  # "change-me-in-production"


def _free_port() -> int:
    """Ask the OS for an available TCP port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Channel reset ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_channel():
    """
    Reset the BrowserChannelManager singleton before/after each test so
    state doesn't leak between tests.
    """
    import backend.ws.manager as mgr
    mgr._channel = None
    yield
    mgr._channel = None


# ── Live server fixture (function-scoped, random port) ─────────────────────────

@pytest_asyncio.fixture
async def live_server():
    """
    Start a real uvicorn server on a random free port.
    Yields the WebSocket base URL, e.g. "ws://127.0.0.1:54321".
    Tears down cleanly after the test.
    """
    import uvicorn
    from backend.main import app

    port = _free_port()
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="critical"
    )
    server = uvicorn.Server(config)

    # Run the server in a background task on the *current* event loop
    task = asyncio.create_task(server.serve())

    # Wait until uvicorn says it's ready
    deadline = asyncio.get_event_loop().time() + 8.0
    while not server.started:
        if asyncio.get_event_loop().time() > deadline:
            server.should_exit = True
            await task
            raise RuntimeError("uvicorn did not start in time")
        await asyncio.sleep(0.05)

    yield f"ws://127.0.0.1:{port}"

    server.should_exit = True
    await task


# ── FakeExtension ──────────────────────────────────────────────────────────────

class FakeExtension:
    """
    Simulates the Chrome extension WebSocket client.

    response_map: {action_name: payload_to_return}
    e.g. {"get_active_tab": {"id": 1, "url": "https://example.com"}}

    Usage:
        ext = FakeExtension(response_map)
        await ext.start(server_ws_url)
        # ... run tool.execute() ...
        await ext.stop()
    """

    def __init__(self, response_map: dict, *, secret: str = SECRET):
        self.response_map = response_map
        self.secret = secret
        self._task: asyncio.Task | None = None
        self._ws = None
        self._ready = asyncio.Event()

    async def _run(self, server_url: str) -> None:
        uri = f"{server_url}/ws/browser?secret={self.secret}"
        try:
            async with ws_connect(uri) as ws:
                self._ws = ws
                self._ready.set()
                async for raw in ws:
                    msg = json.loads(raw)
                    req_id = msg.get("id", "")
                    action = msg.get("action", "")
                    payload = self.response_map.get(action)
                    if payload is None:
                        reply = {"id": req_id, "error": f"Unknown action: {action}"}
                    else:
                        reply = {"id": req_id, "result": payload}
                    await ws.send(json.dumps(reply))
        except Exception:
            pass
        finally:
            self._ready.set()  # unblock callers even on failure

    async def start(self, server_url: str) -> None:
        self._task = asyncio.create_task(self._run(server_url))
        await self._ready.wait()
        await asyncio.sleep(0.08)  # let BrowserChannelManager.connect() finish

    async def stop(self) -> None:
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass


# ===========================================================================
# 1. Tests when the extension is NOT connected
# ===========================================================================

class TestBrowserToolsDisconnected:
    """All tools must degrade gracefully (return error dict) when offline."""

    async def test_get_active_tab_no_extension(self):
        tool = BrowserGetActiveTabTool()
        result = await tool.execute({})
        assert "error" in result
        assert "not connected" in result["error"].lower()

    async def test_get_page_title_no_extension(self):
        result = await BrowserGetPageTitleTool().execute({})
        assert "error" in result

    async def test_navigate_no_extension(self):
        result = await BrowserNavigateTool().execute({"url": "https://example.com"})
        assert "error" in result

    async def test_open_tab_no_extension(self):
        result = await BrowserOpenTabTool().execute({"url": "https://example.com"})
        assert "error" in result

    async def test_close_tab_no_extension(self):
        result = await BrowserCloseTabTool().execute({"tab_id": 99})
        assert "error" in result

    async def test_click_no_extension(self):
        result = await BrowserClickTool().execute({"selector": "#btn"})
        assert "error" in result

    async def test_fill_input_no_extension(self):
        result = await BrowserFillInputTool().execute(
            {"selector": "#inp", "value": "hello"}
        )
        assert "error" in result

    async def test_execute_script_disabled_by_default(self):
        """Script execution must be gated by ALLOW_SCRIPT_EXECUTION=true."""
        result = await BrowserExecuteScriptTool().execute({"code": "return 1+1"})
        assert "error" in result


# ===========================================================================
# 2. Tests WITH a live fake extension over real TCP WebSocket
# ===========================================================================

class TestBrowserToolsConnected:

    async def test_get_active_tab(self, live_server):
        mock = {"id": 1, "url": "https://example.com", "title": "Example"}
        ext = FakeExtension({"get_active_tab": mock})
        await ext.start(live_server)
        try:
            result = await BrowserGetActiveTabTool().execute({})
            assert result == mock, f"Got: {result!r}"
        finally:
            await ext.stop()

    async def test_get_page_title(self, live_server):
        mock = {"title": "TRACE Dashboard"}
        ext = FakeExtension({"get_page_title": mock})
        await ext.start(live_server)
        try:
            result = await BrowserGetPageTitleTool().execute({})
            assert result == mock
        finally:
            await ext.stop()

    async def test_navigate(self, live_server):
        mock = {"ok": True, "url": "https://github.com"}
        ext = FakeExtension({"navigate": mock})
        await ext.start(live_server)
        try:
            result = await BrowserNavigateTool().execute({"url": "https://github.com"})
            assert result == mock
        finally:
            await ext.stop()

    async def test_open_tab(self, live_server):
        mock = {"ok": True, "tab_id": 42}
        ext = FakeExtension({"open_tab": mock})
        await ext.start(live_server)
        try:
            result = await BrowserOpenTabTool().execute({"url": "https://python.org"})
            assert result == mock
        finally:
            await ext.stop()

    async def test_open_tab_active_flag_forwarded(self, live_server):
        """open_tab with active=False must forward that flag to the extension."""
        received: list[dict] = []

        class RecordingExtension(FakeExtension):
            async def _run(self, url: str) -> None:
                uri = f"{url}/ws/browser?secret={self.secret}"
                async with ws_connect(uri) as ws:
                    self._ws = ws
                    self._ready.set()
                    async for raw in ws:
                        msg = json.loads(raw)
                        received.append(msg.get("params", {}))
                        await ws.send(json.dumps({"id": msg["id"], "result": {"ok": True}}))

        ext = RecordingExtension({})
        await ext.start(live_server)
        try:
            await BrowserOpenTabTool().execute({"url": "https://x.com", "active": False})
            assert received, "Extension received no requests"
            assert received[-1].get("active") is False
        finally:
            await ext.stop()

    async def test_close_tab(self, live_server):
        mock = {"ok": True, "tab_id": 7}
        ext = FakeExtension({"close_tab": mock})
        await ext.start(live_server)
        try:
            result = await BrowserCloseTabTool().execute({"tab_id": 7})
            assert result == mock
        finally:
            await ext.stop()

    async def test_click(self, live_server):
        mock = {"ok": True}
        ext = FakeExtension({"click": mock})
        await ext.start(live_server)
        try:
            result = await BrowserClickTool().execute({"selector": "#submit"})
            assert result == mock
        finally:
            await ext.stop()

    async def test_click_element_not_found(self, live_server):
        """Extension error response should be surfaced as-is (no raise)."""
        mock = {"ok": False, "error": "Selector '#ghost' not found"}
        ext = FakeExtension({"click": mock})
        await ext.start(live_server)
        try:
            result = await BrowserClickTool().execute({"selector": "#ghost"})
            assert result == mock
        finally:
            await ext.stop()

    async def test_fill_input(self, live_server):
        mock = {"ok": True}
        ext = FakeExtension({"fill_input": mock})
        await ext.start(live_server)
        try:
            result = await BrowserFillInputTool().execute(
                {"selector": "#email", "value": "test@example.com"}
            )
            assert result == mock
        finally:
            await ext.stop()

    async def test_execute_script_disabled(self, live_server):
        """Script execution should fail even with extension connected if flag is off."""
        ext = FakeExtension({"execute_script": {"result": 2}})
        await ext.start(live_server)
        original = settings.allow_script_execution
        settings.allow_script_execution = False
        try:
            result = await BrowserExecuteScriptTool().execute({"code": "1+1"})
            assert "error" in result
            assert "disabled" in result["error"].lower()
        finally:
            settings.allow_script_execution = original
            await ext.stop()

    async def test_execute_script_enabled(self, live_server):
        """With flag on, the tool should relay to the extension."""
        mock = {"result": 42}
        ext = FakeExtension({"execute_script": mock})
        await ext.start(live_server)
        original = settings.allow_script_execution
        settings.allow_script_execution = True
        try:
            result = await BrowserExecuteScriptTool().execute({"code": "return 6*7"})
            assert result == mock
        finally:
            settings.allow_script_execution = original
            await ext.stop()

    async def test_tool_timeout(self, live_server):
        """Extension that never replies → tool returns a timeout error dict (no raise)."""

        class SilentExtension(FakeExtension):
            async def _run(self, url: str) -> None:
                uri = f"{url}/ws/browser?secret={self.secret}"
                async with ws_connect(uri) as ws:
                    self._ws = ws
                    self._ready.set()
                    # Accept requests but never reply
                    async for _ in ws:
                        await asyncio.sleep(9999)

        ext = SilentExtension({})
        await ext.start(live_server)
        try:
            # BrowserGetActiveTabTool has a 3 s timeout; give 6 s slack
            result = await asyncio.wait_for(
                BrowserGetActiveTabTool().execute({}), timeout=6.0
            )
            assert "error" in result
            assert "timed out" in result["error"].lower()
        finally:
            await ext.stop()

    async def test_wrong_secret_rejected(self, live_server):
        """A wrong secret should result in the channel staying disconnected."""
        ext = FakeExtension({"get_active_tab": {}}, secret="WRONG_SECRET")
        try:
            await ext.start(live_server)
        except Exception:
            pass  # websockets may raise on 4001 close
        try:
            channel = get_channel()
            assert not channel.is_connected
        finally:
            await ext.stop()

    async def test_concurrent_requests_correlation(self, live_server):
        """
        Two concurrent tool calls must each receive their own response —
        BrowserChannelManager correlation-ID tracking must not mix them.
        """
        mock_responses = {
            "get_active_tab": {"id": 1, "url": "https://a.com"},
            "get_page_title": {"title": "Page A"},
        }
        ext = FakeExtension(mock_responses)
        await ext.start(live_server)
        try:
            r_tab, r_title = await asyncio.gather(
                BrowserGetActiveTabTool().execute({}),
                BrowserGetPageTitleTool().execute({}),
            )
            assert r_tab == mock_responses["get_active_tab"]
            assert r_title == mock_responses["get_page_title"]
        finally:
            await ext.stop()


# ===========================================================================
# 3. Schema / describe() tests (no server needed)
# ===========================================================================

TOOL_CLASSES = [
    BrowserGetActiveTabTool,
    BrowserGetPageTitleTool,
    BrowserNavigateTool,
    BrowserOpenTabTool,
    BrowserCloseTabTool,
    BrowserClickTool,
    BrowserFillInputTool,
    BrowserExecuteScriptTool,
]


class TestBrowserToolSchemas:

    def test_all_tools_have_name(self):
        for cls in TOOL_CLASSES:
            assert cls().schema.name, f"{cls.__name__} has empty name"

    def test_all_tools_have_description(self):
        for cls in TOOL_CLASSES:
            assert cls().schema.description, f"{cls.__name__} has empty description"

    def test_all_tools_have_object_input_schema(self):
        for cls in TOOL_CLASSES:
            schema = cls().schema.input_schema
            assert isinstance(schema, dict) and schema.get("type") == "object", \
                f"{cls.__name__} bad input_schema"

    def test_risk_levels_valid(self):
        for cls in TOOL_CLASSES:
            assert cls().schema.risk_level in ("low", "medium", "high")

    def test_execute_script_is_high_risk(self):
        assert BrowserExecuteScriptTool().schema.risk_level == "high"

    def test_describe_returns_required_keys(self):
        for cls in TOOL_CLASSES:
            desc = cls().describe()
            for key in ("name", "description", "input_schema", "risk_level"):
                assert key in desc, f"{cls.__name__}.describe() missing '{key}'"

    def test_navigate_requires_url(self):
        assert "url" in BrowserNavigateTool().schema.input_schema.get("required", [])

    def test_open_tab_requires_url(self):
        assert "url" in BrowserOpenTabTool().schema.input_schema.get("required", [])

    def test_close_tab_requires_tab_id(self):
        assert "tab_id" in BrowserCloseTabTool().schema.input_schema.get("required", [])

    def test_click_requires_selector(self):
        assert "selector" in BrowserClickTool().schema.input_schema.get("required", [])

    def test_fill_input_requires_selector_and_value(self):
        required = BrowserFillInputTool().schema.input_schema.get("required", [])
        assert "selector" in required
        assert "value" in required

    def test_execute_script_requires_code(self):
        assert "code" in BrowserExecuteScriptTool().schema.input_schema.get("required", [])
