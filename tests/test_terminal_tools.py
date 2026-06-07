"""
test_terminal_tools.py — Programmatic tests for terminal PTY tools.

Tests run WITHOUT an agent. Each test directly instantiates the tool class
and calls execute(), or hits the FastAPI HTTP endpoints through TestClient.

Coverage:
  - CreateTerminalTool  → spawn a bash session
  - WriteTerminalTool   → send a command
  - ReadTerminalTool    → read buffered output (ANSI-stripped)
  - KillTerminalTool    → terminate session
  - HTTP endpoints      → /session/create, write, read, status, resize, kill, list
  - Edge cases          → dead-session writes, unknown-session reads, interrupt
"""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

# ── Import the FastAPI app and tool classes directly ──────────────────────────
from backend.tools.terminal_access import (
    app,
    CreateTerminalTool,
    WriteTerminalTool,
    ReadTerminalTool,
    KillTerminalTool,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Sync TestClient for the terminal relay FastAPI app."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _wait_for_output(client: TestClient, session_id: str, keyword: str,
                     timeout: float = 5.0, poll: float = 0.2) -> str:
    """
    Poll GET /session/{id}/read until `keyword` appears in the output
    or `timeout` seconds elapse. Returns the accumulated output text.
    """
    deadline = time.time() + timeout
    accumulated = ""
    while time.time() < deadline:
        r = client.get(f"/session/{session_id}/read")
        assert r.status_code == 200
        accumulated += r.json()["output"]
        if keyword.lower() in accumulated.lower():
            return accumulated
        time.sleep(poll)
    return accumulated


# ===========================================================================
# 1. HTTP endpoint tests (TestClient — synchronous)
# ===========================================================================

class TestTerminalHTTPEndpoints:

    def test_create_session_defaults(self, client):
        """POST /session/create with defaults should return a session_id and pid."""
        r = client.post("/session/create", json={})
        assert r.status_code == 201
        data = r.json()
        assert "session_id" in data
        assert "pid" in data
        assert data["command"] == "/bin/bash"
        # Clean up
        client.delete(f"/session/{data['session_id']}")

    def test_create_session_custom_command(self, client):
        """POST /session/create with sh should spawn sh."""
        r = client.post("/session/create", json={"command": "/bin/sh"})
        assert r.status_code == 201
        data = r.json()
        assert data["command"] == "/bin/sh"
        client.delete(f"/session/{data['session_id']}")

    def test_write_and_read(self, client):
        """Write `echo hello` then read output containing 'hello'."""
        sid = client.post("/session/create", json={}).json()["session_id"]
        try:
            w = client.post(f"/session/{sid}/write", json={"text": "echo hello\n"})
            assert w.status_code == 200
            assert w.json()["written"] > 0

            output = _wait_for_output(client, sid, "hello")
            assert "hello" in output
        finally:
            client.delete(f"/session/{sid}")

    def test_read_returns_bytes_field(self, client):
        """GET /session/{id}/read should return 'bytes' field."""
        sid = client.post("/session/create", json={}).json()["session_id"]
        try:
            time.sleep(0.3)
            r = client.get(f"/session/{sid}/read")
            assert r.status_code == 200
            assert "bytes" in r.json()
            assert "output" in r.json()
        finally:
            client.delete(f"/session/{sid}")

    def test_clear_buffer(self, client):
        """DELETE /session/{id}/read clears the buffer."""
        sid = client.post("/session/create", json={}).json()["session_id"]
        try:
            client.post(f"/session/{sid}/write", json={"text": "echo clear_me\n"})
            _wait_for_output(client, sid, "clear_me")

            clear_r = client.delete(f"/session/{sid}/read")
            assert clear_r.status_code == 200
            cleared = clear_r.json()["cleared_bytes"]
            assert cleared > 0

            # Buffer should now be empty (or near-empty)
            r2 = client.get(f"/session/{sid}/read")
            assert r2.json()["bytes"] == 0 or r2.json()["output"] == ""
        finally:
            client.delete(f"/session/{sid}")

    def test_session_status_alive(self, client):
        """GET /session/{id}/status for a live session should return alive=True."""
        sid = client.post("/session/create", json={}).json()["session_id"]
        try:
            r = client.get(f"/session/{sid}/status")
            assert r.status_code == 200
            data = r.json()
            assert data["alive"] is True
            assert data["session_id"] == sid
        finally:
            client.delete(f"/session/{sid}")

    def test_session_status_after_kill(self, client):
        """After DELETE /session/{id}, the session should no longer be listed."""
        sid = client.post("/session/create", json={}).json()["session_id"]
        client.delete(f"/session/{sid}")
        r = client.get(f"/session/{sid}/status")
        assert r.status_code == 404

    def test_list_sessions(self, client):
        """GET /sessions should include our active session."""
        sid = client.post("/session/create", json={}).json()["session_id"]
        try:
            r = client.get("/sessions")
            assert r.status_code == 200
            ids = [s["session_id"] for s in r.json()]
            assert sid in ids
        finally:
            client.delete(f"/session/{sid}")

    def test_write_to_unknown_session_returns_404(self, client):
        """Writing to a non-existent session should return 404."""
        r = client.post("/session/nonexistent/write", json={"text": "oops\n"})
        assert r.status_code == 404

    def test_read_unknown_session_returns_404(self, client):
        """Reading from a non-existent session should return 404."""
        r = client.get("/session/nonexistent/read")
        assert r.status_code == 404

    def test_status_unknown_session_returns_404(self, client):
        """Status for a non-existent session should return 404."""
        r = client.get("/session/nonexistent/status")
        assert r.status_code == 404

    def test_resize_session(self, client):
        """POST /session/{id}/resize should succeed with valid dimensions."""
        sid = client.post("/session/create", json={}).json()["session_id"]
        try:
            r = client.post(f"/session/{sid}/resize", json={"cols": 120, "rows": 30})
            assert r.status_code == 200
            data = r.json()
            assert data["cols"] == 120
            assert data["rows"] == 30
        finally:
            client.delete(f"/session/{sid}")

    def test_interrupt_session(self, client):
        """POST /session/{id}/interrupt should send ETX and not crash."""
        sid = client.post("/session/create", json={}).json()["session_id"]
        try:
            # Start a long-running command
            client.post(f"/session/{sid}/write", json={"text": "sleep 60\n"})
            time.sleep(0.3)
            r = client.post(f"/session/{sid}/interrupt")
            assert r.status_code == 200
            assert r.json()["sent"] == "SIGINT (ETX)"
        finally:
            client.delete(f"/session/{sid}")

    def test_multiline_output(self, client):
        """Multiple commands should accumulate output in the buffer."""
        sid = client.post("/session/create", json={}).json()["session_id"]
        try:
            client.post(f"/session/{sid}/write", json={"text": "echo line1\n"})
            client.post(f"/session/{sid}/write", json={"text": "echo line2\n"})
            client.post(f"/session/{sid}/write", json={"text": "echo line3\n"})
            output = _wait_for_output(client, sid, "line3", timeout=6)
            assert "line1" in output
            assert "line2" in output
            assert "line3" in output
        finally:
            client.delete(f"/session/{sid}")

    def test_env_variable_in_session(self, client):
        """Shell should support env variable expansion."""
        sid = client.post("/session/create", json={}).json()["session_id"]
        try:
            client.post(f"/session/{sid}/write",
                        json={"text": "MY_VAR=hello_world && echo $MY_VAR\n"})
            output = _wait_for_output(client, sid, "hello_world")
            assert "hello_world" in output
        finally:
            client.delete(f"/session/{sid}")

    def test_hex_decode_false(self, client):
        """GET /session/{id}/read?decode=false should return hex output."""
        sid = client.post("/session/create", json={}).json()["session_id"]
        try:
            client.post(f"/session/{sid}/write", json={"text": "echo abc\n"})
            time.sleep(0.5)
            r = client.get(f"/session/{sid}/read?decode=false")
            assert r.status_code == 200
            data = r.json()
            # Hex output should be a valid hex string
            hex_out = data["output"]
            assert all(c in "0123456789abcdef" for c in hex_out)
        finally:
            client.delete(f"/session/{sid}")


# ===========================================================================
# 2. MCP Tool class tests (async — direct .execute() calls)
# ===========================================================================

class TestTerminalMCPTools:
    """
    Tests that exercise the BaseTool subclasses directly, bypassing HTTP.
    These simulate exactly what the agent calls during a conversation.
    """

    async def test_create_tool_returns_session_id(self):
        tool = CreateTerminalTool()
        result = await tool.execute({})
        assert "session_id" in result
        assert "pid" in result

        # Clean up via KillTool
        kill_tool = KillTerminalTool()
        kill_result = await kill_tool.execute({"session_id": result["session_id"]})
        assert "killed" in kill_result

    async def test_write_and_read_tool_pipeline(self):
        """Full agent-style pipeline: create → write → read → kill."""
        create = CreateTerminalTool()
        write = WriteTerminalTool()
        read = ReadTerminalTool()
        kill = KillTerminalTool()

        # 1. Create session
        session = await create.execute({"command": "/bin/bash"})
        sid = session["session_id"]

        try:
            # 2. Write command
            w = await write.execute({"session_id": sid, "text": "echo trace_test_output\n"})
            assert w["written"] > 0

            # 3. Read — ReadTerminalTool sleeps 0.3s internally, no extra sleep needed
            r = await read.execute({"session_id": sid})
            assert "trace_test_output" in r["output"], \
                f"Expected 'trace_test_output' in output, got: {r['output']!r}"
        finally:
            await kill.execute({"session_id": sid})

    async def test_read_tool_strips_ansi(self):
        """ReadTerminalTool should return plain text, no ANSI escape codes."""
        create = CreateTerminalTool()
        write = WriteTerminalTool()
        read = ReadTerminalTool()
        kill = KillTerminalTool()

        session = await create.execute({})
        sid = session["session_id"]

        try:
            # Force colored output using tput
            await write.execute({"session_id": sid,
                                 "text": "printf '\\033[0;31mREDTEXT\\033[0m\\n'\n"})
            result = await read.execute({"session_id": sid})
            output = result["output"]
            # ANSI sequences should be stripped
            assert "\x1b[" not in output
            assert "\x1b]" not in output
            # But the plain text content should remain
            assert "REDTEXT" in output
        finally:
            await kill.execute({"session_id": sid})

    async def test_read_clears_buffer(self):
        """ReadTerminalTool clears the buffer; a second read should be empty."""
        create = CreateTerminalTool()
        write = WriteTerminalTool()
        read = ReadTerminalTool()
        kill = KillTerminalTool()

        session = await create.execute({})
        sid = session["session_id"]

        try:
            await write.execute({"session_id": sid, "text": "echo once_only\n"})
            r1 = await read.execute({"session_id": sid})
            assert "once_only" in r1["output"]

            # Second immediate read should return nothing new
            await asyncio.sleep(0.1)
            r2 = await read.execute({"session_id": sid})
            assert "once_only" not in r2["output"]
        finally:
            await kill.execute({"session_id": sid})

    async def test_sequential_commands(self):
        """Multiple write→read cycles should each return the right output."""
        create = CreateTerminalTool()
        write = WriteTerminalTool()
        read = ReadTerminalTool()
        kill = KillTerminalTool()

        session = await create.execute({})
        sid = session["session_id"]

        try:
            for i in range(3):
                await write.execute({"session_id": sid,
                                     "text": f"echo cycle_{i}\n"})
                r = await read.execute({"session_id": sid})
                assert f"cycle_{i}" in r["output"], \
                    f"Cycle {i}: expected 'cycle_{i}' in {r['output']!r}"
        finally:
            await kill.execute({"session_id": sid})

    async def test_kill_tool_removes_session(self):
        """After KillTerminalTool, the session must not exist."""
        create = CreateTerminalTool()
        kill = KillTerminalTool()

        session = await create.execute({})
        sid = session["session_id"]

        kill_result = await kill.execute({"session_id": sid})
        assert kill_result["killed"] == sid

        # Verify via read tool that session is gone
        read = ReadTerminalTool()
        # ReadTerminalTool wraps the underlying HTTP call; it will raise HTTPException
        # which gets propagated as an exception or caught — either is acceptable.
        with pytest.raises(Exception):
            await read.execute({"session_id": sid})

    async def test_create_with_nondefault_command(self):
        """CreateTerminalTool should accept a custom command."""
        create = CreateTerminalTool()
        kill = KillTerminalTool()

        result = await create.execute({"command": "/bin/sh"})
        assert result["command"] == "/bin/sh"
        await kill.execute({"session_id": result["session_id"]})

    async def test_tool_describe(self):
        """All tool schemas should be valid and serialisable."""
        for ToolClass in (CreateTerminalTool, WriteTerminalTool,
                          ReadTerminalTool, KillTerminalTool):
            tool = ToolClass()
            desc = tool.describe()
            assert "name" in desc
            assert "description" in desc
            assert "input_schema" in desc
            assert isinstance(desc["risk_level"], str)
