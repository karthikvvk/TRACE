"""
Terminal Relay Server
---------------------
Manages pseudo-terminal (PTY) sessions via a FastAPI HTTP interface.

Endpoints:
  POST /session/create            - Spawn a new PTY session
  DELETE /session/{id}            - Kill a session
  POST /session/{id}/write        - Send text/keystrokes to the terminal
  POST /session/{id}/interrupt    - Send SIGINT (Ctrl+C)
  GET  /session/{id}/read         - Read all buffered output so far
  GET  /session/{id}/status       - Check if the process is still running
  GET  /sessions                  - List all active sessions
"""

import asyncio
import os
import pty
import fcntl
import signal
import select
import threading
import termios
import struct
import uuid
from typing import Optional

from backend.tools.base import BaseTool, ToolSchema

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Terminal Relay Server")

# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

sessions: dict[str, dict] = {}
sessions_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CreateRequest(BaseModel):
    command: str = "/bin/bash"
    cols: int = 220
    rows: int = 50


class WriteRequest(BaseModel):
    text: str


class ResizeRequest(BaseModel):
    cols: int
    rows: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_winsize(fd: int, rows: int, cols: int) -> None:
    """Set the terminal window size."""
    s = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, s)


def _make_nonblocking(fd: int) -> None:
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


def _drain(fd: int, buf: bytearray) -> None:
    """Read everything currently available from fd into buf."""
    while True:
        r, _, _ = select.select([fd], [], [], 0)
        if not r:
            break
        try:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            buf.extend(chunk)
        except (OSError, BlockingIOError):
            break


def _reader_thread(session_id: str) -> None:
    """Background thread: continuously drain PTY master into the output buffer."""
    with sessions_lock:
        session = sessions.get(session_id)
    if session is None:
        return

    master_fd = session["master_fd"]
    buf: bytearray = session["output_buf"]
    buf_lock: threading.Lock = session["buf_lock"]

    while True:
        with sessions_lock:
            if session_id not in sessions:
                break
            alive = session.get("alive", True)
        if not alive:
            break

        r, _, _ = select.select([master_fd], [], [], 0.1)
        if r:
            try:
                chunk = os.read(master_fd, 4096)
                if chunk:
                    with buf_lock:
                        buf.extend(chunk)
                else:
                    break
            except OSError:
                break

    # Mark dead
    with sessions_lock:
        if session_id in sessions:
            sessions[session_id]["alive"] = False


def _get_session(session_id: str) -> dict:
    with sessions_lock:
        session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/session/create", status_code=201)
def create_session(req: CreateRequest):
    """Spawn a new shell (or any command) in a PTY session."""
    session_id = str(uuid.uuid4())[:8]

    master_fd, slave_fd = pty.openpty()
    _set_winsize(master_fd, req.rows, req.cols)
    _make_nonblocking(master_fd)

    pid = os.fork()
    if pid == 0:
        # Child process: become the slave side of the PTY
        os.setsid()
        fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
        for fd in (0, 1, 2):
            os.dup2(slave_fd, fd)
        os.close(slave_fd)
        os.close(master_fd)
        os.execvp(req.command.split()[0], req.command.split())
        os._exit(1)

    # Parent process
    os.close(slave_fd)

    output_buf: bytearray = bytearray()
    buf_lock = threading.Lock()

    session = {
        "id": session_id,
        "pid": pid,
        "master_fd": master_fd,
        "output_buf": output_buf,
        "buf_lock": buf_lock,
        "alive": True,
        "command": req.command,
    }

    with sessions_lock:
        sessions[session_id] = session

    # Start background reader
    t = threading.Thread(target=_reader_thread, args=(session_id,), daemon=True)
    t.start()

    return {"session_id": session_id, "pid": pid, "command": req.command}


@app.post("/session/{session_id}/write")
def write_to_session(session_id: str, req: WriteRequest):
    """Send text/keystrokes to the terminal (raw bytes, supports escape sequences)."""
    session = _get_session(session_id)
    if not session["alive"]:
        raise HTTPException(status_code=410, detail="Session is no longer alive")
    data = req.text.encode("utf-8")
    total = len(data)
    sent = 0
    while sent < total:
        n = os.write(session["master_fd"], data[sent:])
        sent += n
    return {"written": sent}


@app.post("/session/{session_id}/interrupt")
def interrupt_session(session_id: str):
    """Send SIGINT (Ctrl+C) to the foreground process group."""
    session = _get_session(session_id)
    # Writing ETX (0x03) into the PTY is more reliable than kill() for shells
    try:
        os.write(session["master_fd"], b"\x03")
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"sent": "SIGINT (ETX)"}


@app.get("/session/{session_id}/read")
def read_from_session(session_id: str, decode: bool = True):
    """
    Return all buffered output accumulated so far.
    Use ?decode=false to get raw hex if the output is binary.
    The buffer is NOT cleared — call /read repeatedly to poll.
    """
    session = _get_session(session_id)
    with session["buf_lock"]:
        snapshot = bytes(session["output_buf"])

    if decode:
        text = snapshot.decode("utf-8", errors="replace")
        return {"output": text, "bytes": len(snapshot)}
    else:
        return {"output": snapshot.hex(), "bytes": len(snapshot)}


@app.delete("/session/{session_id}/read")
def clear_read_buffer(session_id: str):
    """Clear the output buffer (so next /read starts fresh)."""
    session = _get_session(session_id)
    with session["buf_lock"]:
        cleared = len(session["output_buf"])
        session["output_buf"].clear()
    return {"cleared_bytes": cleared}


@app.get("/session/{session_id}/status")
def session_status(session_id: str):
    """Check if the underlying process is still running."""
    session = _get_session(session_id)
    pid = session["pid"]

    exit_code = None
    alive = session["alive"]

    if alive:
        try:
            result = os.waitpid(pid, os.WNOHANG)
            if result[0] != 0:
                exit_code = result[1]
                with sessions_lock:
                    sessions[session_id]["alive"] = False
                alive = False
        except ChildProcessError:
            alive = False

    return {
        "session_id": session_id,
        "pid": pid,
        "alive": alive,
        "exit_code": exit_code,
        "command": session["command"],
        "buffered_bytes": len(session["output_buf"]),
    }


@app.post("/session/{session_id}/resize")
def resize_session(session_id: str, req: ResizeRequest):
    """Resize the terminal window (TIOCSWINSZ)."""
    session = _get_session(session_id)
    try:
        _set_winsize(session["master_fd"], req.rows, req.cols)
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"cols": req.cols, "rows": req.rows}


@app.delete("/session/{session_id}")
def kill_session(session_id: str):
    """Kill the session process and clean up."""
    session = _get_session(session_id)
    pid = session["pid"]
    try:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    except (ProcessLookupError, ChildProcessError):
        pass
    try:
        os.close(session["master_fd"])
    except OSError:
        pass
    with sessions_lock:
        sessions.pop(session_id, None)
    return {"killed": session_id}


@app.post("/system/update")
def system_update():
    """Trigger a system update (pacman -Syy) in a new PTY session."""
    import time
    req = CreateRequest(command="/bin/bash")
    session_data = create_session(req)
    session_id = session_data["session_id"]
    
    # Wait for bash prompt to load before sending the command
    time.sleep(0.5)
    
    command = "pkexec env DISPLAY=$DISPLAY XAUTHORITY=$XAUTHORITY pacman -Syy\n"
    write_req = WriteRequest(text=command)
    write_to_session(session_id, write_req)
    
    return {"session_id": session_id, "message": "System update started"}


@app.get("/sessions")
def list_sessions():
    """List all sessions and their live status."""
    with sessions_lock:
        return [
            {
                "session_id": sid,
                "pid": s["pid"],
                "alive": s["alive"],
                "command": s["command"],
                "buffered_bytes": len(s["output_buf"]),
            }
            for sid, s in sessions.items()
        ]


# ---------------------------------------------------------------------------
# MCP Tool Wrappers
# ---------------------------------------------------------------------------

import re as _re
import time as _time

_ANSI_ESCAPE = _re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]|\x1B\][^\x07]*\x07|\x1B[^\[\]]')

def _strip_ansi(text: str) -> str:
    """Remove ANSI/VT100 escape sequences so the model sees plain text."""
    return _ANSI_ESCAPE.sub('', text)

class CreateTerminalTool(BaseTool):
    schema = ToolSchema(
        name="create_terminal_session",
        description="Spawn a new shell or command in a PTY session.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "default": "/bin/bash"}
            }
        },
        risk_level="medium"
    )

    async def execute(self, params: dict) -> dict:
        req = CreateRequest(command=params.get("command", "/bin/bash"))
        return create_session(req)


class WriteTerminalTool(BaseTool):
    schema = ToolSchema(
        name="write_to_terminal",
        description=(
            "Send text or keystrokes to an active terminal session. "
            "Always append \\n to execute a command. "
            "CRITICAL SECURITY RULE: NEVER use `sudo` or ask the user for their password. "
            "If a command requires elevated privileges, prefix it with `pkexec env DISPLAY=$DISPLAY XAUTHORITY=$XAUTHORITY ` "
            "(e.g., `pkexec env DISPLAY=$DISPLAY XAUTHORITY=$XAUTHORITY pacman -Syu`). This will securely prompt the user via a GUI popup. "
            "After writing, call read_from_terminal to see the output."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "text": {"type": "string"}
            },
            "required": ["session_id", "text"]
        },
        risk_level="medium"
    )

    async def execute(self, params: dict) -> dict:
        req = WriteRequest(text=params["text"])
        return write_to_session(params["session_id"], req)


class ReadTerminalTool(BaseTool):
    schema = ToolSchema(
        name="read_from_terminal",
        description=(
            "Read new output from a terminal session since the last read call. "
            "The buffer is cleared after each read, so subsequent calls return only NEW output. "
            "Always call this after write_to_terminal to see command results. "
            "If you see a password prompt, you made a mistake by using sudo. Kill the session and use pkexec instead."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"}
            },
            "required": ["session_id"]
        },
        risk_level="low"
    )

    async def execute(self, params: dict) -> dict:
        sid = params["session_id"]
        # Give the PTY a moment to produce output after the last write
        await asyncio.sleep(0.3)
        result = read_from_session(sid, decode=True)
        # Clear the buffer so the next read returns only NEW output
        clear_read_buffer(sid)
        # Strip ANSI/VT100 escape sequences so the model sees plain text
        clean_text = _strip_ansi(result.get("output", ""))
        return {"output": clean_text, "bytes": result.get("bytes", 0)}


class KillTerminalTool(BaseTool):
    schema = ToolSchema(
        name="kill_terminal_session",
        description="Kill an active terminal session.",
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"}
            },
            "required": ["session_id"]
        },
        risk_level="medium"
    )

    async def execute(self, params: dict) -> dict:
        return kill_session(params["session_id"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8765, reload=False)
