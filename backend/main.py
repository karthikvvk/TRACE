"""
main.py — Friday FastAPI application entry point.

Startup sequence:
  1. Init SQLite (create tables if not exist)
  2. Instantiate singletons (WorkingMemory, ToolRouter, Notifier)
  3. Mount all routers
  4. Configure CORS for the extension origin
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.db.init_db import init_db
from backend.memory.working import WorkingMemory
from backend.engine.router import ToolRouter
from backend.engine.notifier import Notifier
from backend.ws.manager import get_channel
from backend.routes import activity, tasks, memory, notify
from backend.routes import browser_ws, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    print("🗄️  Initialising database …")
    init_db()

    print("🧠  Wiring up singletons …")
    app.state.working_memory = WorkingMemory()
    app.state.tool_router = ToolRouter()
    app.state.notifier = Notifier(working_memory=app.state.working_memory)
    app.state.browser_channel = get_channel()  # module-level singleton

    print(f"✅  Friday backend ready → http://{settings.host}:{settings.port}")
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    print("👋  Friday shutting down.")


app = FastAPI(
    title="Friday",
    description="Local-first browser extension agent backend.",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Static UI ─────────────────────────────────────────────────────────────────
_UI_DIR = Path(__file__).parent / "ui"
if _UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=_UI_DIR, html=True), name="ui")

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.extension_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────────────────────
app.include_router(activity.router)
app.include_router(tasks.router)
app.include_router(memory.router)
app.include_router(notify.router)
app.include_router(browser_ws.router)
app.include_router(chat.router)


# ── Health & Meta ────────────────────────────────────────────────────────────
@app.get("/", tags=["meta"], include_in_schema=False)
async def root():
    """Redirect to the chat UI."""
    return RedirectResponse(url="/ui/chat.html")


@app.get("/tools", tags=["meta"])
async def list_tools(request_obj: None = None):
    """Return all registered tools and their schemas (MCP-compatible)."""
    from fastapi import Request
    # Access via app directly
    return {"tools": app.state.tool_router.list_tools()}


@app.get("/context", tags=["meta"])
async def get_context():
    """Return the current working memory context."""
    return app.state.working_memory.snapshot()


@app.get("/test-scrape", tags=["debug"])
async def test_scrape(url: str):
    """Directly test the extension scraper without going through the AI."""
    from backend.tools.browser_tool import BrowserScrapeUrlTool
    tool = BrowserScrapeUrlTool()
    result = await tool.execute({"url": url})
    return {"url": url, "result": result}
