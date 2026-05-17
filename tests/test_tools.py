"""
tests/test_tools.py — Unit tests for Friday's core tool layer.

Run with:
    pytest tests/ -v

Tests cover:
  - Task CRUD tools (create, get, update, delete)
  - Search intent extraction
  - Intent classifier (URL rules, text rules)
  - Anonymizer (PII stripping)
  - Time bucket utility
"""

import asyncio
import os
import pytest
import pytest_asyncio
from pathlib import Path

# ── Point at a fresh test database ───────────────────────────────────────────
os.environ["DB_PATH"] = str(Path(__file__).parent / "test_friday.db")
os.environ["USE_PRESIDIO"] = "false"

# These imports must come AFTER the env override
from backend.db.init_db import init_db, get_db
from backend.tools.task_tool import CreateTaskTool, GetTasksTool, UpdateTaskTool, DeleteTaskTool
from backend.tools.search_tool import extract_query_from_url, ExtractSearchIntentTool
from backend.engine.intent import classify_intent, classify_url_intent, classify_text_intent
from backend.privacy.anonymizer import strip_pii, anonymize
from backend.utils.time_utils import time_bucket_for_hour


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def setup_db():
    """Create fresh tables before each test, wipe after."""
    init_db()
    yield
    # Teardown: drop all rows (keep schema)
    with get_db() as conn:
        conn.execute("DELETE FROM tasks")
        conn.execute("DELETE FROM episodic_events")
        conn.execute("DELETE FROM semantic_facts")
        conn.execute("DELETE FROM notification_log")


# ── Task tool tests ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_task_minimal():
    tool = CreateTaskTool()
    result = await tool.execute({"title": "Call Raj", "source": "chat"})
    assert "task_id" in result
    assert result["task_id"] > 0


@pytest.mark.asyncio
async def test_create_task_rejects_empty_title():
    tool = CreateTaskTool()
    with pytest.raises(ValueError, match="empty"):
        await tool.execute({"title": "  ", "source": "chat"})


@pytest.mark.asyncio
async def test_create_and_get_tasks():
    create = CreateTaskTool()
    get = GetTasksTool()

    await create.execute({"title": "Task A", "source": "chat", "priority": "high"})
    await create.execute({"title": "Task B", "source": "inferred"})

    result = await get.execute({"status": "pending"})
    titles = [t["title"] for t in result["tasks"]]
    assert "Task A" in titles
    assert "Task B" in titles


@pytest.mark.asyncio
async def test_update_task_status():
    create = CreateTaskTool()
    update = UpdateTaskTool()
    get = GetTasksTool()

    created = await create.execute({"title": "Fix bug", "source": "chat"})
    task_id = created["task_id"]

    await update.execute({"task_id": task_id, "status": "done"})

    done = await get.execute({"status": "done"})
    assert any(t["id"] == task_id for t in done["tasks"])


@pytest.mark.asyncio
async def test_update_requires_task_id():
    tool = UpdateTaskTool()
    with pytest.raises(ValueError, match="task_id"):
        await tool.execute({"status": "done"})


@pytest.mark.asyncio
async def test_delete_task():
    create = CreateTaskTool()
    delete = DeleteTaskTool()
    get = GetTasksTool()

    created = await create.execute({"title": "Delete me", "source": "chat"})
    task_id = created["task_id"]

    result = await delete.execute({"task_id": task_id})
    assert result["deleted"] is True

    remaining = await get.execute({"status": "all"})
    assert not any(t["id"] == task_id for t in remaining["tasks"])


# ── Search intent tests ───────────────────────────────────────────────────────

def test_extract_google_search():
    result = extract_query_from_url("https://www.google.com/search?q=best+python+frameworks")
    assert result is not None
    assert result["engine"] == "google"
    assert "python" in result["query"]
    assert result["category"] == "development"


def test_extract_youtube_search():
    result = extract_query_from_url("https://www.youtube.com/results?search_query=learn+react")
    assert result is not None
    assert result["engine"] == "youtube"
    assert result["category"] == "learning"


def test_extract_non_search_url():
    result = extract_query_from_url("https://www.example.com/about")
    assert result is None


def test_extract_flight_search():
    result = extract_query_from_url("https://www.google.com/search?q=cheap+flights+to+goa")
    assert result["category"] == "travel"


@pytest.mark.asyncio
async def test_search_intent_tool():
    tool = ExtractSearchIntentTool()
    result = await tool.execute({"url": "https://duckduckgo.com/?q=dentist+near+me"})
    assert result["category"] == "health"
    assert result["engine"] == "duckduckgo"


# ── Intent classifier tests ───────────────────────────────────────────────────

def test_classify_navigation_github():
    result = classify_intent({"type": "navigation", "url": "https://github.com/search?q=fastapi"})
    assert result["intent"] == "development"
    assert "time_bucket" in result


def test_classify_task_text():
    result = classify_text_intent("remind me to call the doctor tomorrow")
    assert result["is_task"] is True
    assert result["intent"] == "task_creation"


def test_classify_non_task_text():
    result = classify_text_intent("the weather looks nice today")
    assert result["is_task"] is False


def test_classify_linkedin():
    result = classify_url_intent("https://www.linkedin.com/in/someone")
    assert result["intent"] == "career"


def test_time_bucket_morning():
    assert time_bucket_for_hour(9) == "morning"


def test_time_bucket_night():
    assert time_bucket_for_hour(2) == "night"


def test_time_bucket_evening():
    assert time_bucket_for_hour(19) == "evening"


# ── Anonymizer tests ──────────────────────────────────────────────────────────

def test_strip_email():
    result = strip_pii("Contact me at john.doe@example.com for details.")
    assert "john.doe@example.com" not in result
    assert "[EMAIL]" in result


def test_strip_phone():
    result = strip_pii("Call me at +91 98765 43210 anytime.")
    assert "98765" not in result


def test_strip_ip():
    result = strip_pii("Server running at 192.168.1.100")
    assert "192.168.1.100" not in result
    assert "[IP]" in result


def test_anonymize_navigation_event():
    raw = {
        "type": "navigation",
        "url": "https://www.google.com/search?q=cheap+hotels",
        "timestamp": 1700000000000,
    }
    clean = anonymize(raw)
    # Should classify as travel intent
    assert clean["category"] == "travel"
    # Raw URL must not be stored
    assert "url" not in clean or "google.com" not in clean.get("url", "")


def test_anonymize_strips_email_from_text():
    raw = {
        "type": "chat",
        "text": "send an email to jane@company.com about the meeting",
        "timestamp": 1700000000000,
    }
    clean = anonymize(raw)
    assert "jane@company.com" not in clean.get("text", "")
