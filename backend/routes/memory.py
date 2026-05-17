"""
routes/memory.py — Query endpoints for episodic and semantic memory.
"""

from fastapi import APIRouter
from typing import Any

from backend.memory.episodic import EpisodicMemory
from backend.memory.semantic import SemanticMemory

router = APIRouter(prefix="/memory", tags=["memory"])
_episodic = EpisodicMemory()
_semantic = SemanticMemory()


@router.get("/events")
async def query_events(
    keyword: str | None = None,
    intent: str | None = None,
    time_bucket: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Query episodic events with optional filters."""
    events = await _episodic.query(
        keyword=keyword,
        intent=intent,
        time_bucket=time_bucket,
        limit=limit,
    )
    return {"events": events, "count": len(events)}


@router.get("/events/recent")
async def recent_events(n: int = 10) -> dict[str, Any]:
    """Return the N most recent episodic events."""
    events = await _episodic.recent(n=n)
    return {"events": events}


@router.get("/facts")
async def all_facts() -> dict[str, Any]:
    """Return all semantic user facts."""
    facts = await _semantic.all_facts()
    return {"facts": facts}


@router.get("/facts/{key}")
async def get_fact(key: str) -> dict[str, Any]:
    """Return a single semantic fact by key."""
    value = await _semantic.get(key)
    return {"key": key, "value": value}


@router.put("/facts/{key}")
async def set_fact(key: str, value: str, source: str = "explicit") -> dict[str, Any]:
    """Upsert a semantic fact. Value is a JSON-serialisable string."""
    await _semantic.set(key, value, source=source, confidence=1.0)
    return {"key": key, "status": "updated"}
