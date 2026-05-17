"""
routes/notify.py — Proactive notification endpoint.

Called by the extension's alarm loop every 10 minutes.
Returns items that Friday thinks should be surfaced right now.
"""

from fastapi import APIRouter, Request
from typing import Any

router = APIRouter(prefix="/notify", tags=["notify"])


@router.get("/pending")
async def get_pending(request: Request) -> dict[str, Any]:
    """
    Return pending notification items.
    The notifier reads tasks + working memory context to decide what to surface.
    """
    notifier = request.app.state.notifier
    items = notifier.get_pending_notifications()
    return {"items": items, "count": len(items)}


@router.post("/dismiss/{task_id}")
async def dismiss(task_id: int) -> dict[str, Any]:
    """Log that the user dismissed a notification for this task."""
    from backend.db.init_db import get_db
    from backend.utils.time_utils import iso_now

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO notification_log (task_id, message, triggered_at, dismissed)
            VALUES (?, ?, ?, 1)
            """,
            (task_id, "dismissed", iso_now()),
        )
    return {"dismissed": True, "task_id": task_id}


@router.get("/log")
async def notification_log(limit: int = 20) -> dict[str, Any]:
    """Return recent notification history."""
    from backend.db.init_db import get_db

    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM notification_log ORDER BY triggered_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    return {"log": [dict(r) for r in rows]}
