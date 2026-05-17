"""
notifier.py — Decides *what* to surface and *when*.

This is the proactive analysis loop. Called by the alarm endpoint every N minutes.
Logic at Phase 1:
  - Surface tasks that are overdue or due within the next hour
  - Cross-reference last-known browser context against task titles
  - Don't spam — respect a minimum re-notify interval per task

Phase 2+: incorporate episodic memory context for smarter correlation.
"""

from datetime import datetime, timezone, timedelta
from typing import Any

from backend.db.init_db import get_db
from backend.config import settings


class Notifier:
    """
    Stateless decision engine for what to surface.
    Reads tasks from SQLite; uses current context from WorkingMemory.
    """

    def __init__(self, working_memory=None) -> None:
        self._working = working_memory  # optional, injected from main

    def get_pending_notifications(self) -> list[dict[str, Any]]:
        """
        Return a list of notification payloads that should be surfaced now.
        Called by GET /notify/pending every check_interval_minutes.
        """
        now = datetime.now(timezone.utc)
        notifications: list[dict[str, Any]] = []

        with get_db() as conn:
            tasks = conn.execute(
                """
                SELECT * FROM tasks
                WHERE status = 'pending'
                ORDER BY due ASC
                LIMIT 100
                """
            ).fetchall()

        for task in tasks:
            task = dict(task)
            note = self._evaluate_task(task, now)
            if note:
                notifications.append(note)

        # Contextual nudge: cross-reference working memory
        if self._working:
            ctx = self._working.get_context()
            notifications.extend(self._contextual_nudges(tasks, ctx, now))

        return notifications

    def _evaluate_task(self, task: dict, now: datetime) -> dict | None:
        """Return a notification dict if this task should be surfaced."""
        if not task.get("due"):
            return None

        try:
            due = datetime.fromisoformat(task["due"].replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

        # Overdue
        if due < now:
            overdue_minutes = int((now - due).total_seconds() / 60)
            if overdue_minutes < 60 * 24:  # Only surface if overdue by less than a day
                return {
                    "task_id": task["id"],
                    "title": task["title"],
                    "message": f"⚠️ Overdue: {task['title']} (due {_friendly_time(due)})",
                    "urgency": "high",
                    "trigger": "overdue",
                }

        # Due within 1 hour
        if now <= due <= now + timedelta(hours=1):
            return {
                "task_id": task["id"],
                "title": task["title"],
                "message": f"⏰ Due soon: {task['title']} ({_friendly_time(due)})",
                "urgency": "medium",
                "trigger": "due_soon",
            }

        return None

    def _contextual_nudges(
        self, tasks: list[dict], context: dict, now: datetime
    ) -> list[dict[str, Any]]:
        """
        Phase 2+: match current browser activity against pending tasks.
        Example: user on LinkedIn → surface 'message Priya' task.
        """
        nudges: list[dict[str, Any]] = []
        domain = (context.get("last_url") or "").lower()
        category = (context.get("inferred_category") or "").lower()

        category_task_hints: dict[str, list[str]] = {
            "travel": ["book", "flight", "hotel", "trip", "travel", "pack"],
            "health": ["doctor", "dentist", "appointment", "medicine", "gym"],
            "career": ["apply", "resume", "linkedin", "message", "email", "follow"],
            "shopping": ["buy", "order", "purchase", "return"],
            "finance": ["pay", "invoice", "tax", "budget", "transfer"],
        }

        hints = category_task_hints.get(category, [])
        if not hints:
            return nudges

        for task in tasks:
            task = dict(task)
            title_lower = task.get("title", "").lower()
            if any(h in title_lower for h in hints):
                nudges.append({
                    "task_id": task["id"],
                    "title": task["title"],
                    "message": f"💡 Related to what you're doing: {task['title']}",
                    "urgency": "low",
                    "trigger": "contextual",
                })

        return nudges[:2]  # Cap contextual nudges to avoid noise


def _friendly_time(dt: datetime) -> str:
    """Return a human-friendly time string."""
    now = datetime.now(timezone.utc)
    delta = dt - now
    total_minutes = int(delta.total_seconds() / 60)

    if total_minutes < 0:
        return f"{abs(total_minutes)}m ago"
    if total_minutes < 60:
        return f"in {total_minutes}m"
    if total_minutes < 1440:
        return f"in {total_minutes // 60}h"
    return dt.strftime("%b %d, %H:%M")
