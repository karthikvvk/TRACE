"""
working.py — Ephemeral in-process working memory.

Mirrors chrome.storage.session: survives for the lifetime of the server process only.
Holds the current active context: what task is in focus, what the user seems to be doing.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ActiveContext:
    task_id: int | None = None
    inferred_intent: str | None = None
    inferred_category: str | None = None
    last_url: str | None = None
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict[str, Any] = field(default_factory=dict)


class WorkingMemory:
    """
    Simple in-memory dict acting as ephemeral working memory.
    Designed to be a singleton instantiated once in main.py.
    """

    def __init__(self) -> None:
        self._context = ActiveContext()
        self._kv: dict[str, Any] = {}

    # ── Active context ──────────────────────────────────────────────────────

    def set_context(self, **kwargs: Any) -> None:
        """Update fields in the active context."""
        for k, v in kwargs.items():
            if hasattr(self._context, k):
                setattr(self._context, k, v)
        self._context.last_seen = datetime.now(timezone.utc)

    def get_context(self) -> dict[str, Any]:
        ctx = self._context.__dict__.copy()
        ctx["last_seen"] = ctx["last_seen"].isoformat()
        return ctx

    def clear_context(self) -> None:
        self._context = ActiveContext()

    # ── Generic key-value store ─────────────────────────────────────────────

    def set(self, key: str, value: Any) -> None:
        self._kv[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._kv.get(key, default)

    def delete(self, key: str) -> None:
        self._kv.pop(key, None)

    def snapshot(self) -> dict[str, Any]:
        return {"context": self.get_context(), "kv": dict(self._kv)}
