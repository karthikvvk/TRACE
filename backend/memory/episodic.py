"""
episodic.py — Timestamped semantic event log (IndexedDB analogue, stored in SQLite).

Events are written here after PII stripping. They are never raw — always semantic.
"""

import json
from datetime import datetime, timezone
from typing import Any

from backend.db.init_db import get_db
from backend.utils.time_utils import time_bucket_for_hour


class EpisodicMemory:
    """
    Manages the episodic_events table.
    All writes go through log(); all reads go through query().
    """

    async def log(self, params: dict) -> int:
        """Persist a semantic event. Returns the new event ID."""
        now = datetime.now(timezone.utc)
        bucket = params.get("time_bucket") or time_bucket_for_hour(now.hour)
        metadata = params.get("metadata") or {}

        with get_db() as conn:
            cur = conn.execute(
                """
                INSERT INTO episodic_events
                    (event_type, intent, category, time_bucket, session_length_s, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    params.get("event_type", "unknown"),
                    params.get("intent"),
                    params.get("category"),
                    bucket,
                    params.get("session_length_s"),
                    json.dumps(metadata),
                    now.isoformat(),
                ),
            )
            return cur.lastrowid

    async def query(
        self,
        keyword: str | None = None,
        intent: str | None = None,
        time_bucket: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Simple keyword + filter query over episodic events.
        No vector search at MVP — recency + keyword matching is sufficient.
        """
        clauses, values = [], []

        if intent:
            clauses.append("intent = ?")
            values.append(intent)
        if time_bucket:
            clauses.append("time_bucket = ?")
            values.append(time_bucket)
        if keyword:
            clauses.append("(category LIKE ? OR metadata LIKE ? OR event_type LIKE ?)")
            like = f"%{keyword}%"
            values.extend([like, like, like])

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        values.append(min(limit, 200))

        with get_db() as conn:
            rows = conn.execute(
                f"SELECT * FROM episodic_events {where} ORDER BY created_at DESC LIMIT ?",
                values,
            ).fetchall()

        result = []
        for row in rows:
            d = dict(row)
            try:
                d["metadata"] = json.loads(d["metadata"] or "{}")
            except (json.JSONDecodeError, TypeError):
                d["metadata"] = {}
            result.append(d)

        return result

    async def recent(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the n most recent events."""
        return await self.query(limit=n)
