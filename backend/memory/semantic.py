"""
semantic.py — Stable user facts and preferences (SQLite-backed).

Examples of semantic facts:
  - work_start_hour: 9
  - preferred_search_engine: google
  - common_category: development
  - contact_A_relation: colleague

Keys are strings; values are JSON-serialisable scalars or small objects.
Confidence degrades over time if not reinforced.
"""

import json
from datetime import datetime, timezone
from typing import Any

from backend.db.init_db import get_db


class SemanticMemory:
    """CRUD interface for the semantic_facts table."""

    async def set(self, key: str, value: Any, source: str = "inferred", confidence: float = 1.0) -> None:
        """Upsert a user fact."""
        now = datetime.now(timezone.utc).isoformat()
        serialized = json.dumps(value)

        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO semantic_facts (key, value, confidence, source, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    confidence = excluded.confidence,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (key, serialized, confidence, source, now),
            )

    async def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a fact by key. Returns default if not found."""
        with get_db() as conn:
            row = conn.execute(
                "SELECT value FROM semantic_facts WHERE key = ?", (key,)
            ).fetchone()

        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError):
            return row["value"]

    async def all_facts(self) -> dict[str, Any]:
        """Return all facts as a plain dict."""
        with get_db() as conn:
            rows = conn.execute("SELECT key, value FROM semantic_facts").fetchall()

        result = {}
        for row in rows:
            try:
                result[row["key"]] = json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                result[row["key"]] = row["value"]
        return result

    async def delete(self, key: str) -> None:
        with get_db() as conn:
            conn.execute("DELETE FROM semantic_facts WHERE key = ?", (key,))

    async def reinforce(self, key: str, delta: float = 0.05) -> None:
        """Increase confidence of a known fact (up to 1.0)."""
        with get_db() as conn:
            conn.execute(
                """
                UPDATE semantic_facts
                SET confidence = MIN(1.0, confidence + ?),
                    updated_at = ?
                WHERE key = ?
                """,
                (delta, datetime.now(timezone.utc).isoformat(), key),
            )
