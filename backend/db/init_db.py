"""
init_db.py — SQLite schema initialisation.

Called once at server startup. Safe to call multiple times (CREATE IF NOT EXISTS).
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from backend.config import settings


def init_db() -> None:
    """Create all tables if they don't exist yet."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_db() as conn:
        conn.executescript(
            """
            -- ─── Tasks ──────────────────────────────────────────────────────────────
            CREATE TABLE IF NOT EXISTS tasks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT    NOT NULL,
                description TEXT,
                due         TEXT,                          -- ISO-8601 datetime string
                source      TEXT    NOT NULL DEFAULT 'chat',   -- chat | inferred | calendar
                priority    TEXT    NOT NULL DEFAULT 'medium', -- low | medium | high
                tags        TEXT    DEFAULT '',            -- comma-separated
                status      TEXT    NOT NULL DEFAULT 'pending', -- pending | done | snoozed
                created_at  TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL
            );

            -- ─── Episodic Events ────────────────────────────────────────────────────
            CREATE TABLE IF NOT EXISTS episodic_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type      TEXT    NOT NULL,          -- navigation | search | form_submit | chat
                intent          TEXT,                      -- research | shopping | travel …
                category        TEXT,                      -- inferred topic
                time_bucket     TEXT,                      -- morning | afternoon | evening | night
                session_length_s INTEGER,
                metadata        TEXT,                      -- JSON string of extra k/v
                created_at      TEXT    NOT NULL
            );

            -- ─── Semantic / Preference Facts ────────────────────────────────────────
            CREATE TABLE IF NOT EXISTS semantic_facts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                key         TEXT    NOT NULL UNIQUE,       -- e.g. 'work_start_hour'
                value       TEXT    NOT NULL,
                confidence  REAL    NOT NULL DEFAULT 1.0,
                source      TEXT,                          -- inferred | explicit
                updated_at  TEXT    NOT NULL
            );

            -- ─── Notification Log ───────────────────────────────────────────────────
            CREATE TABLE IF NOT EXISTS notification_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id     INTEGER REFERENCES tasks(id) ON DELETE SET NULL,
                message     TEXT    NOT NULL,
                triggered_at TEXT   NOT NULL,
                dismissed   INTEGER NOT NULL DEFAULT 0     -- 0=pending, 1=dismissed
            );
            """
        )


@contextmanager
def get_db() -> sqlite3.Connection:
    """Context manager: yields a row-factory-enabled connection, commits on exit."""
    conn = sqlite3.connect(str(settings.db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
