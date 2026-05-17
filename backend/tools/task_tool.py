"""
task_tool.py — CRUD operations on the tasks table.

These are MCP-shaped tools. Each maps to one operation and one route.
The execute() method is called by the router; the routes call execute() directly.
"""

import sqlite3
from datetime import datetime
from typing import Any

from .base import BaseTool, ToolSchema
from backend.db.init_db import get_db


class CreateTaskTool(BaseTool):
    schema = ToolSchema(
        name="create_task",
        description="Create a new task from user intent, chat, or inferred browser activity.",
        input_schema={
            "title": {"type": "string", "description": "Short task title"},
            "description": {"type": "string", "description": "Optional detail", "nullable": True},
            "due": {"type": "string", "format": "datetime", "description": "ISO-8601 due datetime", "nullable": True},
            "source": {"type": "string", "enum": ["chat", "inferred", "calendar"], "description": "How the task was created"},
            "priority": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
            "tags": {"type": "array", "items": {"type": "string"}, "default": []},
        },
        output_schema={"task_id": {"type": "integer"}, "created_at": {"type": "string"}},
        risk_level="low",
    )

    async def execute(self, params: dict) -> dict[str, Any]:
        title = params.get("title", "").strip()
        if not title:
            raise ValueError("Task title must not be empty.")

        now = datetime.utcnow().isoformat()
        tags = ",".join(params.get("tags", []))

        with get_db() as conn:
            cur = conn.execute(
                """
                INSERT INTO tasks (title, description, due, source, priority, tags, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    title,
                    params.get("description"),
                    params.get("due"),
                    params.get("source", "chat"),
                    params.get("priority", "medium"),
                    tags,
                    now,
                    now,
                ),
            )
            task_id = cur.lastrowid

        return {"task_id": task_id, "created_at": now}


class GetTasksTool(BaseTool):
    schema = ToolSchema(
        name="get_tasks",
        description="Retrieve tasks, optionally filtered by status or priority.",
        input_schema={
            "status": {"type": "string", "enum": ["pending", "done", "snoozed", "all"], "default": "pending"},
            "limit": {"type": "integer", "default": 50},
        },
        output_schema={"tasks": {"type": "array"}},
        risk_level="low",
    )

    async def execute(self, params: dict) -> dict[str, Any]:
        status = params.get("status", "pending")
        limit = min(int(params.get("limit", 50)), 200)

        with get_db() as conn:
            if status == "all":
                rows = conn.execute(
                    "SELECT * FROM tasks ORDER BY due ASC, created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE status = ? ORDER BY due ASC, created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()

        return {"tasks": [dict(r) for r in rows]}


class UpdateTaskTool(BaseTool):
    schema = ToolSchema(
        name="update_task",
        description="Update a task's status, priority, title, or due date.",
        input_schema={
            "task_id": {"type": "integer", "description": "Task ID to update"},
            "status": {"type": "string", "enum": ["pending", "done", "snoozed"], "nullable": True},
            "title": {"type": "string", "nullable": True},
            "due": {"type": "string", "format": "datetime", "nullable": True},
            "priority": {"type": "string", "enum": ["low", "medium", "high"], "nullable": True},
        },
        risk_level="low",
    )

    async def execute(self, params: dict) -> dict[str, Any]:
        task_id = params.get("task_id")
        if not task_id:
            raise ValueError("task_id is required.")

        fields, values = [], []
        for col in ("status", "title", "due", "priority"):
            if col in params and params[col] is not None:
                fields.append(f"{col} = ?")
                values.append(params[col])

        if not fields:
            raise ValueError("No fields to update.")

        now = datetime.utcnow().isoformat()
        fields.append("updated_at = ?")
        values.append(now)
        values.append(task_id)

        with get_db() as conn:
            conn.execute(
                f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?", values
            )

        return {"task_id": task_id, "updated_at": now}


class DeleteTaskTool(BaseTool):
    schema = ToolSchema(
        name="delete_task",
        description="Permanently delete a task by ID.",
        input_schema={"task_id": {"type": "integer"}},
        risk_level="medium",
    )

    async def execute(self, params: dict) -> dict[str, Any]:
        task_id = params.get("task_id")
        if not task_id:
            raise ValueError("task_id is required.")

        with get_db() as conn:
            conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))

        return {"deleted": True, "task_id": task_id}
