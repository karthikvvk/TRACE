"""
routes/tasks.py — CRUD endpoints for the tasks table.

All heavy lifting is in the tool classes; routes are thin dispatchers.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Literal

from backend.tools.task_tool import CreateTaskTool, GetTasksTool, UpdateTaskTool, DeleteTaskTool

router = APIRouter(prefix="/tasks", tags=["tasks"])

_create = CreateTaskTool()
_get = GetTasksTool()
_update = UpdateTaskTool()
_delete = DeleteTaskTool()


class CreateTaskRequest(BaseModel):
    title: str
    description: str | None = None
    due: str | None = None
    source: Literal["chat", "inferred", "calendar"] = "chat"
    priority: Literal["low", "medium", "high"] = "medium"
    tags: list[str] = []


class UpdateTaskRequest(BaseModel):
    status: Literal["pending", "done", "snoozed"] | None = None
    title: str | None = None
    due: str | None = None
    priority: Literal["low", "medium", "high"] | None = None


@router.get("")
async def list_tasks(status: str = "pending", limit: int = 50) -> dict[str, Any]:
    """Return tasks filtered by status."""
    return await _get.execute({"status": status, "limit": limit})


@router.post("")
async def create_task(body: CreateTaskRequest) -> dict[str, Any]:
    """Create a new task."""
    try:
        return await _create.execute(body.model_dump(exclude_none=True))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.patch("/{task_id}")
async def update_task(task_id: int, body: UpdateTaskRequest) -> dict[str, Any]:
    """Update task fields by ID."""
    try:
        return await _update.execute({"task_id": task_id, **body.model_dump(exclude_none=True)})
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.delete("/{task_id}")
async def delete_task(task_id: int) -> dict[str, Any]:
    """Permanently delete a task."""
    return await _delete.execute({"task_id": task_id})


@router.post("/{task_id}/done")
async def mark_done(task_id: int) -> dict[str, Any]:
    """Shortcut: mark a task as done."""
    return await _update.execute({"task_id": task_id, "status": "done"})


@router.post("/{task_id}/snooze")
async def snooze_task(task_id: int) -> dict[str, Any]:
    """Shortcut: snooze a task."""
    return await _update.execute({"task_id": task_id, "status": "snoozed"})
