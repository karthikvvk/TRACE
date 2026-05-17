"""
memory_tool.py — Tools for querying episodic memory and logging events manually.
"""

from typing import Any

from .base import BaseTool, ToolSchema
from backend.memory.episodic import EpisodicMemory


_episodic = EpisodicMemory()


class LogEventTool(BaseTool):
    schema = ToolSchema(
        name="log_event",
        description="Log a semantic browser or user event to episodic memory.",
        input_schema={
            "event_type": {"type": "string", "description": "e.g. navigation, search, form_submit"},
            "intent": {"type": "string", "description": "Classified intent category", "nullable": True},
            "category": {"type": "string", "description": "Topic category", "nullable": True},
            "time_bucket": {"type": "string", "description": "morning/afternoon/evening/night", "nullable": True},
            "session_length_s": {"type": "integer", "description": "Seconds spent on page/action", "nullable": True},
            "metadata": {"type": "object", "description": "Extra semantic key-value pairs", "nullable": True},
        },
        risk_level="low",
    )

    async def execute(self, params: dict) -> dict[str, Any]:
        event_id = await _episodic.log(params)
        return {"event_id": event_id}


class QueryMemoryTool(BaseTool):
    schema = ToolSchema(
        name="query_memory",
        description="Query episodic memory by keyword, intent, or time bucket.",
        input_schema={
            "keyword": {"type": "string", "nullable": True},
            "intent": {"type": "string", "nullable": True},
            "time_bucket": {"type": "string", "nullable": True},
            "limit": {"type": "integer", "default": 20},
        },
        output_schema={"events": {"type": "array"}},
        risk_level="low",
    )

    async def execute(self, params: dict) -> dict[str, Any]:
        events = await _episodic.query(
            keyword=params.get("keyword"),
            intent=params.get("intent"),
            time_bucket=params.get("time_bucket"),
            limit=int(params.get("limit", 20)),
        )
        return {"events": events}
