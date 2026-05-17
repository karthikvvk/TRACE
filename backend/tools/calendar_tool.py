"""
calendar_tool.py — Placeholder for Phase 5 Google Calendar integration.

At Phase 1 we only expose a stub that returns empty data so routes don't break.
When ready: replace execute() with Google Calendar API calls via OAuth token.
"""

from typing import Any
from .base import BaseTool, ToolSchema


class GetCalendarEventsTool(BaseTool):
    schema = ToolSchema(
        name="get_calendar_events",
        description="Fetch upcoming Google Calendar events. (Phase 5 — not yet active)",
        input_schema={
            "hours_ahead": {"type": "integer", "default": 24, "description": "Look-ahead window in hours"},
        },
        output_schema={"events": {"type": "array"}},
        risk_level="medium",
    )

    async def execute(self, params: dict) -> dict[str, Any]:
        # TODO Phase 5: implement OAuth + Google Calendar API
        return {"events": [], "note": "Calendar integration not yet active. Coming in Phase 5."}
