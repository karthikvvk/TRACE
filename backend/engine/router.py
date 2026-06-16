"""
router.py — Tool routing logic.

Maps incoming request types or actions to the appropriate tool.
At Phase 1 this is a simple dict registry. Phase 4 will add LLM-assisted routing.
"""

from typing import Any

from backend.tools.base import BaseTool
from backend.tools.task_tool import CreateTaskTool, GetTasksTool, UpdateTaskTool, DeleteTaskTool
from backend.tools.memory_tool import LogEventTool, QueryMemoryTool
from backend.tools.search_tool import ExtractSearchIntentTool
from backend.tools.web_search_tool import WebSearchTool
from backend.tools.web_fetch_tool import WebFetchTool
from backend.tools.calendar_tool import GetCalendarEventsTool
from backend.tools.terminal_access import CreateTerminalTool, WriteTerminalTool, ReadTerminalTool, KillTerminalTool
from backend.tools.browser_tool import (
    BrowserGetActiveTabTool,
    BrowserGetPageTitleTool,
    BrowserGetTabsTool,
    BrowserGetDomTool,
    BrowserGetMetaTool,
    BrowserGetSelectionTool,
    BrowserGetCookiesTool,
    BrowserScrapeUrlTool,
    BrowserNavigateTool,
    BrowserOpenTabTool,
    BrowserCloseTabTool,
    BrowserClickTool,
    BrowserFillInputTool,
    BrowserExecuteScriptTool,
)


class ToolRouter:
    """
    Registry and dispatcher for all Friday tools.
    Tools are keyed by their schema.name string.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        defaults: list[BaseTool] = [
            CreateTaskTool(),
            GetTasksTool(),
            UpdateTaskTool(),
            DeleteTaskTool(),
            LogEventTool(),
            
            QueryMemoryTool(),

            ExtractSearchIntentTool(),
            WebSearchTool(),
            WebFetchTool(),


            GetCalendarEventsTool(),

            CreateTerminalTool(),
            WriteTerminalTool(),
            ReadTerminalTool(),
            KillTerminalTool(),

            # Browser tools (require extension WS connection)
            BrowserGetActiveTabTool(),
            BrowserGetPageTitleTool(),
            BrowserGetTabsTool(),
            BrowserGetDomTool(),
            BrowserGetMetaTool(),
            BrowserGetSelectionTool(),
            BrowserGetCookiesTool(),
            BrowserScrapeUrlTool(),
            BrowserNavigateTool(),
            BrowserOpenTabTool(),
            BrowserCloseTabTool(),
            BrowserClickTool(),
            BrowserFillInputTool(),
            BrowserExecuteScriptTool(),
        ]
        for tool in defaults:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.schema.name] = tool

    def list_tools(self) -> list[dict]:
        return [t.describe() for t in self._tools.values()]

    async def dispatch(self, tool_name: str, params: dict) -> Any:
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool: {tool_name!r}. Available: {list(self._tools)}")
        return await tool.execute(params)

    def get(self, tool_name: str) -> BaseTool | None:
        return self._tools.get(tool_name)
