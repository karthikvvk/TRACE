"""
browser_tool.py — MCP browser tools for the TRACE agent.

Each tool is a thin async wrapper that sends a correlated command over the
persistent WebSocket to the Chrome extension and awaits the result.
The agent calls these tools by name; it never knows a browser is involved.

Tool categories:
  State  — instant Chrome API reads (active tab, title)
  Read   — fetches data from the current page or all tabs
  Write  — instructs the browser to perform an action

All tools degrade gracefully when the extension is not connected — they
return a structured error dict rather than raising, so the agent can
decide how to proceed.
"""

from typing import Any

from backend.tools.base import BaseTool, ToolSchema
from backend.ws.manager import get_channel
from backend.config import settings


def _no_props() -> dict:
    """Schema for tools that take no parameters."""
    return {"type": "object", "properties": {}}


def _channel_request(action: str, params: dict, timeout: float = 10.0):
    """Shared helper: calls the channel, wraps errors into a dict."""
    async def _exec() -> Any:
        channel = get_channel()
        if not channel.is_connected:
            return {
                "error": (
                    "Browser extension is not connected. "
                    "Make sure Friday is running and the extension is loaded."
                )
            }
        try:
            return await channel.request(action, params, timeout=timeout)
        except TimeoutError:
            return {"error": f"Browser action '{action}' timed out after {timeout}s."}
        except Exception as exc:
            return {"error": str(exc)}
    return _exec


# ── State tools ───────────────────────────────────────────────────────────────

class BrowserGetActiveTabTool(BaseTool):
    schema = ToolSchema(
        name="browser_get_active_tab",
        description=(
            "Return the currently active browser tab: its id, URL, and title. "
            "Use this to understand what the user is looking at right now."
        ),
        input_schema=_no_props(),
        risk_level="low",
    )

    async def execute(self, params: dict) -> Any:
        return await _channel_request("get_active_tab", {}, timeout=3.0)()


class BrowserGetPageTitleTool(BaseTool):
    schema = ToolSchema(
        name="browser_get_page_title",
        description="Return the title of the currently active browser tab.",
        input_schema=_no_props(),
        risk_level="low",
    )

    async def execute(self, params: dict) -> Any:
        return await _channel_request("get_page_title", {}, timeout=3.0)()


# ── Read tools ────────────────────────────────────────────────────────────────

class BrowserGetTabsTool(BaseTool):
    schema = ToolSchema(
        name="browser_get_tabs",
        description=(
            "List all open browser tabs with their ids, URLs, titles, "
            "and which one is currently active."
        ),
        input_schema=_no_props(),
        risk_level="low",
    )

    async def execute(self, params: dict) -> Any:
        return await _channel_request("get_tabs", {}, timeout=5.0)()


class BrowserGetDomTool(BaseTool):
    schema = ToolSchema(
        name="browser_get_dom",
        description=(
            "Return the full HTML DOM (outerHTML) of the currently active browser tab. "
            "Use this to read page content, find elements, or understand page structure. "
            "The result can be large — prefer browser_get_meta for lightweight context."
        ),
        input_schema=_no_props(),
        risk_level="low",
    )

    async def execute(self, params: dict) -> Any:
        return await _channel_request("get_dom", {}, timeout=10.0)()


class BrowserGetMetaTool(BaseTool):
    schema = ToolSchema(
        name="browser_get_meta",
        description=(
            "Return lightweight metadata for the active tab: title, URL, "
            "meta description, and scroll position. Faster than browser_get_dom."
        ),
        input_schema=_no_props(),
        risk_level="low",
    )

    async def execute(self, params: dict) -> Any:
        return await _channel_request("get_meta", {}, timeout=5.0)()


class BrowserGetSelectionTool(BaseTool):
    schema = ToolSchema(
        name="browser_get_selection",
        description=(
            "Return the text the user has currently selected / highlighted "
            "in the active browser tab."
        ),
        input_schema=_no_props(),
        risk_level="low",
    )

    async def execute(self, params: dict) -> Any:
        return await _channel_request("get_selection", {}, timeout=5.0)()


class BrowserGetCookiesTool(BaseTool):
    schema = ToolSchema(
        name="browser_get_cookies",
        description=(
            "Return cookies for a given domain. "
            "Leave domain empty to get all cookies. "
            "Only name, value, domain, and path are returned — never raw session tokens."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Domain to filter cookies for, e.g. 'example.com'. Optional.",
                }
            },
        },
        risk_level="medium",
    )

    async def execute(self, params: dict) -> Any:
        return await _channel_request(
            "get_cookies", {"domain": params.get("domain", "")}, timeout=5.0
        )()


class BrowserScrapeUrlTool(BaseTool):
    schema = ToolSchema(
        name="browser_scrape_url",
        description=(
            "Open a URL in a background tab, wait for it to finish loading, "
            "capture the full HTML, then close the tab. "
            "Useful for reading a page the user hasn't opened yet. "
            "This is slower than browser_get_dom — allow up to 30 seconds."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to open and scrape.",
                }
            },
            "required": ["url"],
        },
        risk_level="medium",
    )

    async def execute(self, params: dict) -> Any:
        return await _channel_request(
            "scrape_url", {"url": params["url"]}, timeout=30.0
        )()


# ── Write tools ───────────────────────────────────────────────────────────────

class BrowserNavigateTool(BaseTool):
    schema = ToolSchema(
        name="browser_navigate",
        description="Navigate the currently active browser tab to a given URL.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to navigate to."}
            },
            "required": ["url"],
        },
        risk_level="medium",
    )

    async def execute(self, params: dict) -> Any:
        return await _channel_request(
            "navigate", {"url": params["url"]}, timeout=10.0
        )()


class BrowserOpenTabTool(BaseTool):
    schema = ToolSchema(
        name="browser_open_tab",
        description="Open a new browser tab at the given URL.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to open."},
                "active": {
                    "type": "boolean",
                    "description": "Whether to bring the new tab into focus. Default true.",
                },
            },
            "required": ["url"],
        },
        risk_level="medium",
    )

    async def execute(self, params: dict) -> Any:
        return await _channel_request(
            "open_tab",
            {"url": params["url"], "active": params.get("active", True)},
            timeout=10.0,
        )()


class BrowserCloseTabTool(BaseTool):
    schema = ToolSchema(
        name="browser_close_tab",
        description="Close a browser tab by its numeric tab id.",
        input_schema={
            "type": "object",
            "properties": {
                "tab_id": {
                    "type": "integer",
                    "description": "The tab id returned by browser_get_tabs or browser_open_tab.",
                }
            },
            "required": ["tab_id"],
        },
        risk_level="medium",
    )

    async def execute(self, params: dict) -> Any:
        return await _channel_request(
            "close_tab", {"tab_id": params["tab_id"]}, timeout=5.0
        )()


class BrowserClickTool(BaseTool):
    schema = ToolSchema(
        name="browser_click",
        description=(
            "Click a DOM element in the active tab identified by a CSS selector. "
            "Returns ok=false with an error if the selector matches nothing."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the element to click, e.g. '#submit-btn'.",
                }
            },
            "required": ["selector"],
        },
        risk_level="medium",
    )

    async def execute(self, params: dict) -> Any:
        return await _channel_request(
            "click", {"selector": params["selector"]}, timeout=10.0
        )()


class BrowserFillInputTool(BaseTool):
    schema = ToolSchema(
        name="browser_fill_input",
        description=(
            "Set the value of an input or textarea in the active tab. "
            "Fires input and change events so React/Vue state updates properly. "
            "Returns ok=false with an error if the selector matches nothing."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the input element.",
                },
                "value": {
                    "type": "string",
                    "description": "The value to set.",
                },
            },
            "required": ["selector", "value"],
        },
        risk_level="medium",
    )

    async def execute(self, params: dict) -> Any:
        return await _channel_request(
            "fill_input",
            {"selector": params["selector"], "value": params["value"]},
            timeout=10.0,
        )()


class BrowserExecuteScriptTool(BaseTool):
    schema = ToolSchema(
        name="browser_execute_script",
        description=(
            "Execute arbitrary JavaScript in the active tab and return the result. "
            "CAUTION: this is a high-risk tool. It is disabled by default and must be "
            "explicitly enabled by setting ALLOW_SCRIPT_EXECUTION=true in .env."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "JavaScript code to execute. Must be a valid JS expression or block.",
                }
            },
            "required": ["code"],
        },
        risk_level="high",
    )

    async def execute(self, params: dict) -> Any:
        if not settings.allow_script_execution:
            return {
                "error": (
                    "browser_execute_script is disabled. "
                    "Set ALLOW_SCRIPT_EXECUTION=true in .env to enable it."
                )
            }
        return await _channel_request(
            "execute_script", {"code": params["code"]}, timeout=15.0
        )()
