"""
web_fetch_tool.py — HTTP-based web scraping tool for TRACE.

Works entirely without a browser extension by using httpx.
Returns the page as plain text (HTML tags stripped) so the LLM
can read it without wading through raw markup.

Use this when the browser extension is not connected or when you
just need to read a URL quickly.
"""

import re
from typing import Any

import httpx

from backend.tools.base import BaseTool, ToolSchema

# Common browser-like headers to avoid simple bot-detection blocks
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

_MAX_TEXT_CHARS = 12_000  # keep LLM context manageable


def _strip_tags(html: str) -> str:
    """Very lightweight tag stripper — no external deps required."""
    # Remove <script> and <style> blocks entirely
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s{2,}", "\n", text)
    return text.strip()


class WebFetchTool(BaseTool):
    schema = ToolSchema(
        name="web_fetch",
        description=(
            "Fetch a URL over HTTP and return the readable text content of the page. "
            "Does NOT require the browser extension — works in any environment. "
            "Use this to read web pages, articles, documentation, or any public URL. "
            "For pages that require JavaScript rendering, use browser_scrape_url instead "
            "(needs extension). Returns up to 12 000 characters of text."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to fetch, e.g. 'https://example.com'.",
                },
                "raw_html": {
                    "type": "boolean",
                    "description": (
                        "If true, return the raw HTML instead of stripped text. "
                        "Useful when you need to inspect page structure. Default false."
                    ),
                },
            },
            "required": ["url"],
        },
        risk_level="low",
    )

    async def execute(self, params: dict) -> Any:
        url: str = params.get("url", "").strip()
        raw_html: bool = params.get("raw_html", False)

        if not url:
            return {"error": "No URL provided."}

        try:
            async with httpx.AsyncClient(
                headers=_HEADERS,
                follow_redirects=True,
                timeout=20.0,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return {"error": f"HTTP {exc.response.status_code} for {url}"}
        except httpx.RequestError as exc:
            return {"error": f"Request failed: {exc}"}

        content_type = resp.headers.get("content-type", "")
        body = resp.text

        if raw_html:
            return {"url": url, "content_type": content_type, "html": body[:_MAX_TEXT_CHARS]}

        # Strip to readable text
        if "text/html" in content_type or "html" in body[:200].lower():
            text = _strip_tags(body)
        else:
            text = body  # JSON, plain text, etc.

        if len(text) > _MAX_TEXT_CHARS:
            text = text[:_MAX_TEXT_CHARS] + f"\n\n[...truncated — {len(text)} total chars]"

        return {"url": url, "content_type": content_type, "text": text}
