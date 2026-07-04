"""
scrape.py — REST scraping endpoint for TRACE.

POST /scrape
  Body:  { "url": "https://...", "format": "text" | "markdown" | "html" }
  Response: {
      "url": str,
      "title": str,
      "content": str,
      "format": str,
      "source": "extension" | "httpx",
      "error": str | None
  }

Strategy:
  1. Try BrowserScrapeUrlTool (Chrome extension channel) — handles JS-rendered pages.
  2. If the extension is not connected or returns an error, fall back to WebFetchTool
     (httpx — always available, no extension required).

The result HTML is parsed with a lightweight regex stripper when converting to
text/markdown so we don't need extra npm packages on the Python side.
"""

import re
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, HttpUrl

from backend.tools.browser_tool import BrowserScrapeUrlTool
from backend.tools.web_fetch_tool import WebFetchTool
from backend.ws.manager import get_channel

router = APIRouter(tags=["scrape"])

# ---------------------------------------------------------------------------
# I/O models
# ---------------------------------------------------------------------------

FormatType = Literal["text", "markdown", "html"]


class ScrapeRequest(BaseModel):
    url: HttpUrl
    format: FormatType = "text"


class ScrapeResponse(BaseModel):
    url: str
    title: str = ""
    content: str = ""
    format: str
    source: Literal["extension", "httpx"]
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_CHARS = 30_000  # generous limit for LLM context


def _extract_title(html: str) -> str:
    """Pull the <title> from raw HTML (regex, no extra deps)."""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _html_to_text(html: str) -> str:
    """Lightweight HTML → plain text conversion."""
    # Drop <script> / <style> blocks
    html = re.sub(
        r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE
    )
    # Remove remaining tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _html_to_markdown(html: str) -> str:
    """
    Very basic HTML → Markdown via simple regex heuristics.
    Handles headings, bold, italic, anchors, paragraphs.
    For richer conversion install markdownify (optional dep).
    """
    try:
        import markdownify  # type: ignore

        return markdownify.markdownify(html, heading_style="ATX", strip=["script", "style"])
    except ImportError:
        pass  # fall back to regex

    h = html
    # headings
    for n in range(6, 0, -1):
        h = re.sub(rf"<h{n}[^>]*>(.*?)</h{n}>", r"{'#' * n} \1\n\n", h, flags=re.DOTALL | re.IGNORECASE)
    # bold / italic
    h = re.sub(r"<(strong|b)[^>]*>(.*?)</\1>", r"**\2**", h, flags=re.DOTALL | re.IGNORECASE)
    h = re.sub(r"<(em|i)[^>]*>(.*?)</\1>", r"*\2*", h, flags=re.DOTALL | re.IGNORECASE)
    # anchors
    h = re.sub(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', r"[\2](\1)", h, flags=re.DOTALL | re.IGNORECASE)
    # paragraphs / line-breaks
    h = re.sub(r"<br\s*/?>", "\n", h, flags=re.IGNORECASE)
    h = re.sub(r"</?p[^>]*>", "\n\n", h, flags=re.IGNORECASE)
    # strip the rest
    h = re.sub(r"<[^>]+>", "", h)
    h = re.sub(r"\n{3,}", "\n\n", h)
    return h.strip()


def _convert(html: str, fmt: FormatType) -> str:
    if fmt == "html":
        return html[:_MAX_CHARS]
    if fmt == "markdown":
        return _html_to_markdown(html)[:_MAX_CHARS]
    # default: text
    return _html_to_text(html)[:_MAX_CHARS]


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/scrape", response_model=ScrapeResponse)
async def scrape(req: ScrapeRequest) -> ScrapeResponse:
    """
    Scrape a URL using the Chrome extension (preferred) or httpx fallback.

    - **extension path**: opens the URL in a background Chrome tab, captures
      the fully-rendered HTML, and closes the tab.  Requires the TRACE
      Chrome extension to be connected via WebSocket.
    - **httpx fallback**: a plain HTTP GET — fast, always available, but
      misses JavaScript-rendered content.
    """
    url_str = str(req.url)
    fmt = req.format

    # ── 1. Try Chrome extension ────────────────────────────────────────────
    channel = get_channel()
    if channel.is_connected:
        try:
            browser_tool = BrowserScrapeUrlTool()
            result = await browser_tool.execute({"url": url_str})

            if isinstance(result, dict) and "error" not in result:
                # Extension returns the raw page HTML (or structured dict)
                raw_html: str = (
                    result.get("html")
                    or result.get("content")
                    or result.get("text")
                    or str(result)
                )
                title = result.get("title") or _extract_title(raw_html)
                content = _convert(raw_html, fmt)
                return ScrapeResponse(
                    url=url_str,
                    title=title,
                    content=content,
                    format=fmt,
                    source="extension",
                )
        except Exception:
            pass  # fall through to httpx

    # ── 2. Fallback: httpx ─────────────────────────────────────────────────
    fetch_tool = WebFetchTool()
    # Request raw HTML so we can do our own conversion
    fetch_result = await fetch_tool.execute({"url": url_str, "raw_html": True})

    if isinstance(fetch_result, dict) and "error" in fetch_result:
        return ScrapeResponse(
            url=url_str,
            title="",
            content="",
            format=fmt,
            source="httpx",
            error=fetch_result["error"],
        )

    raw_html = fetch_result.get("html", "")
    title = _extract_title(raw_html)
    content = _convert(raw_html, fmt)

    return ScrapeResponse(
        url=url_str,
        title=title,
        content=content,
        format=fmt,
        source="httpx",
    )
