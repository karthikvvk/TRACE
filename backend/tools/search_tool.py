"""
search_tool.py — Extract semantic search intent from URL patterns.

Supports Google, Bing, DuckDuckGo, YouTube, GitHub, Reddit, and generic domains.
This runs purely locally — no network calls.
"""

import re
from urllib.parse import urlparse, parse_qs, unquote_plus
from typing import Any

from .base import BaseTool, ToolSchema


# Maps hostname patterns to (engine_name, query_param)
SEARCH_ENGINE_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"(www\.)?google\.[a-z.]+"), "google", "q"),
    (re.compile(r"(www\.)?bing\.com"), "bing", "q"),
    (re.compile(r"(www\.)?duckduckgo\.com"), "duckduckgo", "q"),
    (re.compile(r"(www\.)?search\.yahoo\.com"), "yahoo", "p"),
    (re.compile(r"(www\.)?youtube\.com"), "youtube", "search_query"),
    (re.compile(r"(www\.)?reddit\.com/search"), "reddit", "q"),
    (re.compile(r"github\.com/search"), "github", "q"),
]

# Topic keywords → category label
CATEGORY_KEYWORDS: list[tuple[list[str], str]] = [
    (["flight", "hotel", "trip", "travel", "booking", "airbnb", "hostel"], "travel"),
    (["doctor", "dentist", "clinic", "hospital", "symptom", "medicine", "health"], "health"),
    (["buy", "shop", "price", "amazon", "flipkart", "deal", "discount", "order"], "shopping"),
    (["code", "github", "stackoverflow", "python", "javascript", "react", "bug", "error", "api"], "development"),
    (["news", "article", "politics", "election", "sport"], "news"),
    (["recipe", "food", "cook", "restaurant", "menu"], "food"),
    (["learn", "tutorial", "course", "study", "lecture", "udemy", "coursera"], "learning"),
    (["job", "resume", "career", "interview", "linkedin", "hire"], "career"),
    (["finance", "stock", "invest", "bank", "crypto", "tax", "budget"], "finance"),
    (["movie", "series", "netflix", "youtube", "music", "spotify", "game"], "entertainment"),
]


def extract_query_from_url(url: str) -> dict[str, Any] | None:
    """Parse a URL and return a semantic intent object if it looks like a search."""
    try:
        parsed = urlparse(url)
        host = parsed.netloc

        for pattern, engine, param in SEARCH_ENGINE_PATTERNS:
            if pattern.search(host) or (engine == "reddit" and "/search" in parsed.path):
                qs = parse_qs(parsed.query)
                query_terms = qs.get(param, [])
                if not query_terms:
                    return None
                query_text = unquote_plus(query_terms[0]).lower()
                category = _classify_category(query_text)
                return {
                    "engine": engine,
                    "query": query_text,
                    "category": category,
                    "intent": "search",
                }

        return None
    except Exception:
        return None


def _classify_category(text: str) -> str:
    text_lower = text.lower()
    for keywords, category in CATEGORY_KEYWORDS:
        if any(kw in text_lower for kw in keywords):
            return category
    return "general"


class ExtractSearchIntentTool(BaseTool):
    schema = ToolSchema(
        name="extract_search_intent",
        description="Extract a semantic search intent object from a raw browser URL.",
        input_schema={"url": {"type": "string", "description": "Full browser URL"}},
        output_schema={
            "engine": {"type": "string"},
            "query": {"type": "string"},
            "category": {"type": "string"},
            "intent": {"type": "string"},
        },
        risk_level="low",
    )

    async def execute(self, params: dict) -> dict[str, Any] | None:
        url = params.get("url", "")
        return extract_query_from_url(url)
