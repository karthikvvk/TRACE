"""
web_search_tool.py — HTTP-based DuckDuckGo search tool for TRACE.

Works entirely without API keys or external browser services by querying
DuckDuckGo's HTML search interface and parsing the response directly.
"""

from html.parser import HTMLParser
import urllib.parse
from typing import Any
import httpx

from backend.tools.base import BaseTool, ToolSchema

# Standard user-agent to request the HTML results page successfully
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class DDGHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self.current_result = None
        self.in_title = False
        self.in_snippet = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class", "")
        
        # Check for start of a result container
        if tag == "div" and "web-result" in cls:
            if self.current_result:
                self.results.append(self.current_result)
            self.current_result = {"title": "", "link": "", "snippet": ""}
            
        elif self.current_result:
            if tag == "a" and "result__a" in cls:
                self.in_title = True
                href = attrs_dict.get("href", "")
                if href:
                    self.current_result["link"] = self.self_clean_url(href)
            elif tag == "a" and "result__snippet" in cls:
                self.in_snippet = True
            elif tag == "span" and "result__snippet" in cls:
                self.in_snippet = True

    def handle_endtag(self, tag):
        if self.in_title and tag == "a":
            self.in_title = False
        elif self.in_snippet and (tag == "a" or tag == "span" or tag == "div"):
            self.in_snippet = False

    def handle_data(self, data):
        if self.current_result:
            if self.in_title:
                self.current_result["title"] += data
            elif self.in_snippet:
                self.current_result["snippet"] += data

    def self_clean_url(self, url: str) -> str:
        # DuckDuckGo wraps outbound links in redirect logic
        if "uddg=" in url:
            parsed = urllib.parse.urlparse(url)
            qs = urllib.parse.parse_qs(parsed.query)
            uddg = qs.get("uddg", [])
            if uddg:
                return uddg[0]
        return url

    def close(self):
        super().close()
        if self.current_result:
            self.results.append(self.current_result)
            self.current_result = None


class WebSearchTool(BaseTool):
    schema = ToolSchema(
        name="web_search",
        description=(
            "Search the web using DuckDuckGo HTML search. Returns a list of search "
            "results containing titles, links, and snippets. Use this to find information "
            "on current events, general knowledge, documentation, or to search before fetching a URL."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query, e.g. 'python programming' or 'who won the latest superbowl'.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5, max 10).",
                },
            },
            "required": ["query"],
        },
        risk_level="low",
    )

    async def execute(self, params: dict) -> Any:
        query: str = params.get("query", "").strip()
        max_results: int = params.get("max_results", 5)
        if not query:
            return {"error": "No query provided."}
        
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"
        try:
            async with httpx.AsyncClient(
                headers=_HEADERS,
                follow_redirects=True,
                timeout=15.0,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return {"error": f"HTTP {exc.response.status_code} searching for {query}"}
        except httpx.RequestError as exc:
            return {"error": f"Search request failed: {exc}"}

        try:
            parser = DDGHTMLParser()
            parser.feed(resp.text)
            parser.close()
            
            # Clean up result values
            cleaned_results = []
            for res in parser.results:
                title = res["title"].strip()
                link = res["link"].strip()
                snippet = res["snippet"].strip()
                
                # Make sure we got something useful
                if title or link or snippet:
                    cleaned_results.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet
                    })
                    
            return {"query": query, "results": cleaned_results[:max_results]}
        except Exception as exc:
            return {"error": f"Failed to parse search results: {exc}"}
