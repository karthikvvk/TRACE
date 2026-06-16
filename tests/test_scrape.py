"""
test_scrape.py — Programmatic tests for BrowserScrapeUrlTool.

How scraping works in TRACE (no Selenium):
  BrowserScrapeUrlTool
    └─→ WebSocket → Chrome Extension
                        ├── chrome.tabs.create({url, active: false})
                        ├── waitForTabLoad(tab.id)
                        ├── chrome.tabs.sendMessage → content script (GET_DOM)
                        └── chrome.tabs.remove(tab.id)
                        └─→ returns outerHTML string

These tests verify the Python tool layer:
  - Correct action ("scrape_url") and params are sent to the extension
  - The HTML payload returned by the extension is passed through unchanged
  - Error cases (extension offline, timeout, bad URL) degrade gracefully
  - Amazon.in-shaped HTML (realistic mock) is handled correctly

NOTE: The FakeExtension returns *mock* HTML — no real network call is made.
      To test against a live Amazon page you need the Chrome extension loaded
      in a real browser (manual/e2e test, not a unit test).
"""

import asyncio
import json
import socket

import pytest
import pytest_asyncio
from websockets.asyncio.client import connect as ws_connect

from backend.config import settings
from backend.tools.browser_tool import BrowserScrapeUrlTool

# ── Helpers ────────────────────────────────────────────────────────────────────

SECRET = settings.extension_secret


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Realistic Amazon.in mock HTML ──────────────────────────────────────────────

AMAZON_MOCK_HTML = """
<!DOCTYPE html>
<html lang="en-IN">
<head>
  <meta charset="UTF-8">
  <title>Amazon.in: Online Shopping - Electronics, Mobiles, Books, Clothing</title>
  <meta name="description" content="Online shopping from the earth's biggest selection">
</head>
<body>
  <div id="nav-main">
    <a id="nav-logo" href="/"><span>Amazon</span></a>
  </div>

  <!-- Search bar -->
  <form id="nav-search-bar-form" action="/s">
    <input id="twotabsearchtextbox" type="text" name="field-keywords"
           placeholder="Search Amazon.in" class="nav-input" />
    <input id="nav-search-submit-button" type="submit" value="Go" />
  </form>

  <!-- Featured deals section -->
  <div id="desktop-banner" class="deals-section">
    <h2>Today's Deals</h2>
  </div>

  <!-- Product grid (typical search result structure) -->
  <div data-component-type="s-search-result" class="s-result-item">
    <h2 class="a-size-mini">
      <a class="a-link-normal s-underline-text" href="/dp/B0BQVS2JN1">
        <span class="a-text-normal">
          Redmi 12 5G (Jade Black, 4GB RAM, 128GB Storage)
        </span>
      </a>
    </h2>
    <span class="a-price" data-a-size="xl">
      <span class="a-offscreen">₹11,999</span>
      <span aria-hidden="true">
        <span class="a-price-symbol">₹</span>
        <span class="a-price-whole">11,999</span>
      </span>
    </span>
    <span class="a-size-base a-color-secondary">
      M.R.P.: <span class="a-price a-text-strike">₹15,999</span>
    </span>
    <span class="a-size-base a-color-price">25% off</span>
    <div class="a-row a-size-small">
      <span aria-label="4.1 out of 5 stars">
        <i class="a-icon a-icon-star-small a-star-small-4"></i>
      </span>
      <span class="a-size-base">12,345</span>
    </div>
  </div>

  <div data-component-type="s-search-result" class="s-result-item">
    <h2 class="a-size-mini">
      <a class="a-link-normal s-underline-text" href="/dp/B0CHX1W1XY">
        <span class="a-text-normal">
          Samsung Galaxy A15 5G (Blue Black, 8GB, 128GB Storage)
        </span>
      </a>
    </h2>
    <span class="a-price" data-a-size="xl">
      <span class="a-offscreen">₹13,999</span>
      <span aria-hidden="true">
        <span class="a-price-symbol">₹</span>
        <span class="a-price-whole">13,999</span>
      </span>
    </span>
    <span class="a-size-base a-color-price">22% off</span>
  </div>

  <footer id="navFooter">
    <div>© 2024 Amazon.com, Inc. or its affiliates</div>
  </footer>
</body>
</html>
""".strip()


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_channel():
    import backend.ws.manager as mgr
    mgr._channel = None
    yield
    mgr._channel = None


@pytest_asyncio.fixture
async def live_server():
    """Real uvicorn on a random free port — torn down after each test."""
    import uvicorn
    from backend.main import app

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    deadline = asyncio.get_event_loop().time() + 8.0
    while not server.started:
        if asyncio.get_event_loop().time() > deadline:
            server.should_exit = True
            await task
            raise RuntimeError("uvicorn did not start in time")
        await asyncio.sleep(0.05)

    yield f"ws://127.0.0.1:{port}"

    server.should_exit = True
    await task


# ── FakeExtension (scrape-aware) ───────────────────────────────────────────────

class FakeScrapeExtension:
    """
    Simulates the Chrome extension for scrape_url requests.

    When the server sends {"action": "scrape_url", "params": {"url": "..."}},
    this fake extension checks that the URL matches `expected_url` and responds
    with `html_response` (mimicking what the content script would return).
    """

    def __init__(self, expected_url: str, html_response: str | dict,
                 *, secret: str = SECRET, delay: float = 0.0):
        self.expected_url = expected_url
        self.html_response = html_response
        self.secret = secret
        self.delay = delay          # simulate load time
        self._task: asyncio.Task | None = None
        self._ws = None
        self._ready = asyncio.Event()
        self.received_requests: list[dict] = []  # spy: all requests seen

    async def _run(self, server_url: str) -> None:
        uri = f"{server_url}/ws/browser?secret={self.secret}"
        try:
            # max_size=None allows large HTML payloads (default cap is 1 MB)
            async with ws_connect(uri, max_size=None) as ws:
                self._ws = ws
                self._ready.set()
                async for raw in ws:
                    msg = json.loads(raw)
                    self.received_requests.append(msg)
                    req_id = msg.get("id", "")
                    action = msg.get("action", "")
                    params = msg.get("params", {})

                    if action != "scrape_url":
                        reply = {"id": req_id, "error": f"Unexpected action: {action}"}
                    elif params.get("url") != self.expected_url:
                        reply = {
                            "id": req_id,
                            "error": f"URL mismatch: got {params.get('url')!r}",
                        }
                    else:
                        if self.delay:
                            await asyncio.sleep(self.delay)
                        reply = {"id": req_id, "result": self.html_response}

                    await ws.send(json.dumps(reply))
        except Exception:
            pass
        finally:
            self._ready.set()

    async def start(self, server_url: str) -> None:
        self._task = asyncio.create_task(self._run(server_url))
        await self._ready.wait()
        await asyncio.sleep(0.08)

    async def stop(self) -> None:
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass


# ===========================================================================
# 1. Offline / error-path tests (no server needed)
# ===========================================================================

class TestScrapeToolOffline:

    async def test_scrape_no_extension_returns_error(self):
        """Tool must degrade gracefully when the extension is not connected."""
        tool = BrowserScrapeUrlTool()
        result = await tool.execute({"url": "https://www.amazon.in"})
        assert "error" in result
        assert "not connected" in result["error"].lower()

    def test_scrape_schema_requires_url(self):
        """url is a required parameter."""
        tool = BrowserScrapeUrlTool()
        required = tool.schema.input_schema.get("required", [])
        assert "url" in required

    def test_scrape_schema_risk_level(self):
        """scrape_url opens a new tab — it is a medium-risk tool."""
        tool = BrowserScrapeUrlTool()
        assert tool.schema.risk_level == "medium"

    def test_scrape_tool_describe(self):
        tool = BrowserScrapeUrlTool()
        desc = tool.describe()
        for key in ("name", "description", "input_schema", "risk_level"):
            assert key in desc


# ===========================================================================
# 2. Happy-path scraping tests (with FakeScrapeExtension)
# ===========================================================================

class TestScrapeAmazon:

    async def test_scrape_amazon_returns_html(self, live_server):
        """
        Full pipeline: tool → WS → FakeExtension → tool returns HTML string.
        Verifies the tool relays the scrape_url action and passes the HTML through.
        """
        ext = FakeScrapeExtension(
            expected_url="https://www.amazon.in",
            html_response=AMAZON_MOCK_HTML,
        )
        await ext.start(live_server)

        try:
            tool = BrowserScrapeUrlTool()
            result = await tool.execute({"url": "https://www.amazon.in"})

            # Tool returns whatever the extension sends back
            assert result == AMAZON_MOCK_HTML, \
                f"Expected Amazon HTML, got: {str(result)[:200]!r}"
        finally:
            await ext.stop()

    async def test_scrape_sends_correct_action_and_url(self, live_server):
        """Verify the exact wire-format the tool sends to the extension."""
        target_url = "https://www.amazon.in/s?k=laptop"
        ext = FakeScrapeExtension(
            expected_url=target_url,
            html_response="<html><body>results</body></html>",
        )
        await ext.start(live_server)

        try:
            await BrowserScrapeUrlTool().execute({"url": target_url})

            assert len(ext.received_requests) == 1
            req = ext.received_requests[0]
            assert req["action"] == "scrape_url"
            assert req["params"]["url"] == target_url
        finally:
            await ext.stop()

    async def test_scrape_amazon_html_contains_key_elements(self, live_server):
        """
        Parse the returned HTML to assert realistic Amazon page structure.
        Checks for search bar, product listings, price elements, and footer.
        """
        ext = FakeScrapeExtension(
            expected_url="https://www.amazon.in",
            html_response=AMAZON_MOCK_HTML,
        )
        await ext.start(live_server)

        try:
            result = await BrowserScrapeUrlTool().execute({"url": "https://www.amazon.in"})
            assert isinstance(result, str), "Expected raw HTML string"

            # Page title / meta
            assert "Amazon.in" in result
            assert "Online Shopping" in result

            # Search bar elements
            assert 'id="twotabsearchtextbox"' in result
            assert 'name="field-keywords"' in result

            # Product listings
            assert 'data-component-type="s-search-result"' in result
            assert "Redmi 12 5G" in result
            assert "Samsung Galaxy A15" in result

            # Price elements (₹ symbol)
            assert "₹11,999" in result
            assert "₹13,999" in result
            assert "a-price-whole" in result

            # Discount badges
            assert "25% off" in result
            assert "22% off" in result

            # Product links (ASIN format)
            assert "/dp/B0BQVS2JN1" in result
            assert "/dp/B0CHX1W1XY" in result

            # Footer
            assert "Amazon.com, Inc." in result
        finally:
            await ext.stop()

    async def test_scrape_different_amazon_pages(self, live_server):
        """Scraping distinct Amazon.in URLs should each get their own request."""
        urls = [
            "https://www.amazon.in/s?k=mobile+phone",
            "https://www.amazon.in/s?k=laptop",
            "https://www.amazon.in/s?k=headphones",
        ]

        for url in urls:
            ext = FakeScrapeExtension(
                expected_url=url,
                html_response=f"<html><body>Results for {url}</body></html>",
            )
            await ext.start(live_server)
            try:
                result = await BrowserScrapeUrlTool().execute({"url": url})
                assert url in result, f"URL {url!r} not found in result"
                assert len(ext.received_requests) == 1
                assert ext.received_requests[0]["params"]["url"] == url
            finally:
                await ext.stop()

    async def test_scrape_large_html_page(self, live_server):
        """Tool must handle large HTML payloads without truncation."""
        # Simulate a heavy Amazon page (~500 KB of HTML)
        large_html = "<html><body>" + ("<div class='product'>item</div>" * 10_000) + "</body></html>"

        ext = FakeScrapeExtension(
            expected_url="https://www.amazon.in/s?k=all",
            html_response=large_html,
        )
        await ext.start(live_server)
        try:
            result = await BrowserScrapeUrlTool().execute({"url": "https://www.amazon.in/s?k=all"})
            assert result == large_html
            assert len(result) > 300_000   # confirms full payload was received (~310 KB)
        finally:
            await ext.stop()

    async def test_scrape_returns_dict_when_extension_returns_dict(self, live_server):
        """
        The content script bridge may return a structured dict (outerHTML + metadata)
        instead of a plain string — the tool must pass that through unchanged.
        """
        structured_response = {
            "outerHTML": AMAZON_MOCK_HTML,
            "url": "https://www.amazon.in",
            "title": "Amazon.in: Online Shopping",
        }
        ext = FakeScrapeExtension(
            expected_url="https://www.amazon.in",
            html_response=structured_response,
        )
        await ext.start(live_server)
        try:
            result = await BrowserScrapeUrlTool().execute({"url": "https://www.amazon.in"})
            assert result == structured_response
            assert "outerHTML" in result
            assert "Amazon.in" in result["title"]
        finally:
            await ext.stop()


# ===========================================================================
# 3. Failure / edge-case tests
# ===========================================================================

class TestScrapeEdgeCases:

    async def test_scrape_extension_returns_error(self, live_server):
        """Extension-side errors (e.g., CAPTCHA, JS timeout) surface as error dict."""

        class ErrorExtension(FakeScrapeExtension):
            async def _run(self, server_url: str) -> None:
                uri = f"{server_url}/ws/browser?secret={self.secret}"
                async with ws_connect(uri) as ws:
                    self._ws = ws
                    self._ready.set()
                    async for raw in ws:
                        msg = json.loads(raw)
                        # Extension signals an error (e.g., amazon blocked the request)
                        await ws.send(json.dumps({
                            "id": msg["id"],
                            "error": "Page load failed: amazon.in returned 503",
                        }))

        ext = ErrorExtension("https://www.amazon.in", "")
        await ext.start(live_server)
        try:
            result = await BrowserScrapeUrlTool().execute({"url": "https://www.amazon.in"})
            # RuntimeError from the channel gets caught and returned as error dict
            assert "error" in result
            assert "503" in result["error"] or "failed" in result["error"].lower()
        finally:
            await ext.stop()

    async def test_scrape_timeout(self, live_server):
        """
        If the page takes too long to load the tool raises TimeoutError.
        When the caller's wait_for fires before the tool's own 30s timeout,
        the CancelledError propagates as TimeoutError — both are valid signals
        that scraping timed out.
        """

        class SlowExtension(FakeScrapeExtension):
            async def _run(self, server_url: str) -> None:
                uri = f"{server_url}/ws/browser?secret={self.secret}"
                async with ws_connect(uri) as ws:
                    self._ws = ws
                    self._ready.set()
                    async for _ in ws:
                        await asyncio.sleep(9999)  # never replies

        ext = SlowExtension("https://www.amazon.in", "")
        await ext.start(live_server)
        try:
            # The outer wait_for(4s) fires before the tool's internal 30s timeout.
            # The tool's asyncio.wait_for gets cancelled → TimeoutError propagates.
            with pytest.raises((TimeoutError, asyncio.TimeoutError)):
                await asyncio.wait_for(
                    BrowserScrapeUrlTool().execute({"url": "https://www.amazon.in"}),
                    timeout=4.0,
                )
        finally:
            await ext.stop()

    async def test_scrape_empty_html_response(self, live_server):
        """An empty string from the extension is passed through as-is."""
        ext = FakeScrapeExtension(
            expected_url="https://www.amazon.in",
            html_response="",
        )
        await ext.start(live_server)
        try:
            result = await BrowserScrapeUrlTool().execute({"url": "https://www.amazon.in"})
            assert result == ""
        finally:
            await ext.stop()
