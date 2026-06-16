import asyncio
from backend.tools.browser_tool import BrowserScrapeUrlTool

async def main():
    print("🚀 Triggering real Chrome to scrape Amazon.in...")
    tool = BrowserScrapeUrlTool()
    
    # This will send a message to your REAL Chrome browser to open a tab,
    # wait for Amazon to load, extract the HTML, and close the tab.
    result = await tool.execute({"url": "https://www.amazon.in/s?k=laptop"})
    
    if isinstance(result, dict) and "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Success! Scraped {len(result)} characters of HTML.")
        
        # Save it to a file so you can inspect what it actually got
        with open("amazon_scraped.html", "w", encoding="utf-8") as f:
            f.write(result if isinstance(result, str) else str(result))
        print("📄 Saved output to amazon_scraped.html")

if __name__ == "__main__":
    asyncio.run(main())
