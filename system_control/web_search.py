"""
system_control/web_search.py
Real-time web search for VISION using Tavily's search API, including images.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config


def web_search(query: str, max_results: int = 5) -> dict:
    if not config.TAVILY_API_KEY:
        return {"success": False, "error": "No Tavily API key configured"}

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        response = client.search(
            query=query,
            max_results=max_results,
            include_images=True,
        )

        results = [
            {"title": r.get("title", ""), "content": r.get("content", ""), "url": r.get("url", "")}
            for r in response.get("results", [])
        ]
        images = response.get("images", [])

        return {"success": True, "results": results, "images": images}
    except Exception as e:
        return {"success": False, "error": str(e)}