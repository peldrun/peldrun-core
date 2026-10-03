"""
PELDRUN Core Web Search Tool.
Provides asynchronous internet search capabilities for agent information retrieval,
supporting optional search libraries with lightweight HTTP fallback and hermetic test injection.
"""

from __future__ import annotations

import asyncio
import logging
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Type
import httpx
from pydantic import BaseModel, Field

from peldrun.tools.base import BaseTool, ToolResult

logger = logging.getLogger("peldrun.tools.builtins.web_search")


class WebSearchArgs(BaseModel):
    """Input arguments schema for web search queries."""
    query: str = Field(
        ...,
        description="The search query terms to look up on the web."
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search results to retrieve (default: 5)."
    )


class WebSearchTool(BaseTool):
    """
    Core tool for searching the web and retrieving relevant external information.
    Provides resilient search querying with non-blocking HTTP requests.
    """

    name: str = "web_search"
    description: str = (
        "Search the web for up-to-date information, documentation, and external data. "
        "Returns titles, URLs, and concise snippets for relevant query results."
    )
    args_schema: Optional[Type[BaseModel]] = WebSearchArgs

    def __init__(
        self,
        workspace_root: Optional[str] = None,
        timeout: float = 15.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        super().__init__(workspace_root=workspace_root)
        self.timeout = timeout
        self._transport = transport

    async def _arun(
        self,
        query: str,
        max_results: int = 5,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute web search asynchronously and format results."""
        clean_query = query.strip()
        if not clean_query:
            return ToolResult(
                output="Search query cannot be empty.",
                exit_code=1,
                is_error=True,
            )

        logger.info("Executing web search for query: '%s' (max_results=%d)", clean_query, max_results)

        # Attempt 1: Try duckduckgo_search package if available and not mocked
        if self._transport is None:
            try:
                from duckduckgo_search import DDGS

                def _sync_ddgs_search() -> List[Dict[str, str]]:
                    results = []
                    with DDGS() as ddgs:
                        for item in ddgs.text(clean_query, max_results=max_results):
                            results.append({
                                "title": item.get("title", ""),
                                "url": item.get("href", ""),
                                "snippet": item.get("body", ""),
                            })
                    return results

                items = await asyncio.to_thread(_sync_ddgs_search)
                if items:
                    return self._format_results(items, clean_query)
            except ImportError:
                logger.debug("duckduckgo_search package not installed; falling back to direct HTTP search.")
            except Exception as ddg_err:
                logger.warning("duckduckgo_search invocation failed: %s; falling back to HTTP.", ddg_err)

        # Attempt 2: Fallback to direct HTTP search via DuckDuckGo HTML interface
        try:
            items = await self._fallback_http_search(clean_query, max_results)
            if items:
                return self._format_results(items, clean_query)
            else:
                return ToolResult(
                    output=f"No results found for query: '{clean_query}'.",
                    exit_code=0,
                    is_error=False,
                    metadata={"query": clean_query, "count": 0},
                )
        except Exception as ex:
            logger.exception("Web search failed for query '%s': %s", clean_query, ex)
            return ToolResult(
                output=f"Failed to perform web search: {str(ex)}",
                exit_code=1,
                is_error=True,
                metadata={"query": clean_query, "error_type": type(ex).__name__},
            )

    async def _fallback_http_search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Query DuckDuckGo HTML interface via async HTTP client as fallback."""
        url = "https://html.duckduckgo.com/html/"
        data = {"q": query}
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Content-Type": "application/x-www-form-urlencoded",
        }

        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            transport=self._transport,
        ) as client:
            resp = await client.post(url, data=data, headers=headers)
            resp.raise_for_status()
            html_text = resp.text

        results: List[Dict[str, str]] = []
        link_matches = re.findall(
            r'<a class="result__url" href="([^"]+)".*?>(.*?)</a>',
            html_text,
            re.DOTALL,
        )
        snippet_matches = re.findall(
            r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>',
            html_text,
            re.DOTALL,
        )

        total = min(len(link_matches), max_results)
        for i in range(total):
            raw_url, raw_title = link_matches[i]
            url_match = re.search(r'uddg=([^&]+)', raw_url)
            actual_url = urllib.parse.unquote(url_match.group(1)) if url_match else raw_url.strip()
            title = re.sub(r'<[^>]+>', '', raw_title).strip()
            snippet = ""
            if i < len(snippet_matches):
                snippet = re.sub(r'<[^>]+>', '', snippet_matches[i]).strip()

            results.append({
                "title": title or actual_url,
                "url": actual_url,
                "snippet": snippet,
            })

        return results

    def _format_results(self, items: List[Dict[str, str]], query: str) -> ToolResult:
        """Format parsed search items into structured output."""
        formatted_lines = [f"Search results for '{query}':\n"]
        for idx, item in enumerate(items, start=1):
            title = item.get("title", "No Title")
            url = item.get("url", "")
            snippet = item.get("snippet", "")
            formatted_lines.append(f"{idx}. {title}\n   URL: {url}\n   Snippet: {snippet}\n")

        return ToolResult(
            output="\n".join(formatted_lines).strip(),
            exit_code=0,
            is_error=False,
            metadata={"query": query, "count": len(items), "results": items},
        )