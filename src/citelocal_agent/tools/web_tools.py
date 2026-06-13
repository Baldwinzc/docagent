"""Web tools for the document QA agent — opt-in, off by default.

These let the agent reach **outside** the local knowledge base: ``web_search``
finds relevant pages, ``fetch_url`` reads one. They are produced by
``make_web_tools(backend, max_results, fetch_chars)`` so they are bound to the
backend + config the agent was built with, exactly like ``make_retrieval_tools``.

The whole point of the project is *verifiable* citations, so web results are made
to play by the same rules as document chunks: every result carries a
``web:<url>`` locator and is emitted in the **same block format** ``search_docs``
uses (``[i] locator: <loc>  (relevance <s>)\\n<text>``). ``agent.tool_node`` then
records those locators + text into ``retrieved_locators`` / ``evidence`` with the
same parser, and ``extract_outcome`` verifies a ``web:<url>`` citation the same
way it verifies a file locator — cite a URL you never fetched and it is moved to
``unsupported``. No special-casing downstream.

Backends are pluggable (mirrors the project's nli/llm/off and openai/ollama
style). All third-party imports are **lazy**, so importing this module stays cheap
and offline; the heavy/network deps are only touched when web search is enabled.
"""

from __future__ import annotations

import os
import re
import urllib.request
from html.parser import HTMLParser
from typing import List, Protocol


# --------------------------------------------------------------------------- #
# Result block formatting (must match agent._BLOCK_RE / search_docs output)
# --------------------------------------------------------------------------- #
def _format_block(i: int, locator: str, score: float, text: str) -> str:
    """One result block in the exact format ``_parse_search_results`` expects."""
    return f"[{i}] locator: {locator}  (relevance {score:.2f})\n{text.strip()}"


# --------------------------------------------------------------------------- #
# Lightweight HTML -> text (stdlib only; avoids bs4 / requests as a dependency)
# --------------------------------------------------------------------------- #
class _HTMLToText(HTMLParser):
    """Collect visible text, skipping script/style/head and collapsing space."""

    _SKIP = {"script", "style", "head", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self._chunks.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._chunks)).strip()


def _http_get_text(url: str, max_chars: int, timeout: int = 20) -> str:
    """Fetch ``url`` and return readable text, truncated to ``max_chars``."""
    req = urllib.request.Request(url, headers={"User-Agent": "citelocal-agent/web"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (agent-driven)
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
    html = raw.decode(charset, errors="replace")
    parser = _HTMLToText()
    parser.feed(html)
    text = parser.text()
    return text[:max_chars]


# --------------------------------------------------------------------------- #
# Pluggable backends
# --------------------------------------------------------------------------- #
class WebBackend(Protocol):
    """A web search/fetch backend."""

    def search(self, query: str, k: int) -> List[dict]:
        """Return up to ``k`` results as ``{url, title, snippet, score?}`` dicts."""
        ...

    def fetch(self, url: str, max_chars: int) -> str:
        """Return readable text extracted from ``url``."""
        ...


class DuckDuckGoBackend:
    """Keyless web search via the ``ddgs`` package; fetch via stdlib urllib."""

    def search(self, query: str, k: int) -> List[dict]:
        try:
            from ddgs import DDGS
        except ImportError as e:  # pragma: no cover - import-guard
            raise ImportError(
                "DuckDuckGo backend needs the 'ddgs' package. "
                'Install web extras: pip install -e ".[web]"'
            ) from e
        results: list[dict] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=k):
                url = r.get("href") or r.get("url") or r.get("link") or ""
                if not url:
                    continue
                results.append(
                    {
                        "url": url,
                        "title": r.get("title", ""),
                        "snippet": r.get("body") or r.get("snippet") or "",
                    }
                )
        return results

    def fetch(self, url: str, max_chars: int) -> str:
        return _http_get_text(url, max_chars)


class TavilyBackend:
    """Higher-quality, agent-oriented search via Tavily (needs TAVILY_API_KEY)."""

    def _client(self):
        try:
            from tavily import TavilyClient
        except ImportError as e:  # pragma: no cover - import-guard
            raise ImportError(
                "Tavily backend needs the 'tavily-python' package. "
                'Install: pip install -e ".[tavily]"'
            ) from e
        if not os.environ.get("TAVILY_API_KEY"):
            raise RuntimeError(
                "Tavily backend needs TAVILY_API_KEY in the environment."
            )
        return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    def search(self, query: str, k: int) -> List[dict]:
        resp = self._client().search(query, max_results=k)
        results = []
        for r in resp.get("results", []):
            url = r.get("url", "")
            if not url:
                continue
            results.append(
                {
                    "url": url,
                    "title": r.get("title", ""),
                    "snippet": r.get("content", ""),
                    "score": r.get("score"),
                }
            )
        return results

    def fetch(self, url: str, max_chars: int) -> str:
        try:
            resp = self._client().extract(urls=[url])
            results = resp.get("results", [])
            if results and results[0].get("raw_content"):
                return str(results[0]["raw_content"])[:max_chars]
        except Exception:  # noqa: BLE001 - fall back to a plain HTTP fetch
            pass
        return _http_get_text(url, max_chars)


def get_web_backend(name: str = "ddg") -> WebBackend:
    """Factory: ``'ddg'`` (default, keyless) or ``'tavily'`` (needs a key)."""
    key = (name or "ddg").lower()
    if key == "ddg":
        return DuckDuckGoBackend()
    if key == "tavily":
        return TavilyBackend()
    raise ValueError(
        f"Unknown web search backend {name!r}; choose one of ['ddg', 'tavily']."
    )


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def make_web_tools(backend: WebBackend, max_results: int, fetch_chars: int) -> list:
    """Build ``[web_search, fetch_url]`` bound to a specific backend + config.

    Both emit results in ``search_docs``' block format with ``web:<url>``
    locators, so the agent's ``tool_node`` records them as verifiable evidence
    using the same parser — no downstream changes needed.
    """
    from langchain_core.tools import tool

    @tool
    def web_search(query: str, k: int = max_results) -> str:
        """Search the public web; return ranked results with `web:<url>` locators.

        Use this ONLY when the local knowledge base does not cover the question
        (or it needs current/external facts). Cite the `web:<url>` locators in
        your final Answer, exactly as you cite document locators. Follow up with
        `fetch_url` to read a promising result's full page before answering.
        """
        try:
            results = backend.search(query, k=k)
        except Exception as e:  # noqa: BLE001 - surface to the agent, never crash
            return f"web_search failed: {e}"
        if not results:
            return (
                "No web results found. Reformulate the query, or if nothing turns "
                "up, tell the user the answer could not be found."
            )
        blocks = []
        for i, r in enumerate(results, 1):
            score = r.get("score")
            score = float(score) if score is not None else max(0.1, 1.0 - 0.05 * (i - 1))
            text = "\n".join(p for p in (r.get("title", ""), r.get("snippet", "")) if p)
            blocks.append(_format_block(i, f"web:{r['url']}", score, text))
        return "\n\n".join(blocks)

    @tool
    def fetch_url(url: str) -> str:
        """Fetch and read the main text of a web page (use after `web_search`).

        Returns the page content under its `web:<url>` locator so you can ground
        a precise claim in it and cite `web:<url>` in your Answer.
        """
        try:
            text = backend.fetch(url, max_chars=fetch_chars)
        except Exception as e:  # noqa: BLE001 - surface to the agent, never crash
            return f"fetch_url failed for {url}: {e}"
        if not text.strip():
            return f"fetch_url returned no readable text for {url}."
        return _format_block(1, f"web:{url}", 1.0, text)

    return [web_search, fetch_url]
