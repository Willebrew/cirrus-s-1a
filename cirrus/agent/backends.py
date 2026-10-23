"""Search backends. Stdlib only, so the harness has no heavy dependencies.

Training and RL use ``LocalIndexBackend``-style offline indexes (see
docs/research/datasets.md); live backends are for fresh-data collection,
evaluation and serving.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str
    date: str | None = None


class SearchBackend(Protocol):
    def search(self, query: str, k: int = 8) -> list[SearchHit]: ...

    def fetch(self, url: str) -> str: ...


class MockBackend:
    """Deterministic in-memory backend for tests and offline demos.

    ``corpus`` maps url -> (title, text). Search is naive keyword scoring.
    """

    def __init__(self, corpus: dict[str, tuple[str, str]]):
        self.corpus = corpus
        self.queries: list[str] = []

    def search(self, query: str, k: int = 8) -> list[SearchHit]:
        self.queries.append(query)
        terms = {t.lower() for t in query.split()}
        scored = []
        for url, (title, text) in self.corpus.items():
            words = (title + " " + text).lower().split()
            score = sum(words.count(t) for t in terms)
            if score:
                scored.append((score, url, title, text))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [SearchHit(title=t, url=u, snippet=x[:300]) for _, u, t, x in scored[:k]]

    def fetch(self, url: str) -> str:
        return self.corpus.get(url, ("", ""))[1]


def _get_json(url: str, headers: dict[str, str], timeout: float) -> dict:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class BraveBackend:
    """Brave Search API (https://api.search.brave.com). Needs BRAVE_API_KEY."""

    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None = None, timeout: float = 4.0):
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY", "")
        if not self.api_key:
            raise ValueError("BRAVE_API_KEY is not set")
        self.timeout = timeout

    def search(self, query: str, k: int = 8) -> list[SearchHit]:
        qs = urllib.parse.urlencode({"q": query, "count": k, "extra_snippets": "true"})
        data = _get_json(
            f"{self.endpoint}?{qs}",
            {"Accept": "application/json", "X-Subscription-Token": self.api_key},
            self.timeout,
        )
        hits = []
        for r in data.get("web", {}).get("results", [])[:k]:
            snippet = " ".join([r.get("description", "")] + r.get("extra_snippets", []))
            hits.append(SearchHit(r.get("title", ""), r.get("url", ""), snippet, r.get("page_age")))
        return hits

    def fetch(self, url: str) -> str:
        return fetch_text(url, self.timeout)


class SearxngBackend:
    """Self-hosted SearXNG (JSON format must be enabled in settings.yml)."""

    def __init__(self, base_url: str | None = None, timeout: float = 4.0):
        self.base_url = (base_url or os.environ.get("SEARXNG_URL", "http://localhost:8888")).rstrip("/")
        self.timeout = timeout

    def search(self, query: str, k: int = 8) -> list[SearchHit]:
        qs = urllib.parse.urlencode({"q": query, "format": "json"})
        data = _get_json(f"{self.base_url}/search?{qs}", {"Accept": "application/json"}, self.timeout)
        return [
            SearchHit(r.get("title", ""), r.get("url", ""), r.get("content", ""), r.get("publishedDate"))
            for r in data.get("results", [])[:k]
        ]

    def fetch(self, url: str) -> str:
        return fetch_text(url, self.timeout)


def fetch_text(url: str, timeout: float = 4.0, max_bytes: int = 2_000_000) -> str:
    """Very small HTML-to-text fetcher. Production should use trafilatura or
    a Rust extractor; this keeps the reference harness dependency-free."""
    import html
    import re

    req = urllib.request.Request(url, headers={"User-Agent": "cirrus-s-1a/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(max_bytes).decode("utf-8", errors="replace")
    raw = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()
