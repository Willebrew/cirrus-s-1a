"""The cirrus-s-1a agent loop.

Latency is the product, so the loop is built around overlap:

* searches are dispatched the moment ``<|/search|>`` streams out, while the
  model is still writing its next query;
* all searches/opens of a round run concurrently;
* results are deduplicated by URL and get stable ids for citation;
* the answer is audited for citations before it is returned.

The model is anything implementing ``StreamingModel``; the reference engine,
a vLLM/TRT-LLM server, or ``ScriptedModel`` for tests.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from cirrus.agent import protocol as P
from cirrus.agent.backends import SearchBackend
from cirrus.agent.citations import AnswerAudit, audit_answer


class StreamingModel(Protocol):
    def stream(self, prompt: str, stops: Sequence[str]) -> AsyncIterator[str]:
        """Yield text chunks; stop after emitting any string in ``stops``."""
        ...


@dataclass
class AgentConfig:
    max_rounds: int = 4
    results_per_search: int = 6
    max_sources: int = 40
    page_chars: int = 6000
    search_timeout_s: float = 3.0


@dataclass
class Episode:
    query: str
    transcript: str = ""
    answer: str | None = None
    sources: dict[int, P.Source] = field(default_factory=dict)
    searches: list[str] = field(default_factory=list)
    rounds: int = 0
    audit: AnswerAudit | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        return self.answer == P.NO_ANSWER


class Agent:
    def __init__(self, model: StreamingModel, backend: SearchBackend, cfg: AgentConfig | None = None):
        self.model = model
        self.backend = backend
        self.cfg = cfg or AgentConfig()

    async def _search(self, q: str):
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self.backend.search, q, self.cfg.results_per_search),
                self.cfg.search_timeout_s,
            )
        except Exception:  # a failed search must never kill the episode
            return []

    async def _fetch(self, src: P.Source) -> P.Source:
        try:
            src.page_text = await asyncio.wait_for(
                asyncio.to_thread(self.backend.fetch, src.url), self.cfg.search_timeout_s
            )
        except Exception:
            src.page_text = None
        return src

    def _add_hits(self, ep: Episode, hits) -> list[P.Source]:
        seen = {s.url for s in ep.sources.values()}
        new = []
        for h in hits:
            if h.url in seen or len(ep.sources) >= self.cfg.max_sources:
                continue
            seen.add(h.url)
            src = P.Source(id=len(ep.sources) + 1, title=h.title, url=h.url, snippet=h.snippet, date=h.date)
            ep.sources[src.id] = src
            new.append(src)
        return new

    async def run(self, query: str) -> Episode:
        t0 = time.perf_counter()
        ep = Episode(query=query, transcript=P.render_query(query))
        for r in range(self.cfg.max_rounds):
            ep.rounds = r + 1
            text, tasks, dispatched = "", [], 0
            async for chunk in self.model.stream(ep.transcript, P.TOOL_STOPS):
                text += chunk
                # fire searches as soon as their closing tag has streamed out
                qs = P.parse_turn(text).searches
                for q in qs[dispatched:]:
                    tasks.append(asyncio.create_task(self._search(q)))
                    ep.searches.append(q)
                dispatched = len(qs)
            ep.transcript += text
            turn = P.parse_turn(text)
            if turn.is_final:
                ep.answer = turn.answer
                break
            if not turn.searches and not turn.opens:
                break  # model produced nothing actionable; force an answer below

            opens = [ep.sources[i] for i in turn.opens if i in ep.sources]
            results = await asyncio.gather(*tasks, *(self._fetch(s) for s in opens))
            hit_lists, pages = results[: len(tasks)], results[len(tasks) :]
            new = [s for hits in hit_lists for s in self._add_hits(ep, hits)]
            if new:
                ep.transcript += P.render_results(new)
            for s in pages:
                ep.transcript += P.render_page(s, self.cfg.page_chars)
            ep.timings[f"round{r + 1}_s"] = time.perf_counter() - t0

        if ep.answer is None:
            # Out of rounds: make the model answer from what it has.
            ep.transcript += P.ANSWER_OPEN
            text = ""
            async for chunk in self.model.stream(ep.transcript, (P.ANSWER_CLOSE, P.END)):
                text += chunk
            ep.transcript += text
            ep.answer = P.parse_turn(P.ANSWER_OPEN + text).answer

        if ep.answer and not ep.abstained:
            ep.audit = audit_answer(ep.answer, set(ep.sources))
        ep.timings["total_s"] = time.perf_counter() - t0
        return ep


class ScriptedModel:
    """Replays fixed turns; streams each in small chunks like a real model."""

    def __init__(self, turns: Sequence[str], chunk: int = 7):
        self.turns = list(turns)
        self.chunk = chunk
        self.prompts: list[str] = []

    async def stream(self, prompt: str, stops: Sequence[str]) -> AsyncIterator[str]:
        self.prompts.append(prompt)
        text = self.turns.pop(0) if self.turns else P.NO_ANSWER
        for i in range(0, len(text), self.chunk):
            await asyncio.sleep(0)
            yield text[i : i + self.chunk]
