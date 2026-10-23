"""The cirrus-s-1a conversation and tool-call format.

Design goals:
  * trivially parseable by a streaming harness (special tokens, no JSON)
  * cheap in tokens (every token costs latency)
  * parallel by default: one turn can emit several searches at once, and the
    harness fires each search the moment its closing tag streams out, so
    network latency overlaps with the model still writing the next query
  * the model yields control explicitly with ``<|run|>``
  * every source gets a stable integer id the answer cites as ``[n]``

A full episode looks like:

    <|query|>best budget e-bike 2026<|/query|>
    <|search|>best budget e-bike 2026 review<|/search|>
    <|search|>e-bike under $1000 2026 comparison<|/search|>
    <|run|>
    <|results|>
    <|src 1|>Title | example.com/a | 2026-09-02
    snippet text...
    <|src 2|>...
    <|/results|>
    <|open|>2<|/open|>
    <|run|>
    <|page 2|>full extracted page text, truncated<|/page|>
    <|answer|>
    ## Top picks
    - The Foo X1 costs $899 [2] ...
    <|/answer|>

Everything inside <|results|> and <|page n|> is written by the harness and is
masked out of the training loss.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

QUERY_OPEN, QUERY_CLOSE = "<|query|>", "<|/query|>"
SEARCH_OPEN, SEARCH_CLOSE = "<|search|>", "<|/search|>"
OPEN_OPEN, OPEN_CLOSE = "<|open|>", "<|/open|>"
RUN = "<|run|>"  # model yields to the harness to execute pending calls
RESULTS_OPEN, RESULTS_CLOSE = "<|results|>", "<|/results|>"
PAGE_CLOSE = "<|/page|>"
ANSWER_OPEN, ANSWER_CLOSE = "<|answer|>", "<|/answer|>"
NO_ANSWER = "<|insufficient|>"  # emitted when sources do not support an answer
END = "<|end|>"

# Fixed-arity special tokens. Source / page ids are rendered as plain text
# after these markers so the vocab does not need one token per id.
SPECIAL_TOKENS = [
    QUERY_OPEN, QUERY_CLOSE,
    SEARCH_OPEN, SEARCH_CLOSE,
    OPEN_OPEN, OPEN_CLOSE, RUN,
    RESULTS_OPEN, RESULTS_CLOSE,
    "<|src ", "<|page ", PAGE_CLOSE,
    ANSWER_OPEN, ANSWER_CLOSE,
    NO_ANSWER, END,
]

# Generation stops at any of these so the harness can act.
TOOL_STOPS = (RUN, ANSWER_CLOSE, NO_ANSWER, END)


class ActionKind(str, Enum):
    SEARCH = "search"
    OPEN = "open"
    ANSWER = "answer"


@dataclass
class Action:
    kind: ActionKind
    arg: str


@dataclass
class Source:
    id: int
    title: str
    url: str
    snippet: str
    date: str | None = None
    page_text: str | None = None


@dataclass
class Turn:
    """Model output between two harness interventions."""

    searches: list[str] = field(default_factory=list)
    opens: list[int] = field(default_factory=list)
    answer: str | None = None

    @property
    def is_final(self) -> bool:
        return self.answer is not None


_SEARCH_RE = re.compile(re.escape(SEARCH_OPEN) + r"(.*?)" + re.escape(SEARCH_CLOSE), re.S)
_OPEN_RE = re.compile(re.escape(OPEN_OPEN) + r"\s*(\d+)\s*" + re.escape(OPEN_CLOSE))
_ANSWER_RE = re.compile(re.escape(ANSWER_OPEN) + r"(.*?)(?:" + re.escape(ANSWER_CLOSE) + r"|$)", re.S)


def parse_turn(text: str) -> Turn:
    """Parse one model turn. Unterminated answers are accepted (streaming)."""
    turn = Turn()
    turn.searches = [q.strip() for q in _SEARCH_RE.findall(text) if q.strip()]
    turn.opens = [int(i) for i in _OPEN_RE.findall(text)]
    m = _ANSWER_RE.search(text)
    if m:
        turn.answer = m.group(1).strip()
    elif NO_ANSWER in text:
        turn.answer = NO_ANSWER
    return turn


def _clean(s: str) -> str:
    # Never let retrieved text smuggle protocol tokens into the context.
    return s.replace("<|", "< |").replace("|>", "| >")


def render_query(query: str) -> str:
    return f"{QUERY_OPEN}{_clean(query.strip())}{QUERY_CLOSE}\n"


def render_results(sources: list[Source]) -> str:
    lines = [RESULTS_OPEN]
    for s in sources:
        header = " | ".join(x for x in (_clean(s.title), _clean(s.url), s.date or "") if x)
        lines.append(f"<|src {s.id}|>{header}\n{_clean(s.snippet)}")
    lines.append(RESULTS_CLOSE)
    return "\n".join(lines) + "\n"


def render_page(source: Source, max_chars: int = 6000) -> str:
    body = _clean((source.page_text or source.snippet)[:max_chars])
    return f"<|page {source.id}|>{body}{PAGE_CLOSE}\n"
