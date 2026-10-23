"""Citation checks for cirrus-s-1a answers.

Three layers of defence against made-up content, cheapest first:

1. ``CitationConstraint``: decode-time grammar. Inside ``[...]`` only ids of
   sources that were actually retrieved can be generated, so the model cannot
   cite a source that does not exist.
2. ``audit_answer``: structural audit after (or while) streaming. Finds
   sentences that state something but cite nothing, and citations to unknown
   ids. The harness can strip or flag those sentences.
3. ``lexical_support``: a fast, crude groundedness score (content-word
   overlap with the cited sources). It is a stand-in for the NLI verifier
   (trained on AttributionBench + RAGTruth) that the RL reward and the
   production filter will use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-']*")
_STOP = frozenset(
    """a an the and or but of to in on at for from by with as is are was were be been being it its
    this that these those which who whom whose what when where why how than then so such not no
    can could will would should may might must do does did has have had he she they them their his
    her we our you your i me my about into over under after before between also more most other
    some any each all both few many much very just only same own there here up down out off again""".split()
)


def cited_ids(text: str) -> list[int]:
    ids: list[int] = []
    for m in _CITE_RE.finditer(text):
        ids.extend(int(x) for x in m.group(1).split(","))
    return ids


def split_sentences(answer: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(answer)]
    return [p for p in parts if p]


def _is_claim(sentence: str) -> bool:
    """Heuristic: headings, list labels and very short fragments are not claims."""
    s = sentence.lstrip("-*# ").strip()
    if not s or sentence.lstrip().startswith("#"):
        return False
    words = _WORD.findall(_CITE_RE.sub("", s))
    return len(words) >= 4


@dataclass
class AnswerAudit:
    n_sentences: int = 0
    n_claims: int = 0
    uncited_claims: list[str] = field(default_factory=list)
    unknown_ids: list[int] = field(default_factory=list)
    used_ids: list[int] = field(default_factory=list)

    @property
    def citation_coverage(self) -> float:
        if self.n_claims == 0:
            return 1.0
        return 1.0 - len(self.uncited_claims) / self.n_claims

    @property
    def ok(self) -> bool:
        return not self.unknown_ids and not self.uncited_claims


def audit_answer(answer: str, valid_ids: set[int]) -> AnswerAudit:
    a = AnswerAudit()
    used: set[int] = set()
    for sent in split_sentences(answer):
        a.n_sentences += 1
        ids = cited_ids(sent)
        used.update(ids)
        for i in ids:
            if i not in valid_ids and i not in a.unknown_ids:
                a.unknown_ids.append(i)
        if _is_claim(sent):
            a.n_claims += 1
            if not ids:
                a.uncited_claims.append(sent)
    a.used_ids = sorted(used)
    return a


def _content_words(text: str) -> set[str]:
    return {w.lower().strip(".'-") for w in _WORD.findall(text)} - _STOP


def lexical_support(sentence: str, sources: dict[int, str]) -> float:
    """Fraction of the sentence's content words found in its cited sources.

    Numbers count double: a wrong number is the most common and most harmful
    search-answer hallucination.
    """
    ids = cited_ids(sentence)
    if not ids:
        return 0.0
    words = _content_words(_CITE_RE.sub("", sentence))
    if not words:
        return 1.0
    pool: set[str] = set()
    for i in ids:
        pool |= _content_words(sources.get(i, ""))
    weight = lambda w: 2.0 if any(ch.isdigit() for ch in w) else 1.0  # noqa: E731
    total = sum(weight(w) for w in words)
    hit = sum(weight(w) for w in words if w in pool)
    return hit / total


class CitationConstraint:
    """Character-level grammar for citation brackets.

    Given the answer text generated so far, ``allowed_next`` returns the set
    of characters that may follow while inside an open ``[``; ``None`` means
    unconstrained. The decoder maps this to a token mask (tokens whose text
    starts with an allowed character and keeps the bracket valid).
    """

    def __init__(self, valid_ids: set[int]):
        self.valid = {str(i) for i in valid_ids}

    def _open_fragment(self, text: str) -> str | None:
        i = text.rfind("[")
        if i == -1 or "]" in text[i:]:
            return None
        return text[i + 1 :]

    def allowed_next(self, text: str) -> set[str] | None:
        frag = self._open_fragment(text)
        if frag is None:
            return None
        cur = frag.split(",")[-1].strip()
        allowed: set[str] = set()
        for v in self.valid:
            if v.startswith(cur) and len(v) > len(cur):
                allowed.add(v[len(cur)])
        if cur in self.valid:
            allowed |= {"]", ","}
        if not cur and frag.rstrip().endswith(","):
            allowed.add(" ")
        return allowed
