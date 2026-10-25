"""Search transcripts in the cirrus protocol, built from QA datasets that ship their documents.

No teacher model is needed: HotpotQA (distractor setting) and SQuAD 2.0 come with
the passages that answer each question (and, for HotpotQA, the exact supporting
sentences), so a correct, fully cited transcript can be written by rule:

    <|query|>...<|/query|>
    <|search|>...<|/search|>          model
    <|run|>                            model
    <|results|> snippets <|/results|>  harness (no loss)
    <|open|>2<|/open|><|run|>          model: only when a snippet lacks the evidence
    <|page 2|> passage <|/page|>       harness (no loss)
    <|answer|>**X** [2, 5] ...         model: the answer first, then the evidence quoted
                                       sentence by sentence, each with its citation

Three variants target the behaviours the product depends on:

* answerable: read, cite, copy exact wording (which also raises speculative tau)
* counterfactual: the answer entity is replaced consistently in every passage, so
  the transcript is only right if the model trusts the sources over its memory
* unanswerable: the gold passages are removed (or SQuAD 2.0's adversarial
  unanswerable questions are used) and the target is ``<|insufficient|>``

These are format and grounding lessons for mid-training and SFT; query writing
and multi-round search come from teacher trajectories later.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field, replace

from cirrus.agent import protocol as P
from cirrus.agent.citations import split_sentences


@dataclass
class Doc:
    title: str
    text: str
    gold: bool = False
    support: list[str] = field(default_factory=list)  # sentences of ``text`` that support the answer


@dataclass
class Example:
    id: str
    query: str
    docs: list[Doc]
    answer: str | None  # None: the documents do not answer the query
    kind: str = "answerable"  # answerable | counterfactual | unanswerable


@dataclass
class Segment:
    text: str
    train: bool  # False for text the harness writes (query, results, pages): no loss


# ------------------------------------------------------------------- readers
def from_hotpot(row: dict) -> Example | None:
    """``hotpotqa/hotpot_qa`` (distractor): 10 paragraphs, 2 gold, supporting sentence ids."""
    support: dict[str, set[int]] = {}
    for t, i in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]):
        support.setdefault(t, set()).add(i)
    docs = []
    for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
        ids = support.get(title, set())
        docs.append(Doc(title, "".join(sents).strip(), gold=bool(ids),
                        support=[sents[i].strip() for i in sorted(ids) if i < len(sents)]))
    if not any(d.gold for d in docs) or not row["answer"].strip():
        return None
    return Example(row["id"], row["question"].strip(), docs, row["answer"].strip())


_SENT_END = re.compile(r"(?<=[.!?])\s+")


def from_squad(row: dict, distractors: list[Doc]) -> Example:
    """``rajpurkar/squad_v2``: one paragraph; empty ``answers`` means unanswerable from it."""
    ctx = row["context"].strip()
    answers = row["answers"]["text"]
    docs = [Doc(row["title"].replace("_", " "), ctx)] + distractors
    if not answers:
        return Example(row["id"], row["question"].strip(), docs, None, kind="unanswerable")
    start = row["answers"]["answer_start"][0]
    # the sentence containing the answer is the evidence
    pos = 0
    for sent in _SENT_END.split(ctx):
        if pos <= start < pos + len(sent) + 1:
            docs[0].support = [sent.strip()]
            break
        pos += len(sent) + 1
    docs[0].gold = True
    if not docs[0].support:
        docs[0].support = [ctx]
    return Example(row["id"], row["question"].strip(), docs, answers[0].strip())


# ------------------------------------------------------------------ variants
def answer_shape(ans: str) -> str | None:
    """Coarse type used to pick a plausible substitute; None if not swappable."""
    if re.fullmatch(r"\d{3,4}", ans):
        return "year"
    if re.fullmatch(r"[\d,.]+", ans):
        return "number"
    words = ans.split()
    if 1 <= len(words) <= 4 and all(w[:1].isupper() for w in words) and ans.lower() not in ("yes", "no"):
        return "name"
    return None


def counterfactual(ex: Example, substitute: str) -> Example | None:
    """Replace the answer with ``substitute`` everywhere it occurs in the documents.

    Returns None when the swap would not be clean: the answer is not quoted in the
    evidence, the substitute already appears somewhere, or the query names the answer.
    """
    ans = ex.answer
    if not ans or substitute == ans or answer_shape(ans) is None:
        return None
    pat = re.compile(rf"(?<!\w){re.escape(ans)}(?!\w)")
    if pat.search(ex.query) or not any(pat.search(s) for d in ex.docs if d.gold for s in d.support):
        return None
    sub_pat = re.compile(rf"(?<!\w){re.escape(substitute)}(?!\w)")
    if any(sub_pat.search(d.text) for d in ex.docs) or sub_pat.search(ex.query):
        return None
    swap = lambda m: substitute  # noqa: E731  (a function, so backslashes in the substitute stay literal)
    docs = [replace(d, text=pat.sub(swap, d.text), support=[pat.sub(swap, s) for s in d.support]) for d in ex.docs]
    return replace(ex, docs=docs, answer=substitute, kind="counterfactual")


def unanswerable(ex: Example) -> Example | None:
    """Drop the gold documents; the remaining distractors do not answer the query."""
    rest = [d for d in ex.docs if not d.gold]
    if len(rest) < 2:
        return None
    return replace(ex, docs=rest, answer=None, kind="unanswerable")


# ----------------------------------------------------------------- rendering
def _url(title: str) -> str:
    return "en.wikipedia.org/wiki/" + title.replace(" ", "_")


def _snippet(text: str, chars: int) -> str:
    if len(text) <= chars:
        return text
    cut = text[:chars].rsplit(" ", 1)[0]
    return cut + " ..."


def _search_query(q: str) -> str:
    aux = r"(?:is|are|was|were|did|does|do|has|have|had|can|could|will|would)"
    q = re.sub(rf"^(?:(?:what|which|who|whom|whose|when|where|why|how)(?:\s+{aux})?|{aux})\s+", "", q.strip(),
               flags=re.I)
    return q.rstrip("?").strip() or q


def render(ex: Example, rng: random.Random, snippet_chars: int = 240, max_results: int = 8,
           page_chars: int = 6000) -> list[Segment]:
    """One transcript as segments; harness-written ones carry no loss."""
    docs = list(ex.docs)
    gold = [d for d in docs if d.gold]
    others = [d for d in docs if not d.gold]
    rng.shuffle(others)
    docs = gold + others[: max(0, max_results - len(gold))]
    rng.shuffle(docs)
    sources = [P.Source(id=i + 1, title=d.title, url=_url(d.title), snippet=_snippet(d.text, snippet_chars),
                        page_text=d.text) for i, d in enumerate(docs)]
    by_doc = {id(d): s for d, s in zip(docs, sources)}

    seg = [Segment(P.render_query(ex.query), False),
           Segment(f"{P.SEARCH_OPEN}{_search_query(ex.query)}{P.SEARCH_CLOSE}\n{P.RUN}", True),
           Segment(P.render_results(sources), False)]
    if ex.answer is None:
        seg.append(Segment(P.NO_ANSWER, True))
        return seg

    # open a gold page only when its snippet does not already contain the evidence
    to_open = [by_doc[id(d)] for d in docs if d.gold and not all(s in by_doc[id(d)].snippet for s in d.support)]
    if to_open:
        seg.append(Segment("".join(f"{P.OPEN_OPEN}{s.id}{P.OPEN_CLOSE}\n" for s in to_open) + P.RUN, True))
        seg.append(Segment("".join(P.render_page(s, page_chars) for s in to_open), False))

    cites = sorted(by_doc[id(d)].id for d in docs if d.gold)
    head = ex.answer.strip().rstrip(".")
    head = head.capitalize() if head.lower() in ("yes", "no") else head
    lines = [_cite(f"**{head}**.", cites), ""]
    for d in sorted((d for d in docs if d.gold), key=lambda d: by_doc[id(d)].id):
        for s in d.support:  # every sentence of a quoted span carries the citation
            lines.append("- " + " ".join(_cite(x, [by_doc[id(d)].id]) for x in split_sentences(s)))
    seg.append(Segment(f"{P.ANSWER_OPEN}\n" + "\n".join(lines) + f"\n{P.ANSWER_CLOSE}", True))
    return seg


_TAIL = re.compile(r"([.!?]+[\"')\]]*)$")


def _cite(sentence: str, ids: list[int]) -> str:
    """Put the citation before the closing punctuation ("... in 1800 [2]."), the form
    cirrus.agent.citations audits; text after "]." would read as a new sentence."""
    tag = f" [{', '.join(map(str, ids))}]"
    m = _TAIL.search(sentence)
    return sentence[: m.start()] + tag + m.group(1) if m else sentence + tag


def to_text(segments: list[Segment]) -> str:
    return "".join(s.text for s in segments)


def encode(segments: list[Segment], encode_fn) -> tuple[list[int], list[bool]]:
    """Token ids and a per-token loss mask (True = train on it).

    Segments are tokenized separately; every boundary falls next to a protocol
    special token, so this matches tokenizing the whole transcript at once.
    """
    ids: list[int] = []
    mask: list[bool] = []
    for s in segments:
        t = encode_fn(s.text)
        ids += t
        mask += [s.train] * len(t)
    return ids, mask


# ---------------------------------------------------------------- the mixer
@dataclass
class EpisodeMix:
    p_counterfactual: float = 0.25
    p_unanswerable: float = 0.15
    seed: int = 0


class EpisodeBuilder:
    """Turns a stream of Examples into transcripts at the target variant mix.

    Substitutes for counterfactuals come from answers of the same shape seen
    earlier in the stream (other questions, so plausible but wrong).
    """

    def __init__(self, mix: EpisodeMix | None = None, pool_size: int = 5000):
        self.mix = mix or EpisodeMix()
        self.pool: dict[str, list[str]] = {}
        self.pool_size = pool_size
        self.counts: dict[str, int] = {}

    def _remember(self, ans: str | None, rng: random.Random) -> None:
        shape = answer_shape(ans) if ans else None
        if shape is None:
            return
        pool = self.pool.setdefault(shape, [])
        if len(pool) < self.pool_size:
            pool.append(ans)
        else:
            pool[rng.randrange(self.pool_size)] = ans

    def build(self, ex: Example) -> tuple[Example, list[Segment]]:
        rng = random.Random(f"{self.mix.seed}:{ex.id}")
        variant = ex
        if ex.answer is not None:
            r = rng.random()
            if r < self.mix.p_unanswerable:
                variant = unanswerable(ex) or ex
            elif r < self.mix.p_unanswerable + self.mix.p_counterfactual:
                pool = self.pool.get(answer_shape(ex.answer) or "", [])
                for _ in range(5 if pool else 0):
                    cf = counterfactual(ex, rng.choice(pool))
                    if cf:
                        variant = cf
                        break
        self._remember(ex.answer, rng)
        self.counts[variant.kind] = self.counts.get(variant.kind, 0) + 1
        return variant, render(variant, rng)
