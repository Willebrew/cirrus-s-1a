import asyncio
import random
import re
from collections import Counter
from pathlib import Path

import pytest

from cirrus.agent import protocol as P
from cirrus.agent.backends import SearchHit
from cirrus.agent.citations import audit_answer
from cirrus.agent.loop import Agent, AgentConfig, ScriptedModel
from cirrus.data.episodes import (
    Doc,
    EpisodeBuilder,
    EpisodeMix,
    Segment,
    counterfactual,
    encode,
    from_hotpot,
    from_squad,
    render,
    to_text,
    unanswerable,
)

FILLER = " It is also known for many other things that are described at length in this paragraph."


def hotpot_row(i=0, answer="Paris"):
    titles = [f"Topic {i}-{k}" for k in range(10)]
    sents = [[f"Sentence {k}.0 about {t}.", f" Sentence {k}.1 says nothing." + FILLER] for k, t in enumerate(titles)]
    sents[3] = [f"The capital of Frobia is {answer}.", " It lies on a river." + FILLER * 3]
    sents[7] = ["Frobia is a country in Europe.", " Its people like tea." + FILLER]
    sents[5][1] = sents[5][1] + FILLER * 3 + f" The key fact is that {answer} hosts the Frobian parliament."
    return {"id": f"h{i}", "question": "What is the capital of Frobia?", "answer": answer,
            "supporting_facts": {"title": [titles[3], titles[7], titles[5]], "sent_id": [0, 0, 1]},
            "context": {"title": titles, "sentences": sents}}


def model_text(segs):
    return "".join(s.text for s in segs if s.train)


def test_answerable_transcript_is_valid_protocol():
    ex = from_hotpot(hotpot_row())
    segs = render(ex, random.Random(0))
    text = to_text(segs)
    assert text.startswith(P.render_query(ex.query))
    turn = P.parse_turn(text.split(P.RESULTS_CLOSE)[-1])
    assert turn.is_final and turn.answer.startswith("**Paris** [")
    ids = {int(m) for m in re.findall(r"<\|src (\d+)\|>", text)}
    audit = audit_answer(turn.answer, ids)
    assert audit.ok, audit
    # each quoted evidence line cites the source it came from, and quotes it verbatim
    for line in turn.answer.splitlines()[2:]:
        (sid,) = set(re.findall(r"\[(\d+)\]", line))
        quote = re.sub(r" \[\d+\]", "", line[2:])
        page = text.split(f"<|page {sid}|>")[1].split(P.PAGE_CLOSE)[0] if f"<|page {sid}|>" in text else ""
        snippet = re.search(rf"<\|src {sid}\|>[^\n]*\n(.*?)(?=\n<\|src|\n<\|/results)", text, re.S).group(1)
        assert quote in page or quote in snippet


def test_pages_opened_only_when_snippet_lacks_evidence():
    ex = from_hotpot(hotpot_row())
    segs = render(ex, random.Random(0))
    text = to_text(segs)
    opened = [int(x) for x in re.findall(r"<\|open\|>(\d+)<\|/open\|>", text)]
    # topic 5's evidence sits deep in a long paragraph: it must be opened; topics 3 and 7 lead with it
    by_title = {m.group(2): int(m.group(1)) for m in re.finditer(r"<\|src (\d+)\|>(Topic \S+) \|", text)}
    assert opened == [by_title["Topic 0-5"]]
    assert all(seg.train is False for seg in segs if seg.text.startswith("<|page"))


def test_agent_loop_reproduces_the_transcript():
    """Replaying the model turns through the real agent loop must yield the same text."""
    ex = from_hotpot(hotpot_row())
    segs = render(ex, random.Random(0))
    results = segs[2].text
    hits = [SearchHit(title=m.group(2).split(" | ")[0], url=m.group(2).split(" | ")[1], snippet=m.group(3))
            for m in re.finditer(r"<\|src (\d+)\|>([^\n]*)\n(.*?)(?=\n<\|src|\n<\|/results)", results, re.S)]
    pages = {d.title: d.text for d in ex.docs}

    class Backend:
        def search(self, q, k=8):
            return hits

        def fetch(self, url):
            return pages[url.rsplit("/", 1)[1].replace("_", " ")]

    turns = [s.text for s in segs if s.train]
    ep = asyncio.run(Agent(ScriptedModel(turns), Backend(), AgentConfig(results_per_search=10)).run(ex.query))
    assert ep.transcript == to_text(segs)
    assert ep.audit is not None and ep.audit.ok


def test_counterfactual_swaps_consistently():
    ex = from_hotpot(hotpot_row())
    cf = counterfactual(ex, "Lisbon")
    assert cf.kind == "counterfactual" and cf.answer == "Lisbon"
    assert not any("Paris" in d.text for d in cf.docs)
    assert any("capital of Frobia is Lisbon" in s for d in cf.docs for s in d.support)
    assert "**Lisbon** [" in model_text(render(cf, random.Random(0)))
    assert counterfactual(ex, "Topic") is None  # substitute already present in the documents
    assert counterfactual(from_hotpot(hotpot_row(answer="yes")), "no") is None  # not swappable
    assert counterfactual(ex, r"Ber\1lin").answer == r"Ber\1lin"  # substitutes are literal text


def test_unanswerable_drops_gold_and_abstains():
    un = unanswerable(from_hotpot(hotpot_row()))
    assert un.kind == "unanswerable" and not any(d.gold for d in un.docs)
    segs = render(un, random.Random(0))
    assert segs[-1].text == P.NO_ANSWER and segs[-1].train
    assert "Paris" not in "".join(d.text for d in un.docs if "capital" in d.text)


def test_squad_answerable_and_native_unanswerable():
    distractors = [Doc(f"Other {k}", f"Unrelated paragraph number {k}.") for k in range(4)]
    ctx = "Frobia is old. Its capital is Paris since 1800. It rains."
    row = {"id": "s1", "title": "Frobia", "context": ctx, "question": "What is Frobia's capital?",
           "answers": {"text": ["Paris"], "answer_start": [ctx.index("Paris")]}}
    ex = from_squad(row, distractors)
    assert ex.docs[0].support == ["Its capital is Paris since 1800."]
    assert audit_answer(P.parse_turn(model_text(render(ex, random.Random(1)))).answer, set(range(1, 6))).ok
    un = from_squad({**row, "answers": {"text": [], "answer_start": []}}, distractors)
    assert un.kind == "unanswerable" and render(un, random.Random(1))[-1].text == P.NO_ANSWER


def test_builder_hits_the_target_mix():
    b = EpisodeBuilder(EpisodeMix(p_counterfactual=0.25, p_unanswerable=0.15, seed=1))
    names = ["Paris", "Lisbon", "Oslo", "Rome", "Vienna", "Madrid", "Prague", "Dublin"]
    for i in range(2000):
        b.build(from_hotpot(hotpot_row(i, answer=names[i % len(names)])))
    n = sum(b.counts.values())
    assert abs(b.counts["unanswerable"] / n - 0.15) < 0.03
    assert abs(b.counts["counterfactual"] / n - 0.25) < 0.04
    # deterministic: same seed, same history, same example -> same transcript
    ex = from_hotpot(hotpot_row(5, answer="Rome"))
    twins = [EpisodeBuilder(EpisodeMix(seed=1)) for _ in range(2)]
    for t in twins:
        t.pool = {k: list(v) for k, v in b.pool.items()}
    assert to_text(twins[0].build(ex)[1]) == to_text(twins[1].build(ex)[1])


def test_encode_matches_whole_text_and_masks_harness():
    pytest.importorskip("tokenizers")
    path = Path(__file__).resolve().parents[1] / "tokenizer" / "cirrus-32k.json"
    if not path.exists():
        pytest.skip("tokenizer not trained")
    from cirrus.data.tokenizer import CirrusTokenizer

    tok = CirrusTokenizer(path)
    for ex in (from_hotpot(hotpot_row()), unanswerable(from_hotpot(hotpot_row()))):
        segs = render(ex, random.Random(0))
        ids, mask = encode(segs, tok.encode)
        assert ids == tok.encode(to_text(segs))
        trained = tok.decode([t for t, m in zip(ids, mask) if m])
        assert trained == model_text(segs)
        assert P.RESULTS_OPEN not in trained and P.QUERY_OPEN not in trained


def test_make_episodes_script_offline(tmp_path, monkeypatch):
    import json
    import sys

    datasets = pytest.importorskip("datasets")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import make_episodes

    names = ["Paris", "Lisbon", "Oslo", "Rome", "Vienna"]
    hot = [hotpot_row(i, answer=names[i % 5]) for i in range(60)]
    sq = []
    for i in range(60):
        ctx = f"Article {i} begins here. Its capital is {names[i % 5]} since 18{i:02d}. Rain is common."
        ans = {"text": [names[i % 5]], "answer_start": [ctx.index(names[i % 5])]} if i % 3 else \
            {"text": [], "answer_start": []}
        sq.append({"id": f"s{i}", "title": f"Article_{i}", "context": ctx, "question": "What is the capital?",
                   "answers": ans})

    def fake_load(repo, *a, **kw):
        return iter(hot if "hotpot" in repo else sq)

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    out = tmp_path / "ep.jsonl"
    monkeypatch.setattr(sys, "argv", ["make_episodes.py", "--out", str(out)])
    make_episodes.main()
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    kinds = Counter(r["kind"] for r in rows)
    assert {r["source"] for r in rows} == {"hotpot", "squad"}
    assert kinds["unanswerable"] >= 20 and kinds["answerable"] > 0  # SQuAD's native ones included
    for r in rows:
        assert r["segments"][0][1] is False and r["segments"][-1][1] is True
        assert to_text([Segment(t, m) for t, m in r["segments"]]).startswith(P.QUERY_OPEN)
