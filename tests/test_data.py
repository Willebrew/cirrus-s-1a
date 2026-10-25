import json

import numpy as np
import pytest

from cirrus.data.shards import TokenWindows, train_files, val_files, write_source
from cirrus.data.sources import MIXES, SOURCES, render_finephrase, render_synth

def ENC(batch):
    return [[ord(c) for c in d] for d in batch]


EOS = 2
DOCS = [f"doc {i} " + "x" * (i % 7) for i in range(1000)]


def _all_tokens(root):
    return np.concatenate([np.fromfile(f, dtype=np.uint16) for f in train_files(root)])


def _docs(arr):
    return [bytes(d.astype(np.uint8)).decode() for d in np.split(arr, np.where(arr == EOS)[0] + 1) if d.size]


def test_write_budget_val_and_no_overlap(tmp_path):
    st = write_source(lambda s: iter(DOCS), ENC, EOS, tmp_path, "s", 0, budget=5000, val_tokens=300,
                      shard_tokens=1000, batch_docs=64, log=lambda m: None)
    assert st.done and 5000 <= st.tokens < 5000 + 64 * 20
    val = np.fromfile(val_files(tmp_path)["s"], dtype=np.uint16)
    train_docs = {d.rstrip(chr(EOS)) for d in _docs(_all_tokens(tmp_path))}
    val_docs = {d.rstrip(chr(EOS)) for d in _docs(val)}
    assert val.size >= 300 and not (train_docs & val_docs)
    # rerunning a finished source is a no-op
    again = write_source(lambda s: iter(DOCS), ENC, EOS, tmp_path, "s", 0, budget=5000, log=lambda m: None)
    assert again.tokens == st.tokens


def test_resume_after_crash_matches_clean_run(tmp_path):
    kw = dict(budget=6000, val_tokens=200, shard_tokens=700, batch_docs=32, log=lambda m: None)
    clean = tmp_path / "clean"
    write_source(lambda s: iter(DOCS), ENC, EOS, clean, "s", 0, **kw)

    crashed = tmp_path / "crashed"

    def flaky(limit):
        def gen(state):
            for i, d in enumerate(DOCS):
                if i == limit:
                    raise RuntimeError("killed")
                yield d
        return gen

    for limit in (40, 170, 333):  # crash before val, mid-shard, later
        with pytest.raises(RuntimeError):
            write_source(flaky(limit), ENC, EOS, crashed, "s", 0, **kw)
    write_source(lambda s: iter(DOCS), ENC, EOS, crashed, "s", 0, **kw)
    assert np.array_equal(_all_tokens(clean), _all_tokens(crashed))
    assert np.array_equal(np.fromfile(val_files(clean)["s"], dtype=np.uint16),
                          np.fromfile(val_files(crashed)["s"], dtype=np.uint16))


def test_multi_pass_skips_val_and_stops(tmp_path):
    st = write_source(lambda s: iter(DOCS[:50]), ENC, EOS, tmp_path, "s", 0, budget=10**6, val_tokens=50,
                      shard_tokens=10**6, max_passes=2, batch_docs=8, log=lambda m: None)
    assert st.exhausted and st.passes == 2
    val_docs = set(_docs(np.fromfile(val_files(tmp_path)["s"], dtype=np.uint16)))
    train = _docs(_all_tokens(tmp_path))
    assert not (set(train) & val_docs)
    assert len(train) == 2 * (50 - st.val_docs)
    state = json.loads((tmp_path / "state" / "s-000.json").read_text())
    assert state["done"]


def test_windows_deterministic_and_cover_epoch(tmp_path):
    write_source(lambda s: iter(DOCS), ENC, EOS, tmp_path, "s", 0, budget=5000, shard_tokens=900, log=lambda m: None)
    a = TokenWindows(train_files(tmp_path), 32, seed=3)
    b = TokenWindows(train_files(tmp_path), 32, seed=3)
    assert np.array_equal(a.batch(5, 4), b.batch(5, 4))
    assert a.batch(0, 1).shape == (1, 33)
    epoch = {a.sample(i).tobytes() for i in range(len(a))}
    every = {a.window(j).tobytes() for j in range(len(a))}
    assert epoch == every  # one epoch visits every window exactly once
    assert not np.array_equal(a.sample(0), a.sample(len(a)))  # next epoch reshuffles (w.h.p.)


def test_renderers():
    row = {"text": "Doc body.", "rollout_results": [{"text": "Q: a?\nA: b.", "finish_reason": "stop"}]}
    assert render_finephrase("FAQ")(row) == "Doc body.\n\nFAQ\n\nQ: a?\nA: b."
    assert render_finephrase("FAQ")({**row, "rollout_results": [{"text": "x", "finish_reason": "length"}]}) is None
    srow = {"language": "en", "query": "Who?", "synthetic_answer": "Her.", "query_seed_text": "Seed.",
            "synthetic_reasoning": "hmm"}
    assert render_synth()(srow) == "Source:\nSeed.\n\nQuestion: Who?\n\nAnswer:\nHer."
    assert render_synth()({**srow, "language": "fr"}) is None
    for mix in MIXES.values():
        assert set(mix.weights) <= set(SOURCES)
        assert abs(sum(mix.budgets(1e9).values()) - 1e9) < 10


def test_tokenizer_reorder_keeps_tokenization(tmp_path):
    pytest.importorskip("tokenizers")
    import random

    from cirrus.data.tokenizer import N_RESERVED, CirrusTokenizer, reorder_by_frequency, train

    random.seed(0)
    words = "the of search model answer source price costs $899 tokens query result page".split()
    texts = [" ".join(random.choice(words) for _ in range(40)) + " <|search|>e-bike 2026<|/search|><|run|>"
             for _ in range(2000)]
    tok = train(texts, vocab_size=500)
    tok2 = reorder_by_frequency(tok, texts[:300])
    s = "The bike costs $1,299 in 2026 <|src 3|> <|answer|>hi<|/answer|>"
    assert tok.encode(s).tokens == tok2.encode(s).tokens
    tok2.save(str(tmp_path / "t.json"))
    t = CirrusTokenizer(tmp_path / "t.json")
    assert t.decode(t.encode(s)) == s
    assert t.token_id("<|run|>") < N_RESERVED
    assert [t.tok.id_to_token(i) for i in t.encode("2026")] == ["2", "0", "2", "6"]  # digits split
    counts = np.bincount(np.concatenate([t.encode(x) for x in texts[:300]]), minlength=t.vocab_size)
    body = counts[N_RESERVED:]
    assert (np.diff(body[body > 0]) <= 0).all()  # ids after the specials are frequency ordered


class _Resumable:
    """A stream that resumes from a saved position, like a streaming ``datasets`` iterable."""

    opened_at: list = []

    def __init__(self, state, crash_at=None):
        self.i = state["i"] if state else 0
        self.crash_at = crash_at
        _Resumable.opened_at.append(self.i)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.crash_at:
            raise RuntimeError("killed")
        if self.i >= len(DOCS):
            raise StopIteration
        self.i += 1
        return DOCS[self.i - 1]

    def state_dict(self):
        return {"i": self.i}


def test_resume_uses_stream_state(tmp_path):
    kw = dict(budget=6000, val_tokens=200, shard_tokens=700, batch_docs=32, log=lambda m: None)
    write_source(lambda s: _Resumable(s), ENC, EOS, tmp_path / "clean", "s", 0, **kw)
    _Resumable.opened_at = []
    with pytest.raises(RuntimeError):
        write_source(lambda s: _Resumable(s, crash_at=300), ENC, EOS, tmp_path / "crashed", "s", 0, **kw)
    write_source(lambda s: _Resumable(s), ENC, EOS, tmp_path / "crashed", "s", 0, **kw)
    assert _Resumable.opened_at[0] == 0 and _Resumable.opened_at[1] > 0  # jumped, did not re-read
    assert np.array_equal(_all_tokens(tmp_path / "clean"), _all_tokens(tmp_path / "crashed"))
