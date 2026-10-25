"""scripts/prepare_data.py's worker path, end to end, on a local streaming dataset.

A real ``datasets.IterableDataset`` (built from a generator, no network)
stands in for the Hub, so this covers file sharding across workers, the
stream ``state_dict`` surviving a JSON round trip, and exact resume.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("datasets")
pytest.importorskip("tokenizers")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_data  # noqa: E402

from cirrus.data import sources  # noqa: E402
from cirrus.data.shards import train_files, val_files  # noqa: E402

TOKENIZER = ROOT / "tokenizer" / "cirrus-32k.json"
CRASH = {"after": None, "seen": 0}  # render raises once this many docs have been rendered
OPENED: list = []  # the resume state each stream was opened with


def _render(row):
    CRASH["seen"] += 1
    if CRASH["after"] is not None and CRASH["seen"] > CRASH["after"]:
        raise RuntimeError("killed")
    return row["text"]


@dataclass(frozen=True)
class LocalSource(sources.Source):
    def load(self):
        from datasets import IterableDataset

        def gen(files):
            for f in files:
                for i in range(2000):
                    yield {"text": f"file {f} doc {i}: " + "the quick brown fox " * (i % 5 + 1)}

        return IterableDataset.from_generator(gen, gen_kwargs={"files": list(range(4))})

    def stream(self, rank=0, world=1, state=None):
        OPENED.append(state)
        return super().stream(rank, world, state)


@pytest.fixture
def local_source(monkeypatch):
    if not TOKENIZER.exists():
        pytest.skip("tokenizer/cirrus-32k.json not trained yet")
    monkeypatch.setitem(prepare_data.SOURCES, "local", LocalSource("local", "local/none", _render))
    CRASH.update(after=None, seen=0)
    OPENED.clear()


def _args(out):
    return {"tokenizer": str(TOKENIZER), "out": str(out), "val_tokens": 500, "shard_tokens": 4000, "max_passes": {}}


def _rank_tokens(out, rank):
    files = [f for f in train_files(out) if f.name.startswith(f"{rank:03d}_")]
    return np.concatenate([np.fromfile(f, dtype=np.uint16) for f in files])


def test_worker_writes_disjoint_ranks(local_source, tmp_path):
    r0 = prepare_data.worker("local", 0, 2, 12_000, _args(tmp_path))
    r1 = prepare_data.worker("local", 1, 2, 12_000, _args(tmp_path))
    assert r0["tokens"] >= 12_000 and r1["tokens"] >= 12_000 and not r0["exhausted"]
    assert "local" in val_files(tmp_path)
    state = json.loads((tmp_path / "state" / "local-000.json").read_text())
    assert state["done"] and state["stream_state"] is not None  # a real datasets state, JSON round-tripped

    from cirrus.data.tokenizer import CirrusTokenizer

    tok = CirrusTokenizer(TOKENIZER)
    files0 = {t.split(" doc")[0] for t in tok.decode(_rank_tokens(tmp_path, 0).tolist()).split("<|eos|>") if t}
    files1 = {t.split(" doc")[0] for t in tok.decode(_rank_tokens(tmp_path, 1).tolist()).split("<|eos|>") if t}
    assert files0 and files1 and not (files0 & files1)  # workers own whole files


def test_worker_resumes_from_stream_state(local_source, tmp_path):
    prepare_data.worker("local", 0, 2, 60_000, _args(tmp_path / "clean"))
    clean_read = CRASH["seen"]

    CRASH.update(after=1_500, seen=0)  # dies after two shards are on disk (of ~3,100 docs)
    OPENED.clear()
    with pytest.raises(RuntimeError):
        prepare_data.worker("local", 0, 2, 60_000, _args(tmp_path / "crashed"))
    saved = json.loads((tmp_path / "crashed" / "state" / "local-000.json").read_text())
    assert saved["shards"] > 0 and not saved["done"]

    CRASH.update(after=None, seen=0)
    r = prepare_data.worker("local", 0, 2, 60_000, _args(tmp_path / "crashed"))
    assert OPENED[0] is None and OPENED[-1] == saved["stream_state"]  # reopened at the saved position
    assert CRASH["seen"] == clean_read - saved["docs_in_pass"]  # read only the remainder, nothing twice
    assert r["tokens"] >= 60_000
    assert np.array_equal(_rank_tokens(tmp_path / "clean", 0), _rank_tokens(tmp_path / "crashed", 0))
    assert np.array_equal(np.fromfile(val_files(tmp_path / "clean")["local"], dtype=np.uint16),
                          np.fromfile(val_files(tmp_path / "crashed")["local"], dtype=np.uint16))
