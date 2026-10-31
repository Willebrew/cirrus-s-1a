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


def test_parquet_source_reads_bounded_batches_and_resumes(tmp_path, monkeypatch):
    """A hub-style parquet repo (YAML configs, one big row group): only the rendered columns
    are read, in PARQUET_BATCH_ROWS-row batches, and a saved state resumes mid row group."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    repo = tmp_path / "repo"
    (repo / "faq").mkdir(parents=True)
    (repo / "README.md").write_text("---\nconfigs:\n- config_name: faq\n  data_files:\n  - split: train\n"
                                    "    path: faq/*.parquet\n---\n")
    n = 5000
    table = pa.table({
        "text": [f"document {i}" for i in range(n)],
        "rollout_results": [[{"text": f"Q{i}? A{i}.", "finish_reason": "stop"}] for i in range(n)],
        "unused_blob": ["x" * 2000] * n,  # must never be read
    })
    pq.write_table(table, repo / "faq" / "part0.parquet", row_group_size=n)  # one row group

    seen_kwargs = {}
    import datasets

    real = datasets.load_dataset

    def spy(*a, **kw):
        seen_kwargs.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(datasets, "load_dataset", spy)
    monkeypatch.setattr(sources, "PARQUET_BATCH_ROWS", 256)
    src = sources.Source("t", str(repo), sources.render_finephrase("FAQ"), config="faq",
                         columns=("text", "rollout_results"))
    it = src.stream()
    first = [next(it) for _ in range(700)]
    assert seen_kwargs["columns"] == ["text", "rollout_results"] and seen_kwargs["batch_size"] == 256
    assert first[0] == "document 0\n\nFAQ\n\nQ0? A0." and first[-1].startswith("document 699\n")
    state = json.loads(json.dumps(it.state_dict()))
    resumed = src.stream(state=state)
    assert next(resumed).startswith("document 700\n")
    assert "unused_blob" not in src.load().column_names


def test_resume_keeps_the_original_slice_count(tmp_path, capsys):
    m = {"mix": "lean", "budgets": {"a": 10}, "tokenizer_hash": "h", "workers": {"a": 32}}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps({**m, "written": {"a": 3}}))
    merged = prepare_data.reconcile_manifest(mpath, {**m, "workers": {"a": 10}})
    assert merged["workers"] == {"a": 32} and merged["written"] == {"a": 3}
    assert "keeping that split" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        prepare_data.reconcile_manifest(mpath, {**m, "budgets": {"a": 11}})
    assert prepare_data.reconcile_manifest(tmp_path / "none.json", m) is m


def test_run_slices_respects_memory_and_reports_failures(tmp_path):
    import slice_stub  # light module: each spawned child imports it

    tasks = [("big", 0, 2, 10), ("big", 1, 2, 10), ("small", 0, 2, 5), ("small", 1, 2, 5), ("crash", 0, 1, 7)]
    mem = {"big": 3.0, "small": 1.0, "crash": 1.0}
    logs = []
    done, failed = prepare_data.run_slices(tasks, {"out": str(tmp_path)}, mem.get, max_mem_gb=4.0, max_procs=8,
                                           target=slice_stub.fake_slice, log=logs.append)
    assert sorted((r["source"], r["rank"]) for r in done) == [("big", 0), ("big", 1), ("small", 0), ("small", 1)]
    assert [(f["source"], f["exitcode"]) for f in failed] == [("crash", 3)]
    assert any(m.startswith("FAILED crash/0") for m in logs)
    spans = [(n, float(a), float(b)) for n, a, b in (ln.split() for ln in (tmp_path / "intervals.txt").read_text()
                                                       .splitlines())]
    for t in sorted({s for _, s, _ in spans} | {e - 1e-6 for _, _, e in spans}):
        live = sum(mem[n] for n, s, e in spans if s <= t < e)
        assert live <= 4.0  # never more than the budget at once
    bigs = [(s, e) for n, s, e in spans if n == "big"]
    assert bigs[0][1] <= bigs[1][0] or bigs[1][1] <= bigs[0][0]  # the two 3 GB slices never overlapped


def test_real_slice_entry_reports_a_crash(tmp_path):
    a = {"tokenizer": str(TOKENIZER), "out": str(tmp_path), "val_tokens": 0, "shard_tokens": 1000, "max_passes": {}}
    done, failed = prepare_data.run_slices([("no_such_source", 0, 1, 10)], a, lambda n: 1.0, 4.0, 1,
                                           log=lambda m: None)
    assert not done and failed[0]["exitcode"] == 1
