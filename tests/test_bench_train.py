"""bench_train's micro-batch sweep: OOM in both forms, spill detection, results always written."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench_train  # noqa: E402


class _Tiny(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w = torch.nn.Linear(4, 4)

    def cuda(self):  # stays on CPU: the timings are scripted
        return self


def _sweep(monkeypatch, script):
    """Run sweep() with time_steps replaced by ``script``: micro -> row dict or exception."""
    def fake_time_steps(model, opts, micro, *a):
        r = script[micro]
        if isinstance(r, BaseException):
            raise r
        return {"micro": micro, "step_s": 1.0, "loss": 1.0, **r}

    monkeypatch.setattr(bench_train, "time_steps", fake_time_steps)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    args = SimpleNamespace(compile=False, optim="adamw", seq=4096, steps=1, warmup=0, mtp=0.3, micro=sorted(script))
    rows = []
    bench_train.sweep(args, SimpleNamespace(vocab_size=8), _Tiny, rows, 1e9, 165.2, 25.4)
    return rows


def test_wsl_spill_then_accelerator_error(monkeypatch):
    accel = getattr(torch, "AcceleratorError", RuntimeError)
    rows = _sweep(monkeypatch, {
        1: {"tok_s": 15_000, "peak_gb": 7.0, "reserved_gb": 8.0},
        2: {"tok_s": 19_241, "peak_gb": 13.7, "reserved_gb": 15.0},
        4: {"tok_s": 753, "peak_gb": 23.8, "reserved_gb": 24.5},  # spilled to system RAM
        8: accel("CUDA error: out of memory"),
    })
    assert [r["micro"] for r in rows] == [1, 2, 4]  # stops at the first too-big micro-batch
    assert rows[1]["too_big"] is None and "reserved" in rows[2]["too_big"]


def test_speed_collapse_without_memory_signal(monkeypatch):
    rows = _sweep(monkeypatch, {
        1: {"tok_s": 15_000, "peak_gb": 7.0, "reserved_gb": 8.0},
        2: {"tok_s": 5_000, "peak_gb": 13.7, "reserved_gb": 15.0},
    })
    assert "tok/s" in rows[1]["too_big"]


def test_cuda_oom_is_recorded(monkeypatch):
    rows = _sweep(monkeypatch, {
        1: {"tok_s": 15_000, "peak_gb": 7.0, "reserved_gb": 8.0},
        2: torch.cuda.OutOfMemoryError("CUDA out of memory"),
    })
    assert rows[1]["micro"] == 2 and "OutOfMemoryError" in rows[1]["too_big"]


def test_report_is_written_and_best_skips_too_big(monkeypatch, tmp_path):
    report = {}
    rows = [{"micro": 2, "tok_s": 19_241, "compiled": False, "too_big": None},
            {"micro": 4, "tok_s": 753, "compiled": False, "too_big": "spill"}]

    def boom(*a):
        rows_ref = a[3]
        rows_ref.extend(rows)
        raise RuntimeError("crashed mid-sweep")

    monkeypatch.setattr(bench_train, "sweep", boom)
    args = SimpleNamespace(out=str(tmp_path))
    with pytest.raises(RuntimeError):
        bench_train.finish_sweep(args, None, None, report, 1e9, 165.2)
    (path,) = tmp_path.glob("train-*.json")
    saved = json.loads(path.read_text())
    assert saved["best"]["micro"] == 2 and len(saved["runs"]) == 2
