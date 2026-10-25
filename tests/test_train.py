"""Trainer smoke tests on the debug config (CPU). Skipped without torch."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train as trainer  # noqa: E402

from cirrus.data.shards import write_source  # noqa: E402
from cirrus.train.optim import Muon, newton_schulz, param_groups, wsd_lr  # noqa: E402


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """A learnable toy corpus: repeated 'sentences' over a 512-token vocab."""
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    phrases = [list(rng.integers(3, 512, size=rng.integers(5, 12))) for _ in range(40)]
    docs = [sum((phrases[i] for i in rng.integers(0, 40, size=20)), []) for _ in range(400)]
    enc = lambda batch: batch  # noqa: E731  docs are already token lists
    write_source(lambda s: iter(docs), enc, 2, root, "toy", 0, budget=60_000, val_tokens=2_000,
                 shard_tokens=20_000, batch_docs=16, log=lambda m: None)
    return root


def run(data, out, steps, extra=()):
    return trainer.main(["--data", str(data), "--out", str(out), "--config", "cirrus-s-1a-debug",
                         "--tokens", str(64 * 4 * 50), "--seq", "64", "--batch-tokens", str(64 * 4), "--micro", "2",
                         "--warmup", "5", "--lr", "3e-3", "--eval-every", "1000", "--eval-windows", "4",
                         "--log-every", "1", "--max-steps", str(steps), "--device", "cpu", *extra])


def test_loss_decreases_and_resume_is_exact(data, tmp_path):
    straight = run(data, tmp_path / "a", 12)
    first = json.loads((tmp_path / "a" / "log.jsonl").read_text().splitlines()[0])
    assert straight["loss"] < first["loss"] - 0.3, (first["loss"], straight["loss"])
    assert "val" in straight and np.isfinite(straight["val"]["mean"])

    run(data, tmp_path / "b", 5)  # stop early, then resume the same command
    resumed = run(data, tmp_path / "b", 12)
    sa = torch.load(trainer.latest_ckpt(tmp_path / "a"), weights_only=False)["model"]
    sb = torch.load(trainer.latest_ckpt(tmp_path / "b"), weights_only=False)["model"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert resumed["loss"] == straight["loss"]


def test_resume_rejects_changed_schedule(data, tmp_path):
    run(data, tmp_path / "c", 2)
    with pytest.raises(SystemExit):
        trainer.main(["--data", str(data), "--out", str(tmp_path / "c"), "--config", "cirrus-s-1a-debug",
                      "--tokens", "99999", "--seq", "64", "--batch-tokens", "256", "--micro", "2", "--device", "cpu"])


def test_adamw_option_runs(data, tmp_path):
    m = run(data, tmp_path / "d", 3, ["--optim", "adamw"])
    assert np.isfinite(m["loss"])


def test_optim_pieces():
    torch.manual_seed(0)
    G = torch.randn(48, 32)
    X = newton_schulz(G).float()
    sv = torch.linalg.svdvals(X)
    assert 0.5 < sv.min() and sv.max() < 1.3  # roughly orthogonal

    from cirrus.config import get_config
    from cirrus.model.model import CirrusModel

    model = CirrusModel(get_config("cirrus-s-1a-debug"))
    muon, decay, no_decay = param_groups(model)
    assert model.embed.weight in decay and all(p.ndim == 2 for p in muon)
    assert len(muon) + len(decay) + len(no_decay) == len(list(model.parameters()))
    opt = Muon(muon, lr=1e-2)
    for p in muon:
        p.grad = torch.randn_like(p)
    opt.step()

    assert wsd_lr(0, 100, 10, 0.2) == pytest.approx(0.1)
    assert wsd_lr(50, 100, 10, 0.2) == 1.0
    assert wsd_lr(99, 100, 10, 0.2) < 0.25 and wsd_lr(80, 100, 10, 0.2) == 1.0
