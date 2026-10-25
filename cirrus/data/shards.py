"""Token shards: writing them resumably, and reading them as training windows.

On disk a shard is raw little-endian uint16 token ids (the vocab is 32K) with
documents separated by EOS, plus a small JSON sidecar. Layout:

    <out>/train/<source>/<rank>_<k>.bin     ~shard_tokens each, cut at document ends
    <out>/val/<source>.bin                  first val_tokens of the source (rank 0)
    <out>/state/<source>-<rank>.json        resume point for the writer
    <out>/manifest.json                     mix, budgets, tokenizer

Each source writes exactly its share of the token budget, so sampling
windows uniformly over all train shards reproduces the mix.
"""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DTYPE = np.uint16
VAL_BATCH_DOCS = 32  # val is taken in small whole batches, so it overshoots val_tokens by little


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj) -> None:
    _atomic_write_bytes(path, json.dumps(obj, indent=1).encode())


@dataclass
class WriterState:
    docs_in_pass: int = 0  # docs consumed in the current pass
    stream_state: dict | None = None  # exact resume point of the stream, when it supports one
    passes: int = 0
    shards: int = 0
    tokens: int = 0
    docs: int = 0
    val_tokens: int = 0
    val_docs: int = 0  # the first val_docs docs of every pass are held out
    val_stream_state: dict | None = None  # where a new pass starts (just after the val docs)
    done: bool = False
    exhausted: bool = False  # ran out of data before the budget (after max_passes)


def write_source(
    stream: Callable[[dict | None], Iterator[str]],
    encode_batch: Callable[[list[str]], list[list[int]]],
    eos_id: int,
    out: Path,
    name: str,
    rank: int,
    budget: int,
    val_tokens: int = 0,
    shard_tokens: int = 500_000_000,
    max_passes: float = 1,
    batch_docs: int = 512,
    log: Callable[[str], None] = print,
) -> WriterState:
    """Tokenize documents from ``stream`` into shards until ``budget`` train tokens are written.

    ``stream(state)`` opens the source; if the iterator it returns has a
    ``state_dict()`` (a streaming ``datasets`` iterable does), resuming jumps
    straight to the saved position, otherwise it skips already-consumed docs.
    Every decision (val, shard cut, stop) is taken at batch boundaries, so the
    output is identical however many times the writer was killed and rerun;
    at most one shard of work is redone.
    """
    out = Path(out)
    (out / "train" / name).mkdir(parents=True, exist_ok=True)
    (out / "state").mkdir(parents=True, exist_ok=True)
    state_path = out / "state" / f"{name}-{rank:03d}.json"
    st = WriterState(**json.loads(state_path.read_text())) if state_path.exists() else WriterState()
    if st.done:
        return st
    val_path = out / "val" / f"{name}.bin"
    collecting_val = rank == 0 and val_tokens > 0 and not val_path.exists()
    val: list[np.ndarray] = []
    buf: list[np.ndarray] = []
    buf_tokens = buf_docs = 0

    def save() -> None:
        _atomic_write_json(state_path, asdict(st))

    def open_stream():
        it = iter(stream(st.stream_state))
        if st.stream_state is None or not hasattr(it, "state_dict"):
            for _ in range(st.docs_in_pass):
                next(it, None)
        return it

    while not st.done:
        it = open_stream()
        exhausted = False
        while st.tokens < budget:
            bs = min(batch_docs, VAL_BATCH_DOCS) if collecting_val else batch_docs
            batch = list(itertools.islice(it, bs))
            exhausted = len(batch) < bs
            arrs = []
            for ids in encode_batch(batch) if batch else []:
                arr = np.empty(len(ids) + 1, dtype=DTYPE)
                arr[:-1] = ids
                arr[-1] = eos_id
                arrs.append(arr)
            pos = it.state_dict() if hasattr(it, "state_dict") else None
            if collecting_val:  # val takes whole batches
                val += arrs
                if sum(a.size for a in val) >= val_tokens or exhausted:
                    if val:
                        (out / "val").mkdir(exist_ok=True)
                        _atomic_write_bytes(val_path, np.concatenate(val).tobytes())
                    st.val_tokens, st.val_docs = int(sum(a.size for a in val)), len(val)
                    st.docs_in_pass = st.val_docs
                    st.stream_state = st.val_stream_state = pos
                    collecting_val, val = False, []
                    save()
            else:
                buf += arrs
                buf_tokens += sum(a.size for a in arrs)
                buf_docs += len(arrs)
                if buf_tokens >= shard_tokens or st.tokens + buf_tokens >= budget or exhausted:
                    if buf:
                        arr = np.concatenate(buf)
                        _atomic_write_bytes(out / "train" / name / f"{rank:03d}_{st.shards:05d}.bin", arr.tobytes())
                        st.shards += 1
                        st.tokens += int(arr.size)
                        st.docs += buf_docs
                        st.docs_in_pass += buf_docs
                    st.stream_state = pos
                    buf, buf_tokens, buf_docs = [], 0, 0
                    save()
                    log(f"[{name}/{rank}] shard {st.shards}: {st.tokens:,} / {budget:,} tokens, pass {st.passes}")
            if exhausted:
                break
        if st.tokens >= budget:
            st.done = True
        elif exhausted:
            st.passes += 1
            if st.passes >= max_passes:
                st.done = st.exhausted = True
                log(f"[{name}/{rank}] data exhausted after {st.passes} passes at {st.tokens:,} tokens")
            else:  # next pass starts just after the held-out val docs
                st.docs_in_pass, st.stream_state = st.val_docs, st.val_stream_state
        save()
    return st


# ---------------------------------------------------------------- reading
class TokenWindows:
    """Fixed-length windows over a set of shards, in a seeded shuffled order.

    Windows never straddle files. Sample ``i`` of epoch ``e`` is a pure
    function of (seed, e, i), so resuming needs only the sample counter.
    """

    def __init__(self, files: Iterable[str | Path], seq_len: int, seed: int = 0):
        self.files = sorted(str(f) for f in files)
        if not self.files:
            raise FileNotFoundError("no shards")
        self.seq_len = seq_len
        self.seed = seed
        self.maps = [np.memmap(f, dtype=DTYPE, mode="r") for f in self.files]
        per = np.array([(m.size - 1) // seq_len for m in self.maps], dtype=np.int64)
        self.cum = np.concatenate([[0], np.cumsum(per)])
        self.n = int(self.cum[-1])
        if self.n == 0:
            raise ValueError(f"shards hold fewer than {seq_len + 1} tokens")
        self.tokens = int(sum(m.size for m in self.maps))
        self._perm_epoch, self._perm = -1, None

    def __len__(self) -> int:
        return self.n

    def window(self, j: int) -> np.ndarray:
        """Window j in file order: seq_len + 1 tokens (inputs and shifted labels)."""
        f = int(np.searchsorted(self.cum, j, side="right") - 1)
        off = (j - int(self.cum[f])) * self.seq_len
        return np.asarray(self.maps[f][off : off + self.seq_len + 1], dtype=np.int64)

    def _order(self, epoch: int) -> np.ndarray:
        if epoch != self._perm_epoch:
            rng = np.random.default_rng([self.seed, epoch])
            perm = rng.permutation(self.n)
            self._perm = perm.astype(np.uint32) if self.n < 2**32 else perm
            self._perm_epoch = epoch
        return self._perm

    def sample(self, i: int) -> np.ndarray:
        epoch, k = divmod(i, self.n)
        return self.window(int(self._order(epoch)[k]))

    def batch(self, start: int, size: int) -> np.ndarray:
        return np.stack([self.sample(start + b) for b in range(size)])


def train_files(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("train/*/*.bin"))


def val_files(root: str | Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(Path(root).glob("val/*.bin"))}
